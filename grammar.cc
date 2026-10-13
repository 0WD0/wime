/* wime grammar module: expose the rime-octagram collocation score to Elisp.

   This wraps only the language model, not the Rime input engine.  The module
   is loaded by wime.el and queried during sentence scoring. */

#include <emacs-module.h>

#include <rime/config.h>
#include <rime_api.h>
#include <octagram.h>

#include <string>
#include <vector>

int plugin_is_GPL_compatible;

static rime::OctagramComponent *grammar_component = nullptr;
static rime::Octagram *grammar = nullptr;

static std::string get_string(emacs_env *env, emacs_value arg) {
  ptrdiff_t size = 0;
  env->copy_string_contents(env, arg, nullptr, &size);
  std::vector<char> buffer(size);
  env->copy_string_contents(env, arg, buffer.data(), &size);
  return std::string(buffer.data());
}

static emacs_value symbol(emacs_env *env, const char *name) {
  return env->intern(env, name);
}

static emacs_value Fopen(emacs_env *env, ptrdiff_t nargs, emacs_value *args,
                         void *data) noexcept {
  std::string shared_data_dir = get_string(env, args[0]);
  std::string language = get_string(env, args[1]);

  if (!grammar) {
    RimeApi *api = rime_get_api();
    if (!api) {
      return env->intern(env, "nil");
    }
    RIME_STRUCT(RimeTraits, traits);
    traits.shared_data_dir = shared_data_dir.c_str();
    traits.app_name = "rime.wime";
    traits.distribution_name = "Rime";
    traits.distribution_code_name = "wime-grammar";
    traits.distribution_version = "1.0";
    api->setup(&traits);
    api->initialize(&traits);

    auto *config = new rime::Config();
    config->SetString("grammar/language", language.c_str());
    grammar_component = new rime::OctagramComponent();
    grammar = grammar_component->Create(config);
  }
  return grammar ? env->intern(env, "t") : env->intern(env, "nil");
}

static emacs_value Fquery(emacs_env *env, ptrdiff_t nargs, emacs_value *args,
                          void *data) noexcept {
  if (!grammar) {
    return env->make_float(env, 0.0);
  }
  std::string context = get_string(env, args[0]);
  std::string word = get_string(env, args[1]);
  bool is_rear = env->is_not_nil(env, args[2]);
  return env->make_float(env, grammar->Query(context, word, is_rear));
}

static emacs_value Floaded(emacs_env *env, ptrdiff_t nargs, emacs_value *args,
                           void *data) noexcept {
  return grammar ? env->intern(env, "t") : env->intern(env, "nil");
}

static void defun(emacs_env *env, emacs_value (*function)(emacs_env *, ptrdiff_t,
                                                          emacs_value *,
                                                          void *) noexcept,
                  const char *name, ptrdiff_t min, ptrdiff_t max) {
  emacs_value func = env->make_function(env, min, max, function, "", nullptr);
  emacs_value args[2] = {symbol(env, name), func};
  env->funcall(env, symbol(env, "defalias"), 2, args);
}

int emacs_module_init(struct emacs_runtime *runtime) {
  if (runtime->size < sizeof(*runtime)) {
    return 1;
  }
  emacs_env *env = runtime->get_environment(runtime);
  defun(env, Fopen, "wime-grammar-open", 2, 2);
  defun(env, Fquery, "wime-grammar-query", 3, 3);
  defun(env, Floaded, "wime-grammar-loaded-p", 0, 0);
  return 0;
}
