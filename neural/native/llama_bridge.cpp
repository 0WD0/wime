// Thin C ABI for Wime's external beam search. No sampler or grammar is used.
#include "llama.h"
#include <algorithm>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
std::once_flag initialized;
struct Model { llama_model * value; int32_t vocab; };
struct Context { llama_context * value; Model * model; int32_t batch, sequences; };
void fail(const char * text) { throw std::runtime_error(text); }
}

extern "C" {
const char * wime_llama_error() { return error.c_str(); }
int32_t wime_llama_abi_version() { return 1; }
int32_t wime_llama_max_sequences() { return static_cast<int32_t>(llama_max_parallel_sequences()); }

void * wime_llama_load(const char * path, int32_t gpu_layers) {
    try {
        std::call_once(initialized, [] { llama_backend_init(); });
        auto params = llama_model_default_params();
        params.n_gpu_layers = gpu_layers;
        auto * model = llama_model_load_from_file(path, params);
        if (!model) fail("llama_model_load_from_file failed");
        char architecture[64] = {};
        llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture));
        if (std::strcmp(architecture, "llama") != 0 || llama_model_is_recurrent(model) || llama_model_is_hybrid(model)) {
            llama_model_free(model);
            fail("Wime native bridge currently supports standard Llama models only");
        }
        return new Model{model, llama_vocab_n_tokens(llama_model_get_vocab(model))};
    } catch (const std::exception & e) { error = e.what(); return nullptr; }
}

void wime_llama_free_model(void * handle) {
    if (handle) { auto * model = static_cast<Model *>(handle); llama_model_free(model->value); delete model; }
}
int32_t wime_llama_vocab_size(void * handle) { return static_cast<Model *>(handle)->vocab; }

int32_t wime_llama_token_piece(void * handle, int32_t token, char * buffer, int32_t size) {
    auto * model = static_cast<Model *>(handle);
    if (token < 0 || token >= model->vocab) return 0;
    // HF's serialized added-token flags remain authoritative. Some unused
    // spellings may be classified as control tokens by the GGUF converter.
    return llama_token_to_piece(llama_model_get_vocab(model->value), token, buffer, size, 0, true);
}

void * wime_llama_create_context(void * handle, int32_t size, int32_t batch, int32_t sequences, int32_t threads) {
    try {
        if (size < 1 || batch < 1 || sequences < 1 || threads < 1) fail("invalid context dimensions");
        auto * model = static_cast<Model *>(handle);
        auto params = llama_context_default_params();
        params.n_ctx = size;
        params.n_batch = batch;
        params.n_ubatch = batch;
        params.n_seq_max = sequences;
        params.n_threads = params.n_threads_batch = threads;
        // Unified KV makes seq_cp share cell ownership, not copy KV tensors.
        params.kv_unified = true;
        // Match the reference Transformers BF16 KV, not llama.cpp's F16 default.
        params.type_k = params.type_v = GGML_TYPE_BF16;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        auto * ctx = llama_init_from_model(model->value, params);
        if (!ctx) fail("llama_init_from_model failed");
        return new Context{ctx, model, static_cast<int32_t>(llama_n_batch(ctx)),
                           static_cast<int32_t>(llama_n_seq_max(ctx))};
    } catch (const std::exception & e) { error = e.what(); return nullptr; }
}

void wime_llama_free_context(void * handle) {
    if (handle) { auto * ctx = static_cast<Context *>(handle); llama_free(ctx->value); delete ctx; }
}
void wime_llama_clear(void * handle) {
    llama_memory_clear(llama_get_memory(static_cast<Context *>(handle)->value), false);
}
int32_t wime_llama_seq_copy(void * handle, int32_t source, int32_t destination, int32_t end) {
    try {
        auto * ctx = static_cast<Context *>(handle);
        if (source < 0 || source >= ctx->sequences || destination < 0 || destination >= ctx->sequences || end < 0)
            fail("invalid sequence copy");
        llama_memory_seq_cp(llama_get_memory(ctx->value), source, destination, 0, end);
        return 0;
    } catch (const std::exception & e) { error = e.what(); return -1; }
}
int32_t wime_llama_seq_remove(void * handle, int32_t sequence) {
    auto * ctx = static_cast<Context *>(handle);
    if (sequence < 0 || sequence >= ctx->sequences) { error = "invalid sequence removal"; return -1; }
    if (!llama_memory_seq_rm(llama_get_memory(ctx->value), sequence, -1, -1)) {
        error = "llama_memory_seq_rm failed"; return -1;
    }
    return 0;
}

// Each token belongs to one sequence. Multiple sequences/positions can share
// one call; positions are explicit, and no padding token enters the KV cache.
// Output rows follow flagged input positions and contain raw full-vocab logits.
int32_t wime_llama_decode(void * handle, int32_t count, const int32_t * tokens,
                         const int32_t * positions, const int32_t * sequences,
                         const int32_t * outputs, float * logits, int64_t capacity) {
    llama_batch batch = {};
    bool allocated = false;
    try {
        auto * ctx = static_cast<Context *>(handle);
        if (count < 1 || count > ctx->batch) fail("decode batch exceeds configured capacity");
        int64_t rows = 0;
        for (int32_t i = 0; i < count; ++i) {
            if (tokens[i] < 0 || tokens[i] >= ctx->model->vocab || positions[i] < 0 ||
                sequences[i] < 0 || sequences[i] >= ctx->sequences) fail("invalid decode token, position or sequence");
            rows += outputs[i] != 0;
        }
        if (capacity < rows * ctx->model->vocab) fail("logits output buffer is too small");
        batch = llama_batch_init(count, 0, 1);
        allocated = true;
        batch.n_tokens = count;
        for (int32_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = positions[i];
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = sequences[i];
            batch.logits[i] = outputs[i] != 0;
        }
        const int result = llama_decode(ctx->value, batch);
        if (result != 0) {
            error = "llama_decode failed (code " + std::to_string(result) + "; check KV capacity)";
            llama_batch_free(batch);
            return -1;
        }
        int64_t row = 0;
        for (int32_t i = 0; i < count; ++i) if (outputs[i]) {
            const float * values = llama_get_logits_ith(ctx->value, i);
            if (!values) fail("llama_get_logits_ith returned no requested row");
            std::memcpy(logits + row++ * ctx->model->vocab, values, sizeof(float) * ctx->model->vocab);
        }
        llama_batch_free(batch);
        return 0;
    } catch (const std::exception & e) {
        if (allocated) llama_batch_free(batch);
        error = e.what(); return -1;
    }
}
}
