;;; wime-backend.el --- Wime backend configuration and selection -*- lexical-binding: t; -*-

;;; Commentary:
;; Backend identity, capabilities and asset choices belong to Wime.  The
;; neural worker consumes this configuration; checkpoints provide the warm
;; transition and persistence when a backend is selected interactively.

;;; Code:

(require 'wime)

(defvar wime-checkpoint-directory)
(defvar wime-neural-model-file)
(declare-function wime-checkpoint-profile "wime-checkpoint" ())
(declare-function wime-checkpoint-call "wime-checkpoint" (request))
(declare-function wime-checkpoint-switch "wime-checkpoint" (recipe persist resolve reject))

(defcustom wime-backend 'qwen-hf
  "Wime inference backend.
`qwen-hf' and `minicpm-native' use byte-constrained text decoding, with
conditional guidance and language-model ranking.  `legacy' uses the
original character-synchronous search and fused ranking with Qwen HF.
Use `wime-select-backend' to prewarm and persist a backend change."
  :type '(choice (const qwen-hf) (const minicpm-native) (const legacy))
  :group 'wime)

(defcustom wime-backend-native-library nil
  "Wime llama bridge shared library for `minicpm-native'."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-backend-native-model nil
  "MiniCPM BF16 GGUF matching the configured rescorer tokenizer."
  :type '(choice (const nil) file) :group 'wime)

(defconst wime-backend--definitions
  '((qwen-hf :capabilities (chinese japanese rerank verify decode)
             :rescorer "rescorers/Qwen3.5-2B-Base")
    (minicpm-native :capabilities (chinese decode)
                    :rescorer "rescorers/MiniCPM5-2B" :native t)
    (legacy :capabilities (chinese japanese decode)
            :rescorer "rescorers/Qwen3.5-2B-Base"))
  "Backend capabilities and default assets relative to the data directory.")

(defun wime-backend-supports-p (capability &optional backend)
  "Whether BACKEND (default `wime-backend') supports CAPABILITY."
  (memq capability (plist-get (cdr (assq (or backend wime-backend)
                                         wime-backend--definitions)) :capabilities)))

(defun wime-backend-from-profile (profile)
  "Read and validate PROFILE's explicit backend."
  (let ((name (plist-get profile :backend)))
    (or (car (seq-find (lambda (entry) (equal name (symbol-name (car entry))))
                       wime-backend--definitions))
        (error "Unknown Wime backend: %s" name))))

(defun wime-backend--arguments (search-method rescorer database)
  "Return backend worker arguments for SEARCH-METHOD, RESCORER and DATABASE.
Keep the default verification protocol usable without decoder assets."
  (unless (wime-backend-supports-p search-method)
    (error "Backend %s does not support %s search" wime-backend search-method))
  (when (or (and rescorer (eq search-method 'decode))
            (not (eq wime-backend 'qwen-hf)))
    (unless (and rescorer (eq search-method 'decode) database)
      (error "Selected backend requires a rescorer, database and decode search"))
    (append (list "--database" (expand-file-name database)
                  "--backend" (symbol-name wime-backend))
            (when (plist-get (cdr (assq wime-backend wime-backend--definitions)) :native)
              (unless (and wime-backend-native-library wime-backend-native-model)
                (error "MiniCPM requires a native library and GGUF model"))
              (list "--native-library" (expand-file-name wime-backend-native-library)
                    "--native-model" (expand-file-name wime-backend-native-model))))))

(defvar wime-backend--cancel-switch nil)

(defun wime-select-backend (backend)
  "Prewarm BACKEND and save it as the next-start configuration.
The current worker remains in use until the replacement is ready and all
compositions have finished.  Assets live in `wime-checkpoint-directory'."
  (interactive (list (intern (completing-read "Wime backend: "
                                             (mapcar (lambda (entry) (symbol-name (car entry)))
                                                     wime-backend--definitions) nil t))))
  (unless (assq backend wime-backend--definitions) (user-error "Unknown Wime backend"))
  (require 'wime-checkpoint)
  (when (and (not (wime-backend-supports-p 'japanese backend))
             (cl-some (lambda (buffer)
                        (with-current-buffer buffer
                          (and current-input-method
                               (string-prefix-p "wime-japanese" current-input-method))))
                      (buffer-list)))
    (user-error "Deactivate Japanese input before selecting the Chinese-only MiniCPM backend"))
  (let* ((definition (cdr (assq backend wime-backend--definitions)))
         (native (plist-get definition :native))
         (rescorer (expand-file-name (plist-get definition :rescorer)
                                    wime-checkpoint-directory))
         (library (and native (expand-file-name "native/current/lib/libwime_llama.so" wime-checkpoint-directory)))
         (gguf (and native (expand-file-name "model.bf16.gguf" rescorer)))
         (profile (wime-checkpoint-profile)))
    (dolist (path (append (list wime-neural-model-file wime-database-file
                                (expand-file-name "config.json" rescorer)
                                (expand-file-name "tokenizer.json" rescorer))
                         (and native (list library gguf))))
      (unless (and path (file-readable-p path))
        (user-error "Backend asset is missing: %s" path)))
    (setq profile (plist-put profile :backend (symbol-name backend))
          profile (plist-put profile :search_method "decode")
          profile (plist-put profile :rescorer rescorer)
          ;; Freeze the library version in saved recipes, not its current symlink.
          profile (plist-put profile :native_library (if library (file-truename library) :null))
          profile (plist-put profile :native_model (or gguf :null)))
    (let* ((checkpoint (plist-get (wime-checkpoint-call (list :op "import-model" :path wime-neural-model-file))
                                  :result))
           (recipe (list :checkpoint (plist-get checkpoint :id) :path (plist-get checkpoint :path)
                         :profile profile))
           finished)
      (when wime-backend--cancel-switch (funcall wime-backend--cancel-switch))
      (setq wime-backend--cancel-switch
            (wime-checkpoint-switch
             recipe t
             (lambda (_result) (setq finished t wime-backend--cancel-switch nil)
               (message "Wime backend: %s (saved for startup)" backend))
             (lambda (reason) (setq finished t wime-backend--cancel-switch nil)
               (message "Wime kept the current backend: %s" reason))))
      ;; Command construction can reject a recipe synchronously.
      (when finished (setq wime-backend--cancel-switch nil)))))

(provide 'wime-backend)
;;; wime-backend.el ends here
