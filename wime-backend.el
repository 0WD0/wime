;;; wime-backend.el --- Wime backend configuration and selection -*- lexical-binding: t; -*-

;;; Commentary:
;; Wime selects a complete inference setup.  Worker profiles keep search
;; strategy and model implementation separate: verify uses the existing
;; conditional protocol without loading a general model.  Checkpoints own
;; the warm transition and persistence.

;;; Code:

(require 'wime)

(defvar wime-checkpoint-directory)
(defvar wime-neural-model-file)
(declare-function wime-checkpoint-profile "wime-checkpoint" ())
(declare-function wime-checkpoint-call "wime-checkpoint" (request))
(declare-function wime-checkpoint-switch "wime-checkpoint" (recipe persist resolve reject))

(defcustom wime-backend 'qwen-hf
  "Wime inference backend.
`verify' checks dictionary candidates with the conditional model alone.
`qwen-hf' and `minicpm-native' use byte-constrained text decoding, with
conditional guidance and language-model ranking.  `legacy' uses the
original character-synchronous search and fused ranking with Qwen HF.
Use `wime-select-backend' to prewarm and persist a backend change."
  :type '(choice (const verify) (const qwen-hf) (const minicpm-native) (const legacy))
  :group 'wime)

(defcustom wime-backend-native-library nil
  "Wime llama bridge shared library for `minicpm-native'."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-backend-native-model nil
  "MiniCPM BF16 GGUF matching the configured rescorer tokenizer."
  :type '(choice (const nil) file) :group 'wime)

(defconst wime-backend--definitions
  '((verify :capabilities (chinese verify) :search-method verify)
    (qwen-hf :capabilities (chinese japanese rerank verify decode) :search-method decode
             :rescorer "rescorers/Qwen3.5-2B-Base")
    (minicpm-native :capabilities (chinese decode) :search-method decode
                    :rescorer "rescorers/MiniCPM5-2B" :native t)
    (legacy :capabilities (chinese japanese decode) :search-method decode
            :rescorer "rescorers/Qwen3.5-2B-Base"))
  "Backend capabilities and default assets relative to the data directory.")

(defun wime-backend-supports-p (capability &optional backend)
  "Whether BACKEND (default `wime-backend') supports CAPABILITY."
  (memq capability (plist-get (cdr (assq (or backend wime-backend)
                                         wime-backend--definitions)) :capabilities)))

(defun wime-backend-from-profile (profile)
  "Identify the Wime setup from PROFILE's model and search settings."
  (let* ((name (plist-get profile :backend))
         (backend (car (seq-find (lambda (entry)
                                  (and (not (eq (car entry) 'verify))
                                       (equal name (symbol-name (car entry)))))
                                wime-backend--definitions))))
    (unless backend (error "Unknown model backend: %s" name))
    (if (and (eq backend 'qwen-hf)
             (equal (plist-get profile :search_method) "verify")
             (memq (plist-get profile :rescorer) '(nil :null)))
        'verify
      backend)))

(defun wime-backend--model-name (&optional backend)
  "Return the worker model implementation for the selected Wime BACKEND.
Verify leaves the worker's default implementation unused, with no rescorer."
  (let ((backend (or backend wime-backend)))
    (if (eq backend 'verify) "qwen-hf" (symbol-name backend))))

(defun wime-backend--arguments (search-method rescorer database)
  "Return backend worker arguments for SEARCH-METHOD, RESCORER and DATABASE.
Keep the default verification protocol usable without decoder assets."
  (unless (wime-backend-supports-p search-method)
    (error "Backend %s does not support %s search" wime-backend search-method))
  (if (eq wime-backend 'verify)
      (when rescorer (error "Verify backend uses only the conditional model"))
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
                      "--native-model" (expand-file-name wime-backend-native-model)))))))

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
    (user-error "Backend %s does not support Japanese; deactivate Japanese input first" backend))
  (let* ((definition (cdr (assq backend wime-backend--definitions)))
         (native (plist-get definition :native))
         (rescorer (when-let* ((path (plist-get definition :rescorer)))
                     (expand-file-name path wime-checkpoint-directory)))
         (library (and native (expand-file-name "native/current/lib/libwime_llama.so" wime-checkpoint-directory)))
         (gguf (and native (expand-file-name "model.bf16.gguf" rescorer)))
         (profile (wime-checkpoint-profile)))
    (dolist (path (append (list wime-neural-model-file wime-database-file)
                         (and rescorer (list (expand-file-name "config.json" rescorer)
                                             (expand-file-name "tokenizer.json" rescorer)))
                         (and native (list library gguf))))
      (unless (and path (file-readable-p path))
        (user-error "Backend asset is missing: %s" path)))
    (setq profile (plist-put profile :backend (wime-backend--model-name backend))
          profile (plist-put profile :search_method (symbol-name (plist-get definition :search-method)))
          profile (plist-put profile :rescorer (or rescorer :null))
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
