;;; wime-backend.el --- Wime inference setup selection -*- lexical-binding: t; -*-

;;; Commentary:
;; The menu selects a complete setup.  Runtime state keeps the model
;; implementation, search strategy and model assets separate; there is no
;; second copy of that state under the menu entry's name.

;;; Code:

(require 'wime)

(defvar wime-checkpoint-directory)
(defvar wime-neural-model-file)
(declare-function wime-checkpoint-profile "wime-checkpoint" ())
(declare-function wime-checkpoint-call "wime-checkpoint" (request))
(declare-function wime-checkpoint-switch "wime-checkpoint" (recipe persist resolve reject))

(defcustom wime-backend-model 'qwen-hf
  "Model implementation used when a general language model is configured.
Use `wime-select-backend' to choose and prewarm a complete inference setup."
  :type '(choice (const qwen-hf) (const minicpm-native) (const legacy))
  :group 'wime)

(defcustom wime-backend-native-library nil
  "Wime llama bridge shared library for `minicpm-native'."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-backend-native-model nil
  "MiniCPM BF16 GGUF matching the configured rescorer tokenizer."
  :type '(choice (const nil) file) :group 'wime)

(defconst wime-backend--presets
  '((verify :backend "qwen-hf" :search_method "verify")
    (qwen-hf :backend "qwen-hf" :search_method "decode"
             :rescorer "rescorers/Qwen3.5-2B-Base")
    (minicpm-native :backend "minicpm-native" :search_method "decode"
                    :rescorer "rescorers/MiniCPM5-2B")
    (legacy :backend "legacy" :search_method "decode"
            :rescorer "rescorers/Qwen3.5-2B-Base"))
  "Menu entries with runtime settings and relative model asset paths.")

(defvar wime-backend--cancel-switch nil)

(defun wime-select-backend (name)
  "Prewarm the setup NAME and save it as the next-start configuration.
The current worker remains in use until the replacement is ready and all
compositions have finished.  Assets live in `wime-checkpoint-directory'."
  (interactive (list (intern (completing-read "Wime backend: "
                                             (mapcar #'car wime-backend--presets) nil t))))
  (let ((preset (cdr (assq name wime-backend--presets))))
    (unless preset (user-error "Unknown Wime setup: %s" name))
    (require 'wime-checkpoint)
    (let* ((native (equal (plist-get preset :backend) "minicpm-native"))
           (rescorer (when-let* ((path (plist-get preset :rescorer)))
                       (expand-file-name path wime-checkpoint-directory)))
           (library (and native (expand-file-name "native/current/lib/libwime_llama.so" wime-checkpoint-directory)))
           (gguf (and native (expand-file-name "model.bf16.gguf" rescorer)))
           (profile (wime-checkpoint-profile)))
      (when (and (or native (null rescorer))
                 (cl-some (lambda (buffer)
                            (with-current-buffer buffer
                              (and current-input-method
                                   (string-prefix-p "wime-japanese" current-input-method))))
                          (buffer-list)))
        (user-error "Setup %s cannot convert Japanese; deactivate Japanese input first" name))
      (dolist (path (append (list wime-neural-model-file wime-database-file)
                           (and rescorer (list (expand-file-name "config.json" rescorer)
                                               (expand-file-name "tokenizer.json" rescorer)))
                           (and native (list library gguf))))
        (unless (and path (file-readable-p path))
          (user-error "Backend asset is missing: %s" path)))
      (setq profile (plist-put profile :backend (plist-get preset :backend))
            profile (plist-put profile :search_method (plist-get preset :search_method))
            profile (plist-put profile :rescorer (or rescorer :null))
            ;; Save the exact library version, not the mutable current symlink.
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
                 (message "Wime backend: %s (saved for startup)" name))
               (lambda (reason) (setq finished t wime-backend--cancel-switch nil)
                 (message "Wime kept the current backend: %s" reason))))
        ;; Command construction can reject a recipe synchronously.
        (when finished (setq wime-backend--cancel-switch nil))))))

(provide 'wime-backend)
;;; wime-backend.el ends here
