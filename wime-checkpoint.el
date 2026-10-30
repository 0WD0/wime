;;; wime-checkpoint.el --- Verified runtime checkpoint switching -*- lexical-binding: t; -*-
;;; Code:
(require 'wime-neural)
(require 'json)
(defcustom wime-checkpoint-program nil "Path to Wime Lab's Python controller." :type '(choice (const nil) file) :group 'wime)
(defcustom wime-checkpoint-python "python3" "Python for the experiment controller." :type 'string :group 'wime)
(defcustom wime-checkpoint-directory
  (expand-file-name "wime" (or (getenv "XDG_DATA_HOME") "~/.local/share"))
  "Persistent experiment and checkpoint directory." :type 'directory :group 'wime)
(defvar wime-checkpoint-previous nil "Previous successfully selected runtime recipe.")

(defun wime-checkpoint-call (request)
  "Execute a small controller REQUEST and return its checked response."
  (unless (and wime-checkpoint-program (file-readable-p wime-checkpoint-program))
    (error "Wime Lab controller is unavailable"))
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8-unix) (coding-system-for-write 'utf-8-unix))
      (insert (json-serialize request))
      (unless (zerop (call-process-region (point-min) (point-max) wime-checkpoint-python t t nil
                                        wime-checkpoint-program "--root" wime-checkpoint-directory "request"))
        (error "Wime Lab controller failed"))
      (goto-char (point-min))
      (let ((response (json-parse-buffer :object-type 'plist :array-type 'list
                                         :null-object nil :false-object :false)))
        (unless (eq (plist-get response :ok) t) (error "%s" (plist-get response :error)))
        (plist-get response :value)))))

(defun wime-checkpoint-profile ()
  "Return the actual inference settings as a JSON-compatible recipe."
  (list :width wime-search-width :budget wime-neural-inference-budget
        :prior_weight wime-neural-prior-weight :prior_confidence wime-neural-prior-confidence
        :encoding "wime-composition-v2"
        :single_character (if wime-neural-single-character t :false)
        :seed_draft (if wime-neural-seed-candidates "neural" "traditional")
        :rescorer (or wime-neural-rescorer :null) :rescorer_weight wime-neural-rescorer-weight
        :backend (symbol-name wime-backend-model)
        :native_library (or wime-backend-native-library :null)
        :native_model (or wime-backend-native-model :null)
        :pool_width (or wime-neural-pool-width :null) :beam wime-neural-beam
        :search_method (symbol-name wime-neural-search-method)))

(defun wime-checkpoint--nullable (value)
  "Normalize JSON null VALUE in both serialized and in-memory recipes."
  (unless (eq value :null) value))

(defun wime-checkpoint--settings (recipe)
  "Read RECIPE into runtime variable bindings before changing live state."
  (let* ((profile (plist-get recipe :profile))
         (backend (plist-get profile :backend))
         (method (plist-get profile :search_method)))
    (unless (member backend '("qwen-hf" "minicpm-native" "legacy"))
      (error "Unknown model backend: %s" backend))
    (unless (member method '("rerank" "verify" "decode"))
      (error "Unknown search method: %s" method))
    `((wime-neural-model-file . ,(plist-get recipe :path))
      (wime-backend-model . ,(intern backend))
      (wime-neural-search-method . ,(intern method))
      (wime-search-width . ,(plist-get profile :width))
      (wime-neural-inference-budget . ,(plist-get profile :budget))
      (wime-neural-prior-weight . ,(plist-get profile :prior_weight))
      (wime-neural-prior-confidence . ,(plist-get profile :prior_confidence))
      (wime-neural-single-character . ,(eq (plist-get profile :single_character) t))
      (wime-neural-seed-candidates . ,(equal (plist-get profile :seed_draft) "neural"))
      (wime-neural-rescorer . ,(wime-checkpoint--nullable (plist-get profile :rescorer)))
      (wime-backend-native-library . ,(wime-checkpoint--nullable (plist-get profile :native_library)))
      (wime-backend-native-model . ,(wime-checkpoint--nullable (plist-get profile :native_model)))
      (wime-neural-rescorer-weight . ,(or (plist-get profile :rescorer_weight) 1.5))
      (wime-neural-beam . ,(or (plist-get profile :beam) 16))
      (wime-neural-pool-width . ,(wime-checkpoint--nullable (plist-get profile :pool_width))))))

(defun wime-checkpoint--apply (recipe)
  "Install RECIPE variables; callers own the worker transition."
  (dolist (binding (wime-checkpoint--settings recipe))
    (set (car binding) (cdr binding))))

(defun wime-checkpoint-restore ()
  "Restore the explicit startup recipe, retaining packaged defaults on error."
  (when (and wime-checkpoint-program
             (file-exists-p (expand-file-name "lab.sqlite3" wime-checkpoint-directory)))
    (condition-case err
        (when-let* ((recipe (wime-checkpoint-call '(:op "startup"))))
          (wime-checkpoint--apply recipe))
      (error (display-warning 'wime (format "Saved checkpoint unavailable; packaged model retained: %s"
                                            (error-message-string err)))))))

(defun wime-checkpoint--composing-p ()
  "Return non-nil while any buffer has uncommitted input."
  (cl-some (lambda (buffer)
             (with-current-buffer buffer
               (or (and (stringp wime--input) (> (length wime--input) 0))
                   (and (bound-and-true-p wime-japanese--keys)
                        (> (length wime-japanese--keys) 0)))))
           (buffer-list)))

(defun wime-checkpoint-runtime ()
  "Report configuration separately from the worker's actual command."
  (list :configured wime-neural-model-file
        :loaded (and (process-live-p wime-neural--process)
                     (car (last (process-command wime-neural--process))))
        :pid (and (process-live-p wime-neural--process) (process-id wime-neural--process))
        :enabled (if wime-neural-mode t :false)
        :failed (if wime-neural--failed t :false) :profile (wime-checkpoint-profile)))

(defun wime-checkpoint-switch (recipe persist resolve reject)
  "Prewarm RECIPE, then switch at a composition boundary.
PERSIST also changes the next-start recipe.  RESOLVE and REJECT receive one
value.  Return cancellation that leaves the existing worker and input intact."
  (let (process timer deadline done ready partial)
    (cl-labels
        ((cleanup ()
           (when (timerp timer) (cancel-timer timer))
           (when (timerp deadline) (cancel-timer deadline))
           (when (process-live-p process)
             (set-process-sentinel process #'ignore)
             (delete-process process)))
         (fail (reason)
           (unless done
             (setq done t)
             (cleanup)
             (funcall reject reason)))
         (commit ()
           (unless done
             (cond
              ((not (process-live-p process)) (fail "New GPU worker exited"))
              ((wime-checkpoint--composing-p)
               (setq timer (run-at-time .15 nil #'commit)))
              (t
               (condition-case err
                   (progn
                     (when persist
                       (wime-checkpoint-call
                        (list :op "select" :checkpoint (plist-get recipe :checkpoint)
                              :profile (plist-get recipe :profile))))
                     (setq wime-checkpoint-previous
                           (list :path wime-neural-model-file :profile (wime-checkpoint-profile)))
                     (wime-neural-mode -1)
                     (wime-checkpoint--apply recipe)
                     (wime-neural-mode 1)
                     (setq wime-neural--process process done t)
                     (process-put process 'partial "")
                     (process-put process 'wime-initialized t)
                     (set-process-filter process #'wime-neural--filter)
                     (set-process-sentinel process #'wime-neural--sentinel)
                     (setq process nil)
                     (when (timerp deadline) (cancel-timer deadline))
                     (funcall resolve (wime-checkpoint-runtime)))
                 (error (fail (error-message-string err)))))))))
      (condition-case err
          (progn
            (setq process
                  (make-process
                   :name "wime-checkpoint-prewarm" :buffer nil :noquery t :connection-type 'pipe
                   :coding 'utf-8-unix :stderr " *wime-checkpoint-errors*"
                   :command (let ((settings (wime-checkpoint--settings recipe)))
                              (cl-progv (mapcar #'car settings) (mapcar #'cdr settings)
                                (wime-neural--command wime-neural-model-file)))
                   :sentinel (lambda (p _event)
                               (unless (process-live-p p) (fail "New GPU worker could not load checkpoint")))
                   :filter
                   (lambda (_p chunk)
                     (unless ready
                       (setq partial (concat partial chunk))
                       (when (string-match "\n" partial)
                         (condition-case error-data
                             (let ((reply (json-parse-string (substring partial 0 (match-beginning 0))
                                                            :object-type 'plist)))
                               (unless (and (= (plist-get reply :id) 0) (numberp (plist-get reply :score)))
                                 (error "GPU prewarm did not return a valid inference result"))
                               (setq ready t)
                               (when (timerp deadline) (cancel-timer deadline))
                               (commit))
                           (error (fail (error-message-string error-data)))))))))
            (setq deadline (run-at-time wime-neural-startup-timeout nil #'fail "GPU checkpoint prewarm timed out"))
            (process-send-string process
                                 (concat (json-serialize
                                          '(:id 0 :operation "verify" :syllables 2 :context ""
                                            :input "nihk" :draft "你好" :candidates [])) "\n")))
        (error (fail (error-message-string err))))
      (lambda () (unless done (setq done t) (cleanup))))))

(provide 'wime-checkpoint)
;;; wime-checkpoint.el ends here
