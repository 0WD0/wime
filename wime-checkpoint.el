;;; wime-checkpoint.el --- Verified runtime checkpoint switching -*- lexical-binding: t; -*-

;; SPDX-License-Identifier: GPL-3.0-or-later

;;; Code:
(require 'wime-neural)
(require 'json)
(defcustom wime-checkpoint-runtime-directory nil
  "Directory containing the packaged wime_research backend, or nil when installed."
  :type '(choice (const nil) directory) :group 'wime)
(defcustom wime-checkpoint-python "python3" "Python for the experiment controller." :type 'string :group 'wime)
(defcustom wime-checkpoint-directory wime-data-directory
  "Persistent experiment and checkpoint directory." :type 'directory :group 'wime)
(defvar wime-checkpoint-previous nil "Previous successfully selected runtime recipe.")

(defun wime-checkpoint-command (&optional python)
  "Controller argv using PYTHON or the configured experiment Python."
  (list (or python wime-checkpoint-python) "-m" "wime_research.lab"
        "--root" (expand-file-name wime-checkpoint-directory) "request"))

(defun wime-checkpoint-call (request)
  "Execute a small controller REQUEST and return its checked response."
  (let ((default-directory (file-name-as-directory (expand-file-name
                               (or wime-checkpoint-runtime-directory temporary-file-directory))))
        (command (wime-checkpoint-command))
        (coding-system-for-read 'utf-8-unix) (coding-system-for-write 'utf-8-unix))
    (with-temp-buffer
      (insert (decode-coding-string (json-serialize request) 'utf-8-unix))
      (let ((code (apply #'call-process-region (point-min) (point-max) (car command) t t nil (cdr command))))
        (unless (equal code 0)
          (error "Wime Lab controller exited %s: %s" code (string-trim (buffer-string)))))
      (goto-char (point-min))
      (let ((response (json-parse-buffer :object-type 'plist :array-type 'list
                                       :null-object nil :false-object :false)))
        (unless (eq (plist-get response :ok) t) (error "%s" (plist-get response :error)))
        (plist-get response :value)))))

(defun wime-checkpoint-profile ()
  "Return the actual inference settings as a JSON-compatible recipe."
  (list :width wime-search-width :encoding "wime-composition-v2"
        :single_character (if wime-neural-single-character t :false)
        :rescorer (or wime-neural-rescorer :null) :rescorer_weight wime-neural-rescorer-weight
        :beam wime-neural-beam))

(defun wime-checkpoint--nullable (value)
  "Normalize JSON null VALUE in both serialized and in-memory recipes."
  (unless (eq value :null) value))

(defun wime-checkpoint--settings (recipe)
  "Read RECIPE into runtime variable bindings before changing live state."
  (let ((profile (plist-get recipe :profile)))
    `((wime-neural-model-file . ,(plist-get recipe :path))
      (wime-search-width . ,(plist-get profile :width))
      (wime-neural-single-character . ,(eq (plist-get profile :single_character) t))
      (wime-neural-rescorer . ,(wime-checkpoint--nullable (plist-get profile :rescorer)))
      (wime-neural-rescorer-weight . ,(or (plist-get profile :rescorer_weight) 1.5))
      (wime-neural-beam . ,(or (plist-get profile :beam) 32)))))

(defun wime-checkpoint--apply (recipe)
  "Install RECIPE variables; callers own the worker transition."
  (dolist (binding (wime-checkpoint--settings recipe))
    (set (car binding) (cdr binding))))

(defun wime-checkpoint-restore ()
  "Restore the explicit startup recipe, retaining packaged defaults on error."
  (when (file-exists-p (expand-file-name "lab.sqlite3" wime-checkpoint-directory))
    (condition-case err
        (when-let* ((recipe (wime-checkpoint-call '(:op "startup"))))
          (wime-checkpoint--apply recipe))
      (error (display-warning 'wime (format "Saved checkpoint unavailable; packaged model retained: %s"
                                            (error-message-string err)))))))

(defun wime-checkpoint--composing-p ()
  "Return non-nil while any buffer has uncommitted input."
  (cl-some (lambda (buffer)
             (with-current-buffer buffer
               (or (wime--active-p) (length> (bound-and-true-p wime-japanese--keys) 0))))
           (buffer-list)))

(defun wime-checkpoint-runtime ()
  "Report configuration separately from the worker's actual command."
  (list :configured wime-neural-model-file
        :loaded (and (process-live-p wime-neural--process)
                     (car (last (process-command wime-neural--process))))
        :pid (and (process-live-p wime-neural--process) (process-id wime-neural--process))
        :enabled (if (eq wime-decoding 'lattice) :false t)
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
                     (wime-neural-restart)
                     (wime-checkpoint--apply recipe)
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
                  (wime-neural--make-process
                   (let ((settings (wime-checkpoint--settings recipe)))
                     (cl-progv (mapcar #'car settings) (mapcar #'cdr settings)
                       (wime-neural--command wime-neural-model-file)))
                   :name "wime-checkpoint-prewarm" :stderr " *wime-checkpoint-errors*"
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
                               (unless (and (= (plist-get reply :id) 0)
                                            (let ((pair (aref (plist-get reply :candidates) 0)))
                                              (and (stringp (aref pair 0)) (numberp (aref pair 1)))))
                                 (error "GPU prewarm did not return a valid inference result"))
                               (setq ready t)
                               (when (timerp deadline) (cancel-timer deadline))
                               (commit))
                           (error (fail (error-message-string error-data)))))))))
            (setq deadline (run-at-time wime-neural-startup-timeout nil #'fail "GPU checkpoint prewarm timed out"))
            (process-send-string process
                                 (concat (json-serialize
                                          '(:id 0 :operation "decode" :syllables 2 :context ""
                                            :input "nihk")) "\n")))
        (error (fail (error-message-string err))))
      (lambda () (unless done (setq done t) (cleanup))))))

(provide 'wime-checkpoint)
;;; wime-checkpoint.el ends here
