;;; wime-speech.el --- Streaming dictation at point -*- lexical-binding: t; -*-

;; SPDX-License-Identifier: GPL-3.0-or-later

;; Package-Requires: ((emacs "29.1"))

;;; Commentary:
;; `wime-speech-dictate' toggles recording through a persistent worker that
;; runs Qwen3-ASR on pw-record audio.  The text before point is its context.
;; Partial text shows at point every two seconds; the final text is inserted
;; where dictation started, about 0.1 s after it stops.

;;; Code:

(require 'cl-lib)
(require 'json)
(require 'wime-neural)

(defcustom wime-speech-command
  (list (expand-file-name "python/bin/python" wime-data-directory)
        "-m" "wime_runtime.speech")
  "Command and fixed arguments for the dictation worker.
The model directory and options are appended."
  :type '(repeat string) :group 'wime)

(defcustom wime-speech-model
  (expand-file-name "asr/Qwen3-ASR-1.7B-hf" wime-data-directory)
  "Directory of Qwen/Qwen3-ASR-1.7B-hf."
  :type 'directory :group 'wime)

(defcustom wime-speech-target nil
  "PipeWire source name for pw-record, or nil for the default source."
  :type '(choice (const nil) string) :group 'wime)

(defconst wime-speech-context-limit 1024
  "Characters before point sent as the recognizer's context.")

(defface wime-speech-partial-face
  '((t :inherit shadow :underline t))
  "Face of dictated text that may still change."
  :group 'wime)

(cl-defstruct (wime-speech--session (:constructor wime-speech--make-session))
  id marker overlay (state 'loading) (text ""))

(defvar wime-speech--process nil)
(defvar wime-speech--session nil
  "The dictation in progress or awaiting its final text, or nil.")
(defvar wime-speech--serial 0)

(defun wime-speech--show (session)
  "Render SESSION's state and partial text in its overlay."
  (overlay-put (wime-speech--session-overlay session) 'after-string
               (concat (pcase (wime-speech--session-state session)
                         ('loading "…")
                         ('recording "●")
                         ('finishing "◌"))
                       (propertize (wime-speech--session-text session)
                                   'face 'wime-speech-partial-face))))

(defun wime-speech--end (session)
  (delete-overlay (wime-speech--session-overlay session))
  (set-marker (wime-speech--session-marker session) nil)
  (when (eq session wime-speech--session)
    (setq wime-speech--session nil)))

(defun wime-speech--insert (session text)
  "Insert TEXT where SESSION started, after point if point is still there."
  (let ((marker (wime-speech--session-marker session)))
    (when (buffer-live-p (marker-buffer marker))
      (with-current-buffer (marker-buffer marker)
        (if (= (point) marker)
            (insert text)
          (save-excursion (goto-char marker) (insert text)))))))

(defun wime-speech--reply (response)
  (let ((session wime-speech--session))
    (when (and session (eql (alist-get 'id response) (wime-speech--session-id session)))
      (cond ((assq 'final response)
             (wime-speech--insert session (alist-get 'final response))
             (wime-speech--end session))
            ((assq 'partial response)
             (setf (wime-speech--session-text session) (alist-get 'partial response))
             (wime-speech--show session))
            ((assq 'recording response)
             (when (eq (wime-speech--session-state session) 'loading)
               (setf (wime-speech--session-state session) 'recording))
             (wime-speech--show session))))))

(defun wime-speech--filter (process chunk)
  "Decode possibly split JSON lines from PROCESS CHUNK."
  (let* ((data (concat (process-get process 'partial) chunk))
         (lines (split-string data "\n")))
    (process-put process 'partial (car (last lines)))
    (dolist (line (butlast lines))
      (wime-speech--reply (json-parse-string line :object-type 'alist :null-object nil)))))

(defun wime-speech--sentinel (process _event)
  (unless (process-live-p process)
    (when wime-speech--session
      (wime-speech--end wime-speech--session))
    (message "wime speech worker closed; log: %s" (process-get process 'wime-log))))

(defun wime-speech--start ()
  "Return a running dictation worker, starting it if needed."
  (unless (process-live-p wime-speech--process)
    (unless (file-directory-p wime-speech-model)
      (user-error "Speech model is not installed: %s" wime-speech-model))
    (setq wime-speech--process
          (wime-neural--make-process
           (append wime-speech-command
                   (list (expand-file-name wime-speech-model))
                   (and wime-speech-target (list "--target" wime-speech-target)))
           :name "wime-speech"
           :filter #'wime-speech--filter :sentinel #'wime-speech--sentinel)))
  wime-speech--process)

(defun wime-speech--send (&rest request)
  (process-send-string (wime-speech--start) (concat (json-encode request) "\n")))

;;;###autoload
(defun wime-speech-dictate ()
  "Start dictating at point, or stop and insert the recognized text.
The first dictation also loads the model; recording starts once it is ready."
  (interactive)
  (let ((session wime-speech--session))
    (cond ((null session)
           (barf-if-buffer-read-only)
           (let ((context (concat (wime--context-prefix) (wime--context wime-speech-context-limit))))
             (setq session (wime-speech--make-session
                            :id (cl-incf wime-speech--serial) :marker (point-marker)
                            :overlay (make-overlay (point) (point) nil t nil)))
             (wime-speech--send :op "start" :id (wime-speech--session-id session) :context context)
             (setq wime-speech--session session)
             (wime-speech--show session)))
          ((eq (wime-speech--session-state session) 'finishing)
           (message "wime speech: still recognizing"))
          (t
           (setf (wime-speech--session-state session) 'finishing)
           (wime-speech--show session)
           (wime-speech--send :op "stop" :id (wime-speech--session-id session))))))

(defun wime-speech-cancel ()
  "Drop the dictation in progress without inserting anything."
  (interactive)
  (when-let* ((session wime-speech--session))
    (wime-speech--send :op "cancel" :id (wime-speech--session-id session))
    (wime-speech--end session)))

(provide 'wime-speech)
;;; wime-speech.el ends here
