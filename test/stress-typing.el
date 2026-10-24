;;; stress-typing.el --- Real-worker typing and stale-reply stress -*- lexical-binding: t; -*-
(require 'wime-neural)
(require 'json)
(setq wime-database-file (getenv "WIME_DATABASE")
      wime-grammar-module (getenv "WIME_GRAMMAR_MODULE")
      wime-grammar-shared-dir (getenv "WIME_GRAMMAR")
      wime-neural-program (getenv "WIME_NEURAL_PROGRAM")
      wime-neural-model-file (getenv "WIME_NEURAL_MODEL")
      wime-neural-search-method 'verify wime-neural-delay 0
      wime-search-width 16 wime-neural-inference-budget 4
      wime-neural-seed-candidates t
      wime-neural-single-character (equal (getenv "WIME_SINGLE_CHARACTER") "1"))
(setq wime-neural-context-anchor (string-to-number (or (getenv "WIME_CONTEXT_ANCHOR") "0")))
(setq wime-neural-prior-confidence (string-to-number (or (getenv "WIME_PRIOR_CONFIDENCE") "1")))

(defun wime-stress--wait ()
  (let ((deadline (+ (float-time) 20)))
    (while (and (< (float-time) deadline)
                (or (memq wime-neural--timer timer-list)
                    wime-neural--busy wime-neural--pending))
      (accept-process-output wime-neural--process 0.001))
    (when (or wime-neural--failed wime-neural--busy wime-neural--pending
              (memq wime-neural--timer timer-list))
      (error "Worker did not settle"))))

(defun wime-stress--dispatch ()
  "Send the scheduled job now, before a selection/edit invalidates its reply."
  (unless (timerp wime-neural--timer) (error "No scheduled request"))
  (let ((job (car (timer--args wime-neural--timer))))
    (cancel-timer wime-neural--timer)
    (setq wime-neural--timer nil)
    (wime-neural--enqueue job)
    (unless wime-neural--busy (error "No in-flight request"))))

(let ((cases (with-temp-buffer
               (insert-file-contents (getenv "WIME_CASES"))
               (json-parse-buffer :object-type 'alist :array-type 'list)))
      (keys 0) (checks 0) (requests 0))
  (unwind-protect
      (progn
        (wime-neural-mode 1)
        (dolist (case cases)
          (with-temp-buffer
            (insert (alist-get 'context case))
            ;; Real keys, interspersed with output processing and backspace;
            ;; intermediate syllables race with newer requests in the worker.
            (let ((code (alist-get 'input case)))
              (dolist (key (string-to-list code))
                (wime-input-method key)
                (cl-incf keys)
                (when (= (% keys 3) 0)
                  (accept-process-output wime-neural--process 0.001))
                (when (= (% keys 5) 0)
                  (wime-delete-backward)
                  (wime-input-method key)))
              (wime-stress--wait)
              (unless (equal (car wime--candidates) (alist-get 'expected case))
                (error "Typing/replay mismatch: %s" (alist-get 'id case)))
              (cl-incf checks)
              (dolist (mutation '(selection edit boundary cancel))
                ;; With empty context, narrowing to point changes nothing.
                ;; Add temporary editable text so this really invalidates the
                ;; captured context, then restore it before the next scenario.
                (let ((boundary-text (and (eq mutation 'boundary)
                                          (string-empty-p (wime--context)))))
                (when boundary-text (insert "边界测试"))
                (wime--refresh)
                (wime-stress--dispatch)
                (cl-incf requests)
                (let ((before wime--candidates) (position (point)))
                  (pcase mutation
                    ('selection (setq wime--selected 1))
                    ('edit (insert "另一个话题"))
                    ('boundary (setq-local wime-context-start-function #'point))
                    ('cancel (wime--reset)))
                  (wime-stress--wait)
                  (unless (if (eq mutation 'cancel)
                              (and (null wime--input) (null wime--candidates))
                            (equal before wime--candidates))
                    (error "Stale reply applied after %s: %s" mutation (alist-get 'id case)))
                  (cl-incf checks)
                  (when (eq mutation 'edit) (delete-region position (point)))
                  (setq wime-context-start-function nil wime--selected 0))
                (when boundary-text (delete-region (- (point) 4) (point)))))
              ;; Re-enter the same code after cancellation; the earlier reply
              ;; must not poison the new composition's snapshot or cache.
              (mapc #'wime-input-method (string-to-list code))
              (wime-stress--wait)
              (unless (equal (car wime--candidates) (alist-get 'expected case))
                (error "Retyping/replay mismatch: %s" (alist-get 'id case)))
              (cl-incf checks)
              (wime--reset))))
        (with-temp-file (getenv "WIME_RECORDING")
          (insert (json-serialize `(:cases ,(length cases) :typed_keys ,keys
                                   :checks ,checks :invalidated_inflight_requests ,requests
                                   :failures 0)))))
    (wime-neural-mode -1)))
