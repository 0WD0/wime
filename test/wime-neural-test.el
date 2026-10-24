;;; wime-neural-test.el --- Async candidate lifecycle tests -*- lexical-binding: t; -*-
(require 'ert)
(require 'wime-neural)

(defmacro wime-neural-test--composition (&rest body)
  (declare (indent 0))
  `(with-temp-buffer
     (let ((wime-neural-mode t))
       (setq wime--input "aabb"
             wime--selected 0
             wime--candidate-records
             (list (wime--make-candidate :text "甲乙" :end 2 :score 1)
                   (wime--make-candidate :text "丙丁" :end 2 :score 0)
                   (wime--make-candidate :text "甲" :end 1 :score 2))
             wime--candidates '("甲乙" "丙丁" "甲"))
       (let ((job (wime-neural--make-job
                   :id 1 :buffer (current-buffer) :input wime--input
                   :position (point) :tick (buffer-chars-modified-tick)
                   :context "" :records wime--candidate-records :syllables 2)))
         (unwind-protect (progn ,@body) (wime--reset))))))

(ert-deftest wime-neural-promotes-only-a-current-full-reading ()
  (wime-neural-test--composition
    (wime-neural--apply job '((promoted . 2)))
    (should (equal (car wime--candidates) "甲乙"))
    (wime-neural--apply job '((promoted . 1) (elapsed_ms . 3.5)))
    (should (equal wime--candidates '("丙丁" "甲乙" "甲")))
    (should (= wime-neural-last-latency 3.5))))

(ert-deftest wime-neural-verification-only-publishes-the-best-verified-draft ()
  (wime-neural-test--composition
    (let ((wime-neural--pending nil) (wime-neural-inference-budget 2))
      (setf (wime-neural--job-method job) 'verify
            (wime-neural--job-draft job) (car wime--candidate-records))
      (cl-letf (((symbol-function 'wime--sentence-records)
                 (lambda (_units prefix)
                   (when (equal prefix "丙")
                     (list (wime--make-candidate :text "丙乙" :end 2 :score -1))))))
        (wime-neural--apply job '((score . -2.0) (prefixes . ["错" "丙"])))
        (should (equal (car wime--candidates) "甲乙"))
        (should wime-neural--pending)
        (let ((continuation wime-neural--pending))
          (setq wime-neural--pending nil)
          (wime-neural--apply continuation '((score . -1.0) (prefixes . ["丁"])))
          (should (equal (car wime--candidates) "丙乙"))
          (should-not wime-neural--pending))))))

(ert-deftest wime-neural-verification-retains-baseline-on-regression-or-impossible-prefix ()
  (wime-neural-test--composition
    (let ((wime-neural--pending nil) (wime-neural-inference-budget 2))
      (setf (wime-neural--job-method job) 'verify
            (wime-neural--job-draft job) (car wime--candidate-records))
      (cl-letf (((symbol-function 'wime--sentence-records) (lambda (&rest _) nil)))
        (wime-neural--apply job '((score . -1.0) (prefixes . ["无解"])))
        (should (equal (car wime--candidates) "甲乙"))
        (should-not wime-neural--pending)))))

(ert-deftest wime-neural-prior-rejects-weak-edits-but-allows-strong-corrections ()
  (dolist (neural-score '(-1.9 -1.0))
    (wime-neural-test--composition
      (let ((wime-neural--pending nil) (wime-neural-inference-budget 2))
        (setf (wime-neural--job-method job) 'verify
              (wime-neural--job-prior-weight job) 0.1
              (wime-neural--job-draft job) (car wime--candidate-records))
        (cl-letf (((symbol-function 'wime--sentence-records)
                   (lambda (&rest _)
                     (list (wime--make-candidate :text "丙乙" :end 2 :score -3)))))
          (wime-neural--apply job '((score . -2.0) (prefixes . ["丙"])))
          (let ((continuation wime-neural--pending))
            (setq wime-neural--pending nil)
            (wime-neural--apply continuation `((score . ,neural-score)))
            (should (equal (car wime--candidates)
                           (if (= neural-score -1.0) "丙乙" "甲乙")))))))))

(ert-deftest wime-neural-auxiliary-key-constrains-search-but-not-pinyin-conditioning ()
  (wime-neural-test--composition
    (let ((wime-neural--pending nil) (wime-neural-inference-budget 2) searched)
      (setq wime--input "moMx;")
      (setf (wime-neural--job-input job) wime--input
            (wime-neural--job-method job) 'verify
            (wime-neural--job-draft job) (car wime--candidate-records))
      (should (equal (alist-get 'input (json-parse-string (wime-neural--request job)
                                                        :object-type 'alist)) "mox;"))
      (cl-letf (((symbol-function 'wime--sentence-records)
                 (lambda (units _prefix) (setq searched units) nil)))
        (wime-neural--apply job '((score . -1.0) (prefixes . ["模"])))
        (should (equal searched '(("mo" . "M") ("x;" . ""))))))))

(ert-deftest wime-neural-conditional-scheduling-requires-the-trained-keyboard-scheme ()
  (wime-neural-test--composition
    (let ((wime-neural-search-method 'verify) (wime-neural-model-file "model.safetensors")
          (wime-neural--failed nil) (scheduled nil) (scheme "liangti"))
      (cl-letf (((symbol-function 'wime--database) (lambda () 'database))
                ((symbol-function 'sqlite-select) (lambda (&rest _) (list (list scheme))))
                ((symbol-function 'run-with-timer) (lambda (&rest _) (setq scheduled t))))
        (wime-neural--schedule)
        (should-not scheduled)
        (setq scheme "mspy")
        (wime-neural--schedule)
        (should scheduled)))))

(ert-deftest wime-neural-rejects-replaced-compositions ()
  (wime-neural-test--composition
    ;; Same input after backspace and retyping still has different candidate records.
    (setq wime--candidate-records (copy-sequence wime--candidate-records))
    (wime-neural--apply job '((promoted . 1)))
    (should (equal (car wime--candidates) "甲乙"))))

(ert-deftest wime-neural-rejects-buffer-edits-and-manual-selection ()
  (wime-neural-test--composition
    (setq wime--selected 1)
    (should-not (wime-neural--valid-p job))
    (setq wime--selected 0)
    (insert "新")
    (delete-char -1)
    (should-not (wime-neural--valid-p job))))

(ert-deftest wime-neural-cancel-and-commit-invalidate-replies ()
  (wime-neural-test--composition
    (wime--commit "甲乙")
    (wime-neural--apply job '((promoted . 1)))
    (should (equal (buffer-string) "甲乙"))
    (should-not wime--candidates)))

(ert-deftest wime-neural-busy-worker-retains-only-latest-request ()
  (wime-neural-test--composition
    (let ((wime-neural--busy job) (wime-neural--pending nil)
          (wime-neural--failed nil))
      (wime-neural--enqueue job)
      (let ((new (copy-wime-neural--job job)))
        (setf (wime-neural--job-id new) 2)
        (wime-neural--enqueue new)
        (should (eq wime-neural--pending new))))))

(ert-deftest wime-neural-request-encodes-runtime-coverage ()
  (wime-neural-test--composition
    (let* ((request (json-parse-string (wime-neural--request job) :object-type 'alist))
           (candidates (alist-get 'candidates request)))
      (should (= (alist-get 'syllables request) 2))
      (should (= (alist-get 'end (aref candidates 2)) 1))
      (should (eq (alist-get 'dictionary (aref candidates 0)) :false)))))

(ert-deftest wime-neural-filter-handles-split-lines-and-unrelated-ids ()
  (wime-neural-test--composition
    (let* ((process (make-pipe-process :name "wime-protocol-test" :noquery t))
           (wime-neural--process process) (wime-neural--busy job)
           (wime-neural--pending nil) (wime-neural--timeout-timer nil))
      (unwind-protect
          (progn
            (wime-neural--filter process "{\"id\":9,\"promoted\":1}\n{\"id\":1,")
            (should (eq wime-neural--busy job))
            (should (equal (car wime--candidates) "甲乙"))
            (wime-neural--filter process "\"promoted\":1,\"elapsed_ms\":2}\n")
            (should-not wime-neural--busy)
            (should (equal (car wime--candidates) "丙丁")))
        (delete-process process)))))

(ert-deftest wime-neural-dictionary-leader-does-not-schedule-work ()
  (wime-neural-test--composition
    (let ((wime-neural-model-file "model.safetensors") (wime-neural--failed nil)
          (called nil))
      (setf (wime-candidate-dictionary (car wime--candidate-records)) t)
      (cl-letf (((symbol-function 'run-with-timer) (lambda (&rest _) (setq called t))))
        (wime-neural--schedule)
        (should-not called)))))

(ert-deftest wime-neural-failure-preserves-traditional-candidates ()
  (wime-neural-test--composition
    (let ((wime-neural--process nil) (wime-neural--busy job)
          (wime-neural--pending job) (wime-neural--failed nil))
      (wime-neural--fail "test failure")
      (should wime-neural--failed)
      (should-not wime-neural--pending)
      (should (equal (car wime--candidates) "甲乙")))))

(ert-deftest wime-neural-only-initialization-gets-the-longer-timeout ()
  (wime-neural-test--composition
    (let* ((process (make-pipe-process :name "wime-timeout-test" :noquery t))
           (wime-neural--process process) (wime-neural--busy nil)
           (wime-neural--pending nil) (wime-neural--failed nil)
           (wime-neural--timeout-timer nil)
           (wime-neural-startup-timeout 10) (wime-neural-timeout 2) delays)
      (unwind-protect
          (cl-letf (((symbol-function 'wime-neural--start) (lambda () process))
                    ((symbol-function 'run-with-timer)
                     (lambda (delay &rest _) (push delay delays) nil))
                    ((symbol-function 'process-send-string) #'ignore))
            (wime-neural--enqueue job)
            (wime-neural--reply '((id . 99)))
            (should-not (process-get process 'wime-initialized))
            (wime-neural--reply '((id . 1)))
            (should (process-get process 'wime-initialized))
            (wime-neural--enqueue job)
            (should (equal delays '(2 10))))
        (delete-process process)))))

(ert-deftest wime-neural-real-worker-protocol ()
  (let ((program (getenv "WIME_NEURAL_PROGRAM"))
        (model (getenv "WIME_NEURAL_MODEL")))
    (unless (and program model) (ert-skip "model and worker not configured"))
    (wime-neural-test--composition
      (let ((wime-neural-program program) (wime-neural-model-file model)
            (wime-neural--process nil) (wime-neural--pending nil)
            (wime-neural--busy nil) (wime-neural--failed nil)
            (wime-neural--timeout-timer nil))
        (unwind-protect
            (progn
              (wime-neural--enqueue job)
              (let ((deadline (+ (float-time) 5)))
                (while (and wime-neural--busy (< (float-time) deadline))
                  (accept-process-output wime-neural--process 0.05)))
              (should-not wime-neural--failed)
              (should-not wime-neural--busy)
              (should (numberp wime-neural-last-latency)))
          (wime-neural--stop))))))
