;;; wime-neural-test.el --- Async candidate lifecycle tests -*- lexical-binding: t; -*-
(require 'ert)
(require 'wime-neural)

(defvar wime-japanese-model-file)
(defvar wime-japanese-table-file)

(defmacro wime-neural-test--mspy (&rest body)
  "Run BODY with a database reporting the model's trained scheme."
  (declare (indent 0))
  `(let ((wime--position-keys (cons wime--default-keys wime--default-keys)))
     (cl-letf (((symbol-function 'wime--database) (lambda () 'test-database))
               ((symbol-function 'sqlite-select) (lambda (&rest _) '(("mspy")))))
       ,@body)))

(defmacro wime-neural-test--single-syllable (&rest body)
  "Run BODY with a small contextual dictionary for a single syllable."
  (declare (indent 0))
  `(wime-neural-test--mspy
     (cl-letf (((symbol-function 'wime--entries-for)
                (lambda (code)
                  (when (equal code "qd") '(("枪" "a" 100) ("强" "b" 100)))))
               ((symbol-function 'wime--grammar-score)
                (lambda (context word _rear)
                  (if (and (equal context "真刀真") (equal word "枪")) 1.0 0.0))))
       ,@body)))

(ert-deftest wime-neural-commits-retain-original-history-and-actual-input ()
  (let ((wime-user-database nil))
    (with-temp-buffer
      (let ((history (make-string 64 ?史)))
        (insert history)
        (setq wime--input "bmAyi"
              wime--candidate-records (list (wime--make-candidate :text "便宜" :end 2)))
        (wime--commit "便宜")
        (let ((state (wime--composition-state)))
          (should (equal (alist-get 'context state) history))
          (should (equal (alist-get 'committed state)
                         '((text . "便宜") (input . "bmyi") (aux . [[0 "a"]]))))))
      (setq wime--input (apply #'concat (make-list 10 "x;uiB"))
            wime--candidate-records
            (list (wime--make-candidate :text (apply #'concat (make-list 10 "行事")) :end 20)))
      (wime--commit (wime-candidate-text (car wime--candidate-records)))
      (let* ((state (wime--composition-state)) (fixed (alist-get 'committed state)))
        (should (= (length (alist-get 'context state)) 64))
        (should (= (length (alist-get 'text fixed)) 22))
        (should (equal (alist-get 'input fixed) (concat "bmyi" (apply #'concat (make-list 10 "x;ui")))))
        (should (equal (aref (alist-get 'aux fixed) 10) [21 "b"])))
      (let ((wime-database-file "another-scheme.db"))
        (should-not (alist-get 'committed (wime--composition-state))))
      (setq-local wime-context-start-function (lambda () (- (point) 1)))
      (should-not (alist-get 'committed (wime--composition-state)))
      (setq-local wime-context-start-function nil)
      (insert " ")
      (should-not (alist-get 'committed (wime--composition-state)))
      (delete-char -1)
      (should-not (alist-get 'committed (wime--composition-state))))))

(ert-deftest wime-neural-request-snapshots-the-composition-state ()
  (let ((wime-user-database nil))
    (with-temp-buffer
      (insert "允许")
      (setq wime--input "bmyi"
            wime--candidate-records (list (wime--make-candidate :text "便宜" :end 2)))
      (wime--commit "便宜")
      (setq wime--input "x;ui"
            wime--candidate-records (list (wime--make-candidate :text "行事" :end 2 :score 0)))
      (let ((wime-neural-mode t) (wime-neural-model-file "model") (wime-neural--failed nil)
            (wime-neural-search-method 'verify) scheduled)
        (wime-neural-test--mspy
         (cl-letf (((symbol-function 'run-with-timer)
                    (lambda (_delay _repeat _function job) (setq scheduled job) nil)))
           (wime-neural--schedule))
         (let ((request (json-parse-string (wime-neural--request scheduled) :object-type 'alist)))
           (should (equal (alist-get 'encoding request) "wime-composition-v2"))
           (should (equal (alist-get 'context request) "允许"))
           (should (equal (alist-get 'input (alist-get 'committed request)) "bmyi"))
           (should (equal (alist-get 'input request) "x;ui"))))))))

(ert-deftest wime-neural-raw-commit-ends-the-confirmed-segment ()
  (let ((wime-user-database nil))
    (with-temp-buffer
      (setq wime--input "bmyi"
            wime--candidate-records (list (wime--make-candidate :text "便宜" :end 2)))
      (wime--commit "便宜")
      (setq wime--input "raw")
      (wime--commit nil)
      (should (equal (wime--composition-state) '((context . "便宜raw")))))))

(ert-deftest wime-neural-job-notices-history-changes-beyond-visible-tail ()
  (let ((wime-user-database nil))
    (with-temp-buffer
      (insert (make-string 64 ?史))
      (setq wime--input "bmyi"
            wime--candidate-records (list (wime--make-candidate :text "便宜" :end 2)))
      (wime--commit "便宜")
      (setq wime--input "x;ui"
            wime--candidate-records (list (wime--make-candidate :text "行事" :end 2 :score 0)))
      (let ((wime-neural-mode t) (wime-neural-model-file "model") (wime-neural--failed nil)
            (wime-neural-search-method 'verify) scheduled)
        (wime-neural-test--mspy
         (cl-letf (((symbol-function 'run-with-timer)
                    (lambda (_delay _repeat _function job) (setq scheduled job) nil)))
           (wime-neural--schedule))
         (should (wime-neural--valid-p scheduled))
         (narrow-to-region 2 (point-max))
         (should (equal (wime-neural--job-context scheduled) (wime--context)))
         (should-not (wime-neural--valid-p scheduled)))))))

(ert-deftest wime-neural-single-syllable-retains-contextual-grammar-ranking ()
  (with-temp-buffer
    (let ((wime-neural-mode t) (wime-neural--failed nil) (wime-neural-single-character nil)
          (wime-neural-model-file "model.safetensors") scheduled)
      (wime-neural-test--single-syllable
        (insert "真刀真")
        (setq wime--input "qd" wime--candidates (wime--candidates "qd"))
        (should (equal wime--candidates '("枪" "强")))
        (cl-letf (((symbol-function 'run-with-timer)
                   (lambda (&rest _) (setq scheduled t))))
          (dolist (wime-neural-search-method '(verify decode))
            (wime-neural--schedule)
            (should-not scheduled)
            (should (equal wime--candidates '("枪" "强")))))))))

(ert-deftest wime-neural-single-character-is-opt-in ()
  (with-temp-buffer
    (let ((wime-neural-mode t) (wime-neural--failed nil)
          (wime-neural-single-character t)
          (wime-neural-model-file "model.safetensors") scheduled)
      (wime-neural-test--single-syllable
        (insert "真刀真")
        (setq wime--input "qd" wime--candidates (wime--candidates "qd"))
        (cl-letf (((symbol-function 'run-with-timer)
                   (lambda (_delay _repeat _function job) (setq scheduled job) nil)))
          (let ((wime-neural-search-method 'verify))
            (wime-neural--schedule)
            (should (= (wime-neural--job-syllables scheduled) 1))
            (should (equal (wime-neural--job-context scheduled) "真刀真"))
            (setq scheduled nil wime--input "q" wime--candidates (wime--candidates "q"))
            (wime-neural--schedule)
            (should-not scheduled)))))))

(ert-deftest wime-neural-context-settings-remain-compatible ()
  (with-temp-buffer
    (insert "上段\n\n当前输入")
    (setq-local wime-neural-context-paragraph-only t)
    (should wime-context-paragraph-only)
    (should (equal (wime-neural--context) "当前输入"))
    (setq-local wime-neural-context-start-function (lambda () (- (point) 2)))
    (should (equal (wime--context) "输入"))))

(defmacro wime-neural-test--composition (&rest body)
  (declare (indent 0))
  `(with-temp-buffer
     (let ((wime-neural-mode t) (wime-user-database nil) (wime-neural-seed-candidates nil)
           (wime-neural-prior-weight 0.1))
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

(ert-deftest wime-neural-shows-only-full-decoded-readings ()
  (wime-neural-test--composition
    (setf (wime-neural--job-method job) 'decode)
    (wime-neural--apply job '((candidates . [["甲" -1.0]])))
    (should (equal (car wime--candidates) "甲乙"))
    (wime-neural--apply job '((candidates . [["丙丁" -1.0]]) (elapsed_ms . 3.5)))
    (should (equal wime--candidates '("丙丁" "甲乙" "甲")))
    (should (= wime-neural-last-latency 3.5))))

(ert-deftest wime-neural-rejects-context-boundary-changes-without-text-edits ()
  (dolist (change '(overlay narrowing provider))
    (wime-neural-test--composition
      (insert "之前 当前")
      (setf (wime-neural--job-position job) (point)
            (wime-neural--job-tick job) (buffer-chars-modified-tick)
            (wime-neural--job-context job) (wime-neural--context))
      (should (wime-neural--valid-p job))
      (pcase change
        ('overlay (overlay-put (make-overlay 1 4) 'read-only t))
        ('narrowing (narrow-to-region 4 (point-max)))
        ('provider (setq-local wime-neural-context-start-function (lambda () 4))))
      (should (= (buffer-chars-modified-tick) (wime-neural--job-tick job)))
      (should-not (wime-neural--valid-p job))
      (setf (wime-neural--job-method job) 'decode)
      (wime-neural--apply job '((candidates . [["丙丁" -1.0]])))
      (should (equal (car wime--candidates) "甲乙")))))

(ert-deftest wime-neural-scheduled-request-contains-only-bounded-editable-context ()
  (wime-neural-test--composition
    (insert (propertize "提示符> " 'read-only t 'rear-nonsticky t) "这个函数")
    (let ((wime-neural-model-file "model.safetensors") (wime-neural--failed nil) scheduled)
      (cl-letf (((symbol-function 'run-with-timer)
                 (lambda (_delay _repeat _function job) (setq scheduled job) nil)))
        (wime-neural--schedule))
      (should (equal (wime-neural--job-context scheduled) "这个函数"))
      (should (equal (alist-get 'context (json-parse-string (wime-neural--request scheduled)
                                                           :object-type 'alist))
                     "这个函数")))))

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

(ert-deftest wime-neural-cuda-seed-promotes-only-a-verified-full-pool-member ()
  (wime-neural-test--composition
    (let ((wime-neural--pending nil))
      (setf (wime-neural--job-method job) 'verify
            (wime-neural--job-seed-candidates job) t
            (wime-neural--job-draft job) (car wime--candidate-records))
      (dolist (index '(2 -1 99 "丙丁" nil))
        (wime-neural--apply job `((verified_index . ,index) (score . -0.1)))
        (should (equal (car wime--candidates) "甲乙"))
        (should (= (wime-neural--job-round job) 0)))
      (wime-neural--apply job '((verified_index . 1) (score . -0.1)))
      (should (equal (car wime--candidates) "丙丁"))
      (should (eq (wime-neural--job-best job) (nth 1 (wime-neural--job-records job)))))))

(ert-deftest wime-neural-publishes-joint-scores-for-the-pool-and-drafts ()
  (wime-neural-test--composition
    (let ((wime-neural--pending nil))
      (setf (wime-neural--job-method job) 'verify
            (wime-neural--job-seed-candidates job) t
            (wime-neural--job-draft job) (car wime--candidate-records))
      (wime-neural--apply job '((verified_index . 1) (score . -0.2)
                                (candidate_scores . [[0 -0.5] [1 -0.2] [7 -0.1]])))
      (should (equal (car wime--candidates) "丙丁"))
      ;; Prior weight 0.1 times the per-syllable path score, as in selection.
      (should (equal (sort (copy-sequence wime--candidate-scores)
                           (lambda (a b) (string< (car a) (car b))))
                     '(("丙丁" . -0.2) ("丙丁" . -0.2) ("甲乙" . -0.45)))))))

(ert-deftest wime-neural-cuda-seed-is-only-requested-for-the-initial-round ()
  (wime-neural-test--composition
    (setf (wime-neural--job-seed-candidates job) t
          (wime-neural--job-method job) 'verify
          (wime-neural--job-draft job) (car wime--candidate-records))
    (let ((request (json-parse-string (wime-neural--request job) :object-type 'alist)))
      (should (eq (alist-get 'seed_candidates request) t))
      (should (= (alist-get 'score (aref (alist-get 'candidates request) 1)) 0)))
    (setf (wime-neural--job-round job) 1)
    (let ((request (json-parse-string (wime-neural--request job) :object-type 'alist)))
      (should (eq (alist-get 'seed_candidates request) :false)))
    ;; A late seed index must not be mistaken for the current constrained draft.
    (wime-neural--apply job '((verified_index . 1) (score . -0.1)))
    (should (equal (car wime--candidates) "甲乙"))))

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
    (setf (wime-neural--job-method job) 'decode)
    (wime-neural--apply job '((candidates . [["丙丁" -1.0]])))
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
    (setf (wime-neural--job-method job) 'decode)
    (wime-neural--apply job '((candidates . [["丙丁" -1.0]])))
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
      (setf (wime-neural--job-method job) 'decode)
      (unwind-protect
          (progn
            (wime-neural--filter process "{\"id\":9,\"candidates\":[[\"丙丁\",-1]]}\n{\"id\":1,")
            (should (eq wime-neural--busy job))
            (should (equal (car wime--candidates) "甲乙"))
            (wime-neural--filter process "\"candidates\":[[\"丙丁\",-1]],\"elapsed_ms\":2}\n")
            (should-not wime-neural--busy)
            (should (equal (car wime--candidates) "丙丁")))
        (delete-process process)))))

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

(ert-deftest wime-neural-rescorer-travels-with-checkpoint-recipes ()
  (require 'wime-checkpoint)
  (let ((wime-neural-program "/bin/true") (wime-neural-interpreter "/usr/bin/python3")
        (wime-japanese-model-file nil) (wime-japanese-table-file nil)
        wime-neural-rescorer wime-neural-rescorer-weight wime-neural-model-file
        wime-search-width wime-neural-inference-budget wime-neural-prior-weight
        wime-neural-prior-confidence
        wime-neural-single-character wime-neural-seed-candidates wime-neural-search-method
        wime-neural-pool-width)
    (wime-checkpoint--apply '(:path "/m.safetensors"
                              :profile (:search_method "verify" :width 16 :budget 4 :rescorer "/lm" :rescorer_weight 2
                                        :pool_width 32)))
    (should (equal (wime-neural--command wime-neural-model-file)
                   '("/usr/bin/python3" "/bin/true" "--rescorer" "/lm"
                     "--rescorer-weight" "2" "/m.safetensors")))
    (should (equal (plist-get (wime-checkpoint-profile) :rescorer) "/lm"))
    (should (= wime-neural-pool-width 32))
    ;; A conditional-only profile disables the general rescorer.
    (wime-checkpoint--apply '(:path "/m.safetensors" :profile (:search_method "verify" :width 16 :budget 4)))
    (should-not wime-neural-rescorer)
    (should (= wime-neural-rescorer-weight 1.5))
    (should (eq (plist-get (wime-checkpoint-profile) :rescorer) :null))))

(ert-deftest wime-neural-wider-pool-is-searched-once-apart-from-typing ()
  (wime-neural-test--composition
    (let* ((wime-search-width 16) (wime-neural-pool-width 32)
           (wime--search-cache 'typing) (wime-neural--pool-cache nil)
           (wide (list (wime--make-candidate :text "戊己" :end 2 :score -9)))
           (process (make-pipe-process :name "wime-pool-test" :noquery t))
           (wime-neural--process process) (wime-neural--busy nil) (wime-neural--pending nil)
           (wime-neural--failed nil) (wime-neural--timeout-timer nil) searches)
      (unwind-protect
          (cl-letf (((symbol-function 'wime--sentence-records)
                     (lambda (_units &optional _prefix)
                       (push (cons wime-search-width wime--search-cache) searches)
                       (setq wime--search-cache 'pool)
                       wide))
                    ((symbol-function 'wime-neural--start) (lambda () process))
                    ((symbol-function 'run-with-timer) #'ignore)
                    ((symbol-function 'process-send-string) #'ignore))
            (setf (wime-neural--job-seed-candidates job) t)
            (wime-neural--enqueue job)
            (should (equal searches '((32))))
            (should (eq wime--search-cache 'typing))
            (should (eq wime-neural--pool-cache 'pool))
            (should (equal (alist-get 'text (aref (alist-get 'candidates (json-parse-string
                                                                         (wime-neural--request job)
                                                                         :object-type 'alist))
                                                  0))
                           "戊己"))
            (wime-neural--verify-apply job '((verified_index . 0) (score . -1.0) (prefixes . [])))
            (should (equal (car wime--candidates) "戊己")))
        (delete-process process)))))

(ert-deftest wime-neural-learned-readings-earn-the-bonus-and-pool-scores-map-to-the-pool ()
  (wime-neural-test--composition
    (let* ((wime-user-database (make-temp-file "wime-user" nil ".sqlite3")) (wime--user-db nil)
           (wime--memory (make-hash-table :test #'equal)) (wime-neural-memory-weight 10.0)
           (pool (list (wime--make-candidate :text "丙丁" :end 2 :score 0)
                       (wime--make-candidate :text "甲乙" :end 2 :score 1))))
      (unwind-protect
          (progn
            (wime--user-db)
            (puthash "aabb" '(("丙丁" . "")) wime--memory)
            (setf (wime-neural--job-pool job) pool (wime-neural--job-seed-candidates job) t)
            (should (equal (mapcar (lambda (c) (alist-get 'bonus c))
                                   (alist-get 'candidates (json-parse-string (wime-neural--request job)
                                                                             :object-type 'alist :array-type 'list)))
                           '(10.0 0.0)))
            (wime-neural--verify-apply job '((verified_index . 1) (score . -1.0) (prefixes . [])
                                             (candidate_scores . [[0 -2.0] [1 -1.0]])))
            ;; Worker indices name pool members, not the display records.
            (should (assoc "丙丁" (wime-neural--job-scored job)))
            (should (= (cdr (assoc "丙丁" (wime-neural--job-scored job))) 8.0)))
        (sqlite-close (cdr wime--user-db))
        (delete-file wime-user-database)))))

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

(ert-deftest wime-neural-confident-seed-keeps-the-same-prior-for-later-rounds ()
  (wime-neural-test--composition
    (let ((wime-neural--pending nil)
          (wime-neural-inference-budget 4)
          (candidate (wime--make-candidate :text "丙丁" :end 2 :score -100)))
      (setf (wime-neural--job-seed-candidates job) t
            (wime-neural--job-prior-confidence job) .5
            (wime-neural--job-prior-weight job) .1)
      (cl-letf (((symbol-function 'wime--sentence-records)
                 (lambda (&rest _) (list candidate))))
        (wime-neural--verify-apply
         job '((verified_index . 0) (score . -.05) (effective_prior_weight . 0)
               (prefixes . ("丙")))))
      (should (= (wime-neural--job-prior-weight job) 0))
      (should (= (wime-neural--job-best-score job) -.05))
      (let ((next wime-neural--pending))
        (setq wime-neural--pending nil)
        (should (= (wime-neural--job-prior-weight next) 0))
        (should (= (alist-get 'prior_weight
                              (json-parse-string (wime-neural--request next) :object-type 'alist)) 0))
        (wime-neural--verify-apply next '((score . -.04)))
        (should (equal (car wime--candidates) "丙丁"))))))

(ert-deftest wime-neural-prior-cannot-be-disabled-without-opt-in ()
  (wime-neural-test--composition
    (setf (wime-neural--job-seed-candidates job) t
          (wime-neural--job-prior-confidence job) 1.0
          (wime-neural--job-prior-weight job) .1)
    (wime-neural--verify-apply
     job '((verified_index . 0) (score . -.05) (effective_prior_weight . 0)))
    (should (= (wime-neural--job-prior-weight job) .1))))

(ert-deftest wime-neural-decode-sends-the-whole-input-with-pins-and-learned-readings ()
  (with-temp-buffer
    (let* ((wime-neural-mode t) (wime-neural--failed nil) (wime-neural-model-file "model")
           (wime-neural-search-method 'decode) (wime-neural-rescorer "/lm")
           (wime-neural-memory-weight 10.0)
           (wime-user-database (make-temp-file "wime-user" nil ".sqlite3")) (wime--user-db nil)
           (wime--memory (make-hash-table :test #'equal)) scheduled)
      (unwind-protect
          (progn
            (wime--user-db)
            (puthash "aabb" '(("甲乙" . "")) wime--memory)
            (cl-letf (((symbol-function 'run-with-timer)
                       (lambda (_delay _repeat _function job) (setq scheduled job) nil))
                      ((symbol-function 'sqlite-select) (lambda (&rest _) '(("mspy")))))
              ;; A half syllable has no lattice sentence, but decodes.
              (setq wime--input "aabbc" wime--candidate-records nil wime--pins '((0 ?甲 "aa")))
              (wime-neural--schedule)
              (should (eq (wime-neural--job-method scheduled) 'decode))
              (let ((request (json-parse-string (wime-neural--request scheduled)
                                                :object-type 'alist :array-type 'list)))
                (should (equal (alist-get 'operation request) "decode"))
                (should (equal (alist-get 'input request) "aabbc"))
                (should (= (alist-get 'syllables request) 3))
                (should (equal (alist-get 'pins request) '((0 "甲"))))
                (should (equal (alist-get 'learned request) '((0 "甲乙"))))
                (should (= (alist-get 'memory_weight request) 10.0)))
              ;; Auxiliary codes constrain their syllable, in lower case.
              (setq wime--input "aaBCbbD" scheduled nil wime--pins nil
                    wime--candidate-records (list (wime--make-candidate :text "甲乙" :end 2 :score 0)))
              (wime-neural--schedule)
              (should (eq (wime-neural--job-method scheduled) 'decode))
              (let ((request (json-parse-string (wime-neural--request scheduled)
                                                :object-type 'alist :array-type 'list)))
                (should (equal (alist-get 'input request) "aabb"))
                (should (equal (alist-get 'aux request) '((0 "bc") (1 "d")))))
              ;; Without a rescorer the lattice is verified.
              (let ((wime-neural-rescorer nil))
                (setq wime--input "aabb" scheduled nil)
                (wime-neural--schedule)
                (should (eq (wime-neural--job-method scheduled) 'verify)))
              (setq wime--input "a" scheduled nil wime--candidate-records nil)
              (wime-neural--schedule)
              (should-not scheduled)))
        (sqlite-close (cdr wime--user-db))
        (delete-file wime-user-database)))))

(ert-deftest wime-neural-decode-publishes-checked-readings-ahead-of-the-lattice ()
  (wime-neural-test--composition
    (setf (wime-neural--job-method job) 'decode)
    (wime-neural--apply job '((id . 1) (elapsed_ms . 3.0)
                              (candidates . [["丙丁" -1.0] ["甲" -0.5] ["ab" -0.1] ["戊己" -2.0]])))
    (should (equal wime--candidates '("丙丁" "戊己" "甲乙" "甲")))
    (should (equal (mapcar #'wime-candidate-text wime--candidate-records) '("丙丁" "戊己" "甲乙" "甲")))
    (should (equal wime--candidate-scores '(("丙丁" . -1.0) ("戊己" . -2.0))))
    (should (equal wime--default-reading "丙丁"))
    (should (= wime-neural-last-latency 3.0))
    ;; A reading that breaks a pin is never shown.
    (setq wime--pins '((0 ?甲 "aa")) wime--candidates '("甲乙" "甲")
          wime--candidate-records (wime-neural--job-records job))
    (wime-neural--apply job '((id . 1) (candidates . [["丙丁" -1.0]])))
    (should (equal wime--candidates '("甲乙" "甲")))))

(ert-deftest wime-neural-decode-travels-with-checkpoint-recipes ()
  (require 'wime-checkpoint)
  (let ((wime-neural-program "/bin/true") (wime-neural-interpreter nil) (wime-database-file "/w.db")
        (wime-japanese-model-file nil) (wime-japanese-table-file nil)
        wime-neural-rescorer wime-neural-rescorer-weight wime-neural-model-file
        wime-search-width wime-neural-inference-budget wime-neural-prior-weight
        wime-neural-prior-confidence
        wime-neural-single-character wime-neural-seed-candidates wime-neural-search-method
        wime-neural-pool-width)
    (wime-checkpoint--apply '(:path "/m.safetensors"
                              :profile (:width 16 :budget 4 :rescorer "/lm" :search_method "decode")))
    (should (eq wime-neural-search-method 'decode))
    (should (equal (wime-neural--command wime-neural-model-file)
                   '("/bin/true" "--rescorer" "/lm" "--rescorer-weight" "1.5"
                     "--database" "/w.db" "--beam" "16" "/m.safetensors")))
    (should (equal (plist-get (wime-checkpoint-profile) :search_method) "decode"))
    (wime-checkpoint--apply '(:path "/m.safetensors" :profile (:search_method "verify" :width 16 :budget 4 :rescorer "/lm")))
    (should (eq wime-neural-search-method 'verify))
    (should-not (member "--database" (wime-neural--command wime-neural-model-file)))))

(ert-deftest wime-neural-decode-request-carries-typed-tones ()
  (wime-neural-test--composition
    (setq wime--input "aa3bb")
    (setf (wime-neural--job-input job) wime--input
          (wime-neural--job-method job) 'decode)
    (let ((request (json-parse-string (wime-neural--request job) :object-type 'alist)))
      (should (equal (alist-get 'input request) "aabb"))
      (should (equal (alist-get 'tones request) [[0 3]])))))

(ert-deftest wime-neural-decode-request-carries-complete-lattice-readings ()
  (wime-neural-test--composition
    (setf (wime-neural--job-method job) 'decode)
    (let ((request (json-parse-string (wime-neural--request job) :object-type 'alist :array-type 'list)))
      (should (equal (mapcar (lambda (c) (alist-get 'text c)) (alist-get 'candidates request))
                     '("甲乙" "丙丁"))))))
