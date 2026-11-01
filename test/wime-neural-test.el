;;; wime-neural-test.el --- Async candidate lifecycle tests -*- lexical-binding: t; -*-
(require 'ert)
(require 'wime-neural)

(defvar wime-japanese-model-file)
(defvar wime-japanese-table-file)

(defmacro wime-neural-test--database (&rest body)
  "Run BODY with a database reporting model features."
  (declare (indent 0))
  `(let ((wime--position-keys (cons wime--default-keys wime--default-keys)))
     (cl-letf (((symbol-function 'wime--database) (lambda () 'test-database))
               ((symbol-function 'sqlite-select) (lambda (&rest _) '(("wime-syllable-features-v1")))))
       ,@body)))

(defmacro wime-neural-test--single-syllable (&rest body)
  "Run BODY with a small contextual dictionary for a single syllable."
  (declare (indent 0))
  `(wime-neural-test--database
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

(ert-deftest wime-neural-reads-a-longer-history-than-the-lattice ()
  (with-temp-buffer
    (insert (make-string 2000 ?史))
    (should (= (length (wime--context)) 64))
    (should (= (length (wime-neural--context)) wime-neural-context-limit))
    (should (= (length (alist-get 'context (wime--composition-state wime-neural-context-limit)))
               wime-neural-context-limit))))

(ert-deftest wime-neural-history-starts-with-the-context-prefix ()
  (with-temp-buffer
    (insert "我想")
    (setq-local wime-context-prefix-function (lambda () "对方：明天去哪\n我："))
    (should (equal (wime--context) "我想"))
    (should (equal (wime-neural--context) "对方：明天去哪\n我：我想"))
    (should (equal (alist-get 'context (wime-neural--state)) "对方：明天去哪\n我：我想"))
    (setq-local wime-context-prefix-function (lambda () (error "client failure")))
    (wime-neural--forget-prefix)
    (should (equal (wime-neural--context) "我想"))))

(ert-deftest wime-neural-request-snapshots-the-composition-state ()
  (let ((wime-user-database nil))
    (with-temp-buffer
      (insert "允许")
      (setq wime--input "bmyi"
            wime--candidate-records (list (wime--make-candidate :text "便宜" :end 2)))
      (wime--commit "便宜")
      (setq wime--input "x;ui"
            wime--candidate-records (list (wime--make-candidate :text "行事" :end 2 :score 0)))
      (let ((wime-decoding 'hybrid) (wime-neural-model-file "model") (wime-neural--failed nil)
            (wime-neural-rescorer "/lm") scheduled)
        (wime-neural-test--database
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
      (insert (make-string wime-neural-context-limit ?史))
      (setq wime--input "bmyi"
            wime--candidate-records (list (wime--make-candidate :text "便宜" :end 2)))
      (wime--commit "便宜")
      (setq wime--input "x;ui"
            wime--candidate-records (list (wime--make-candidate :text "行事" :end 2 :score 0)))
      (let ((wime-decoding 'hybrid) (wime-neural-model-file "model") (wime-neural--failed nil)
            (wime-neural-rescorer "/lm") scheduled)
        (wime-neural-test--database
         (cl-letf (((symbol-function 'run-with-timer)
                    (lambda (_delay _repeat _function job) (setq scheduled job) nil)))
           (wime-neural--schedule))
         (should (wime-neural--valid-p scheduled))
         (narrow-to-region 2 (point-max))
         (should (equal (wime-neural--job-context scheduled) (wime-neural--context)))
         (should-not (wime-neural--valid-p scheduled)))))))

(ert-deftest wime-neural-single-syllable-retains-contextual-grammar-ranking ()
  (with-temp-buffer
    (let ((wime-decoding 'hybrid) (wime-neural--failed nil) (wime-neural-single-character nil)
          (wime-neural-model-file "model.safetensors") scheduled)
      (wime-neural-test--single-syllable
        (insert "真刀真")
        (setq wime--input "qd" wime--candidates (wime--candidates "qd"))
        (should (equal wime--candidates '("枪" "强")))
        (cl-letf (((symbol-function 'run-with-timer)
                   (lambda (&rest _) (setq scheduled t))))
          (progn
            (wime-neural--schedule)
            (should-not scheduled)
            (should (equal wime--candidates '("枪" "强")))))))))

(ert-deftest wime-neural-single-character-is-opt-in ()
  (with-temp-buffer
    (let ((wime-decoding 'hybrid) (wime-neural--failed nil)
          (wime-neural-single-character t)
          (wime-neural-model-file "model.safetensors") scheduled)
      (wime-neural-test--single-syllable
        (insert "真刀真")
        (setq wime--input "qd" wime--candidates (wime--candidates "qd"))
        (cl-letf (((symbol-function 'run-with-timer)
                   (lambda (_delay _repeat _function job) (setq scheduled job) nil)))
          (let ((wime-neural-rescorer "/lm"))
            (wime-neural--schedule)
            (should (= (wime-neural--job-syllables scheduled) 1))
            (should (equal (wime-neural--job-context scheduled) "真刀真"))
            (setq scheduled nil wime--input "q" wime--candidates (wime--candidates "q"))
            (wime-neural--schedule)
            (should-not scheduled)))))))

(ert-deftest wime-decoding-chooses-the-lattice-the-model-or-both ()
  (with-temp-buffer
    (let ((wime-decoding 'hybrid) (wime-neural--failed nil) (wime-neural-single-character t)
          (wime-neural-model-file "model.safetensors") (wime-neural-rescorer "/lm")
          (wime-lattice-inhibit-functions '(wime-neural--exclusive-p)))
      (wime-neural-test--single-syllable
        (cl-letf (((symbol-function 'run-with-timer) (lambda (&rest _) nil)))
          (pcase-dolist (`(,decoding ,lattice ,decoded) '((lattice t nil) (hybrid t t) (neural nil t)))
            (let ((wime-decoding decoding))
              (setq wime--input "qd" wime--candidates (wime--candidates "qd"))
              (should (eq (and wime--candidates wime--candidate-records t) lattice))
              (wime-neural--schedule)
              (should (eq (and wime-neural--job t) decoded))))
          ;; Without a worker, neural decoding converts with the lattice.
          (let ((wime-decoding 'neural) (wime-neural--failed t))
            (should (wime--candidates "qd"))))))))

(defmacro wime-neural-test--composition (&rest body)
  (declare (indent 0))
  `(wime-neural-test--database
     (with-temp-buffer
     (let ((wime-decoding 'hybrid) (wime-user-database nil))
       (setq wime--input "aabb"
             wime--selected 0
             wime--candidate-records
             (list (wime--make-candidate :text "甲乙" :end 2 :score 1)
                   (wime--make-candidate :text "丙丁" :end 2 :score 0)
                   (wime--make-candidate :text "甲" :end 1 :score 2))
             wime--candidates '("甲乙" "丙丁" "甲"))
       (let ((job (wime-neural--make-job
                   :id 1 :buffer (current-buffer) :input wime--input
                   :position (point)
                   :context "" :records wime--candidate-records :syllables 2 :method 'decode)))
         (unwind-protect (progn ,@body) (wime--reset)))))))

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
            (wime-neural--job-context job) (wime-neural--context))
      (should (wime-neural--valid-p job))
      (pcase change
        ('overlay (overlay-put (make-overlay 1 4) 'read-only t))
        ('narrowing (narrow-to-region 4 (point-max)))
        ('provider (setq-local wime-context-start-function (lambda () 4))))
      (should-not (wime-neural--valid-p job))
      (setf (wime-neural--job-method job) 'decode)
      (wime-neural--apply job '((candidates . [["丙丁" -1.0]])))
      (should (equal (car wime--candidates) "甲乙")))))

(ert-deftest wime-neural-busy-chat-keeps-the-decode-until-the-composer-changes ()
  (wime-neural-test--composition
    (insert (propertize "旧消息\n> " 'read-only t 'rear-nonsticky t) "我想")
    (let ((wime-neural-model-file "model.safetensors") (wime-neural--failed nil)
          (messages "对方：明天去哪\n我：") scheduled)
      (setq-local wime-context-prefix-function (lambda () messages))
      (cl-letf (((symbol-function 'run-with-timer)
                 (lambda (_delay _repeat _function job) (setq scheduled job) nil)))
        (wime-neural--schedule))
      (should (equal (wime-neural--job-context scheduled) "对方：明天去哪\n我：我想"))
      ;; A new row lands above the composer and the timeline's context moves on.
      (let ((inhibit-read-only t))
        (save-excursion (goto-char (point-min)) (insert (propertize "新消息\n" 'read-only t))))
      (setq messages "对方：新消息\n我：")
      (should (wime-neural--valid-p scheduled))
      ;; Editing the draft before point still voids it.
      (save-excursion (goto-char (- (point) 1)) (insert "去"))
      (should-not (wime-neural--valid-p scheduled))
      ;; The next composition takes the newer messages.
      (wime--reset)
      (should (equal (wime-neural--context) "对方：新消息\n我：我去想")))))

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

(ert-deftest wime-neural-requires-database-phonetic-features ()
  (wime-neural-test--composition
    (let ((wime-neural-model-file "model.safetensors") (wime-neural--failed nil)
          (encoding "wime-syllable-features-v1") scheduled)
      (cl-letf (((symbol-function 'wime--database) (lambda () 'database))
                ((symbol-function 'sqlite-select)
                 (lambda (&rest _) (list (list encoding))))
                ((symbol-function 'run-with-timer) (lambda (&rest _) (setq scheduled t))))
        (wime-neural--schedule)
        (should scheduled)
        (setq scheduled nil encoding "unknown")
        (wime-neural--schedule)
        (should-not scheduled)))))

(ert-deftest wime-switching-databases-stops-the-old-worker-and-clears-key-metadata ()
  (with-temp-buffer
    (let ((wime-database-file "/old.db") (wime--db nil)
          (wime--position-keys '("a" . "b"))
          (wime-neural--process nil) (wime-neural--busy t) (wime-neural--pending t)
          (wime-neural--timeout-timer nil))
      (wime-select-database "/new.db")
      (should (equal wime-database-file "/new.db"))
      (should-not wime-neural--busy)
      (should-not wime-neural--pending)
      (should-not wime--position-keys))))

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

(ert-deftest wime-neural-decoding-choices-wait-for-the-model ()
  (wime-neural-test--composition
    (let* ((wime-decoding 'neural) (wime-settle-hook '(wime-neural--await))
           (process (make-pipe-process :name "wime-await-test" :noquery t))
           (wime-neural--process process) (wime-neural--busy nil) (wime-neural--pending nil)
           (wime-neural--failed nil) (wime-neural--timeout-timer nil) sent)
      (setq wime--candidate-records nil wime--candidates nil
            wime-neural--job job wime-neural--timer (run-with-timer 60 nil #'ignore))
      (setf (wime-neural--job-records job) nil)
      (unwind-protect
          (cl-letf (((symbol-function 'wime-neural--start) (lambda () process))
                    ((symbol-function 'process-send-string) (lambda (_ request) (push request sent)))
                    ((symbol-function 'accept-process-output)
                     (lambda (&rest _) (wime-neural--reply '((id . 1) (candidates . [["丙丁" -1.0]]))))))
            (wime--commit-selected)
            (should sent)
            (should (equal (buffer-string) "丙丁")))
        (delete-process process)))))

(ert-deftest wime-neural-decoding-failure-returns-to-the-lattice ()
  (wime-neural-test--composition
    (let ((wime-neural--process nil) (wime-neural--pending nil) refreshed)
      (cl-letf (((symbol-function 'wime--refresh) (lambda () (setq refreshed t))))
        (dolist (decoding '(hybrid neural))
          (let ((wime-decoding decoding) (wime-neural--busy job) (wime-neural--failed nil))
            (wime-neural--fail "test failure")
            (should (eq refreshed (eq decoding 'neural)))))
        ;; A worker lost between requests leaves keys shown until a choice.
        (let ((wime-decoding 'neural) (wime-settle-hook '(wime-neural--await))
              (wime-neural--busy nil) (wime-neural--failed t))
          (setq refreshed nil wime-neural--job job wime--model-ranked nil)
          (run-hooks 'wime-settle-hook)
          (should refreshed))))))

(ert-deftest wime-neural-missing-model-files-are-quiet-only-for-hybrid ()
  (wime-neural-test--composition
    (dolist (decoding '(hybrid neural))
      (let ((wime-decoding decoding) (wime-neural-model-file "/missing/model.safetensors")
            (wime-neural-rescorer "/missing") (wime-neural--process nil)
            (wime-neural--busy nil) (wime-neural--pending nil) (wime-neural--failed nil) messages)
        (cl-letf (((symbol-function 'message) (lambda (&rest args) (push args messages)))
                  ((symbol-function 'wime--refresh) #'ignore)
                  ((symbol-function 'wime-neural--start) (lambda () (error "Started"))))
          (wime-neural--enqueue job))
        (should wime-neural--failed)
        (should (eq (and messages t) (eq decoding 'neural)))))))

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
            (should (equal delays '(2.2 10.2))))
        (delete-process process)))))

(ert-deftest wime-neural-real-worker-protocol ()
  (let ((command (getenv "WIME_NEURAL_COMMAND"))
        (model (getenv "WIME_NEURAL_MODEL")))
    (unless (and command model) (ert-skip "model and worker not configured"))
    (wime-neural-test--composition
      (let ((wime-neural-command (json-parse-string command :array-type 'list))
            (wime-neural-directory (getenv "WIME_NEURAL_DIRECTORY")) (wime-neural-model-file model)
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

(ert-deftest wime-neural-decode-sends-the-whole-input-with-pins-and-learned-readings ()
  (with-temp-buffer
    (let* ((wime-decoding 'hybrid) (wime-neural--failed nil) (wime-neural-model-file "model")
           (wime-neural-rescorer "/lm")
           (wime-neural-memory-weight 10.0)
           (wime-user-database (make-temp-file "wime-user" nil ".sqlite3")) (wime--user-db nil) (wime-learn t)
           (wime--memory (make-hash-table :test #'equal)) scheduled)
      (unwind-protect
          (progn
            (wime--user-db)
            (puthash "aabb" '(("甲乙" . "")) wime--memory)
            (cl-letf (((symbol-function 'run-with-timer)
                       (lambda (_delay _repeat _function job) (setq scheduled job) nil))
                      ((symbol-function 'sqlite-select) (lambda (&rest _) '(("wime-syllable-features-v1")))))
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
              ;; Without a rescorer the lattice stays usable, without a job.
              (let ((wime-neural-rescorer nil))
                (setq wime--input "aabb" scheduled nil)
                (wime-neural--schedule)
                (should-not scheduled))
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
  (let ((wime-neural-command '("python" "-m" "wime_runtime.worker")) (wime-database-file "/w.db")
        (wime-japanese-model-file nil) (wime-japanese-table-file nil)
        wime-neural-rescorer wime-neural-rescorer-weight wime-neural-model-file
        wime-search-width wime-neural-single-character wime-neural-beam)
    ;; Saved verification settings no longer select a separate runtime path.
    (wime-checkpoint--apply
     '(:path "/m.safetensors" :profile (:width 16 :budget 4 :search_method "verify"
                                        :pool_width 32 :rescorer "/lm")))
    (should (equal (wime-neural--command wime-neural-model-file)
                   '("python" "-m" "wime_runtime.worker" "--rescorer" "/lm" "--rescorer-weight" "1.5"
                     "--database" "/w.db" "--beam" "32" "/m.safetensors")))
    (should-not (plist-member (wime-checkpoint-profile) :search_method))
    (wime-checkpoint--apply
     '(:path "/m.safetensors" :profile (:width 16 :rescorer :null)))
    (should-not wime-neural-rescorer)
    (should (eq (plist-get (wime-checkpoint-profile) :rescorer) :null))))

(ert-deftest wime-neural-decode-request-carries-typed-tones ()
  (wime-neural-test--composition
    (setq wime--input "aa3bb")
    (setf (wime-neural--job-input job) wime--input
          (wime-neural--job-method job) 'decode)
    (let ((request (json-parse-string (wime-neural--request job) :object-type 'alist)))
      (should (equal (alist-get 'input request) "aabb"))
      (should (equal (alist-get 'tones request) [[0 3]])))))

(ert-deftest wime-neural-default-rescorer-is-the-best-installed ()
  (let ((data (make-temp-file "wime-rescorers-" t)))
    (unwind-protect
        (let ((small (expand-file-name "rescorers/Qwen3.5-0.8B-Base" data))
              (large (expand-file-name "rescorers/Qwen3.5-2B-Base" data)))
          (should-not (wime-neural-installed-rescorer data))
          (make-directory small t)
          (should (equal (wime-neural-installed-rescorer data) small))
          (make-directory large t)
          (should (equal (wime-neural-installed-rescorer data) large)))
      (delete-directory data t))))
