;;; wime-edit-neural-test.el --- Editable composition async tests -*- lexical-binding: t; -*-

(require 'ert)
(require 'wime-neural)
(require 'wime-edit)
(defvar wime-edit-mode)
(defvar wime-edit--generation)

(defmacro wime-edit-neural-test--composition (&rest body)
  "Run BODY with an editable composition and a current neural job.
Stub the editor interface so these tests need neither a dictionary nor a
worker process, and can check each asynchronous invalidation separately."
  (declare (indent 0))
  `(with-temp-buffer
     (insert "前文甲乙")
     (let ((wime-neural-mode t)
           (wime-edit-mode t)
           (wime-edit--generation 7)
           (active t) (writable t) (decodable t) (context "前文") published)
       (setq wime--input "aabb"
             wime--selected 0
             wime--candidate-records
             (list (wime--make-candidate :text "甲乙" :end 2 :score 1)
                   (wime--make-candidate :text "丙丁" :end 2 :score 0))
             wime--candidates '("甲乙" "丙丁"))
       (cl-letf (((symbol-function 'wime-edit--active-p) (lambda () active))
                 ((symbol-function 'wime-edit--writable-p) (lambda () writable))
                 ((symbol-function 'wime-edit--decodable-p) (lambda () decodable))
                 ((symbol-function 'wime--context) (lambda (&optional _limit) context))
                 ((symbol-function 'wime--display) (lambda () (setq published t))))
         (let ((job (wime-neural--make-job
                     :id 1 :buffer (current-buffer) :input wime--input
                     :position (point) :tick (buffer-chars-modified-tick)
                     :context context :records wime--candidate-records
                     :syllables 2 :edit-generation wime-edit--generation)))
           ,@body)))))

(ert-deftest wime-edit-neural-schedule-captures-run-generation ()
  (wime-edit-neural-test--composition
    (let ((wime-neural-model-file "model")
          (wime-neural--failed nil)
          (wime-neural-search-method 'rerank) scheduled)
      (cl-letf (((symbol-function 'run-with-timer)
                 (lambda (_delay _repeat _function request)
                   (setq scheduled request) nil)))
        (wime-neural--schedule))
      (should scheduled)
      (should (= (wime-neural--job-edit-generation scheduled) 7))
      (should (equal (wime-neural--job-context scheduled) "前文"))
      (backward-char)
      (should (wime-neural--valid-p scheduled)))))

(ert-deftest wime-edit-neural-publishes-after-cursor-movement ()
  (wime-edit-neural-test--composition
    (backward-char)
    (should (wime-neural--valid-p job))
    (let ((position (point)))
      (wime-neural--apply job '((promoted . 1) (elapsed_ms . 3.5)))
      (should published)
      (should (equal wime--candidates '("丙丁" "甲乙")))
      (should (= wime-neural-last-latency 3.5))
      (should (= (point) position)))))

(ert-deftest wime-edit-neural-waits-for-complete-middle-syllable ()
  (wime-edit-neural-test--composition
    (let ((wime-neural-model-file "model")
          (wime-neural--failed nil)
          (wime-neural-search-method 'rerank) scheduled)
      (cl-letf (((symbol-function 'run-with-timer)
                 (lambda (_delay _repeat _function request)
                   (setq scheduled request) nil)))
        ;; Candidate coverage alone cannot detect the lost syllable boundary.
        (setq decodable nil)
        (wime-neural--schedule)
        (should-not scheduled)
        (setq decodable t)
        (wime-neural--schedule)
        (should scheduled)))))

(ert-deftest wime-edit-neural-rejects-replaced-or-suspended-runs ()
  (dolist (change '(generation inactive disabled protected))
    (wime-edit-neural-test--composition
      (pcase change
        ('generation (cl-incf wime-edit--generation))
        ('inactive (setq active nil))
        ('disabled (setq wime-edit-mode nil))
        ('protected (setq writable nil)))
      ;; The keys, candidates and buffer can all be unchanged here.
      (should (= (buffer-chars-modified-tick) (wime-neural--job-tick job)))
      (should-not (wime-neural--valid-p job))
      (wime-neural--apply job '((promoted . 1)))
      (should-not published)
      (should (equal (car wime--candidates) "甲乙")))))

(ert-deftest wime-edit-neural-retains-existing-staleness-checks ()
  (dolist (change '(text input records context selected))
    (wime-edit-neural-test--composition
      (pcase change
        ('text (insert "新"))
        ('input (setq wime--input "aabbA"))
        ('records (setq wime--candidate-records
                        (copy-sequence wime--candidate-records)))
        ('context (setq context "不同前文"))
        ('selected (setq wime--selected 1)))
      (should-not (wime-neural--valid-p job))
      (wime-neural--apply job '((promoted . 1)))
      (should-not published))))

(ert-deftest wime-edit-neural-verification-continuation-retains-generation ()
  (wime-edit-neural-test--composition
    (let ((wime-neural--pending nil)
          (wime-neural-inference-budget 2))
      (setf (wime-neural--job-method job) 'verify
            (wime-neural--job-draft job) (car wime--candidate-records))
      (cl-letf (((symbol-function 'wime--sentence-records)
                 (lambda (&rest _)
                   (list (wime--make-candidate :text "丙乙" :end 2 :score 0)))))
        (backward-char)
        (wime-neural--apply job '((score . -2.0) (prefixes . ["丙"])))
        (should wime-neural--pending)
        (should (= (wime-neural--job-edit-generation wime-neural--pending) 7))
        (should (wime-neural--valid-p wime-neural--pending))
        (cl-incf wime-edit--generation)
        (should-not (wime-neural--valid-p wime-neural--pending))))))

(ert-deftest wime-edit-neural-legacy-composition-still-requires-point ()
  (wime-edit-neural-test--composition
    (setf (wime-neural--job-edit-generation job) nil)
    (should (wime-neural--valid-p job))
    (backward-char)
    (should-not (wime-neural--valid-p job))))

(ert-deftest wime-edit-neural-japanese-composition-still-requires-point ()
  (wime-edit-neural-test--composition
    (setf (wime-neural--job-method job) 'japanese)
    (cl-letf (((symbol-function 'wime-japanese--current-p) (lambda (_job) t)))
      (setq active nil writable nil)
      (should (wime-neural--valid-p job))
      (backward-char)
      (should-not (wime-neural--valid-p job)))))

(defmacro wime-edit-neural-test--buffer (&rest body)
  "Run BODY using real editor metadata and rendering, without a worker."
  (declare (indent 0))
  `(with-temp-buffer
     (buffer-enable-undo)
     (insert "前文"
             (propertize "甲" 'wime-edit-unit
                         (wime-edit--make-unit :keys "aa" :original ?甲))
             (propertize "乙" 'wime-edit-unit
                         (wime-edit--make-unit :keys "bb" :original ?乙)))
     (let ((wime-neural-mode t) (wime-candidate-style 'list)
           (wime-user-database nil))
       (wime-edit-mode 1)
       (wime-edit--adopt 3 5)
       (setq wime--candidate-records
             (list (wime--make-candidate :text "甲乙" :end 2 :score 1)
                   (wime--make-candidate :text "丙丁" :end 2 :score 0))
             wime--candidates '("甲乙" "丙丁"))
       (wime--display)
       (undo-boundary)
       (let ((job (wime-neural--make-job
                   :id 1 :buffer (current-buffer) :input wime--input
                   :position (point) :tick (buffer-chars-modified-tick)
                   :context (wime-neural--context) :records wime--candidate-records
                   :syllables 2 :edit-generation wime-edit--generation)))
         ,@body))))

(ert-deftest wime-edit-neural-callback-updates-real-text-without-moving-cursor ()
  (wime-edit-neural-test--buffer
    (goto-char 4)
    (set-mark 5)
    (should (equal (wime-neural--job-context job) "前文"))
    (should (wime-neural--valid-p job))
    (wime-neural--apply job '((promoted . 1)))
    (should (equal (buffer-substring-no-properties 1 5) "前文丙丁"))
    (should (= (point) 4))
    (should (= (mark) 5))
    (should (= wime-edit--generation (wime-neural--job-edit-generation job)))
    (should (equal (mapcar #'wime-edit--unit-keys wime-edit--units) '("aa" "bb")))
    (undo-boundary)
    (undo-only 1)
    (should (equal (buffer-substring-no-properties 1 5) "前文甲乙"))
    (should (equal (wime-edit--unit-keys (wime-edit--unit-at 3)) "aa"))))

(ert-deftest wime-edit-neural-callback-rejects-new-editor-boundaries ()
  (dolist (change '(narrowing overlay hidden read-only deactivated database))
    (wime-edit-neural-test--buffer
      (pcase change
        ('narrowing (narrow-to-region 4 5))
        ('overlay (overlay-put (make-overlay 3 4) 'read-only t))
        ('hidden (overlay-put (make-overlay 3 4) 'invisible t))
        ('read-only (setq buffer-read-only t))
        ('deactivated (wime-deactivate))
        ('database (setq-local wime-database-file "another-scheme.db")))
      (should-not (wime-neural--valid-p job))
      (wime-neural--apply job '((promoted . 1)))
      (save-restriction
        (widen)
        (should (equal (buffer-substring-no-properties 1 5) "前文甲乙"))))))

(ert-deftest wime-edit-neural-real-unit-boundaries-gate-scheduling ()
  (wime-edit-neural-test--buffer
    (let ((wime-neural-model-file "model")
          (wime-neural--failed nil)
          (wime-neural-search-method 'rerank) scheduled)
      (cl-letf (((symbol-function 'run-with-timer)
                 (lambda (_delay _repeat _function request)
                   (setq scheduled request) nil)))
        (setq wime-edit--units
              (list (wime-edit--make-unit :keys "a")
                    (wime-edit--make-unit :keys "bb")))
        (wime-edit--load-input)
        (should (= (length (wime--parse wime--input)) 2))
        (wime-neural--schedule)
        (should-not scheduled)
        (setq wime-edit--units
              (list (wime-edit--make-unit :keys "aa")
                    (wime-edit--make-unit :keys "bb")))
        (wime-edit--load-input)
        (wime-neural--schedule)
        (should scheduled)))))

(ert-deftest wime-edit-active-keymap-stays-in-its-buffer ()
  (let ((emulation-mode-map-alists emulation-mode-map-alists)
        (overriding-terminal-local-map nil))
    (wime-edit-neural-test--buffer
      (wime-active-mode 1)
      (should (eq (key-binding (kbd "DEL")) #'wime-edit-delete-backward))
      (should (eq (key-binding (kbd "RET")) #'wime-edit-return))
      (should-not overriding-terminal-local-map)
      (with-temp-buffer
        (should-not (eq (key-binding (kbd "DEL")) #'wime-edit-delete-backward))
        (should-not (eq (key-binding (kbd "RET")) #'wime-edit-return))
        (should-not (eq (key-binding (kbd "C-c C-c")) #'wime-edit-finish))
        ;; Clearing another buffer must not remove the original run's keys.
        (wime--reset))
      (should (eq (key-binding (kbd "DEL")) #'wime-edit-delete-backward))
      (wime-edit-pause)
      (should-not (eq (key-binding (kbd "DEL")) #'wime-edit-delete-backward)))))

(ert-deftest wime-edit-active-keymap-precedes-modal-emulation ()
  (let* ((modal-map (make-sparse-keymap))
         (emulation-mode-map-alists (list (list (cons t modal-map))))
         (overriding-terminal-local-map nil))
    (define-key modal-map (kbd "DEL") #'ignore)
    (wime-edit-neural-test--buffer
      (wime-active-mode 1)
      (should (eq (key-binding (kbd "DEL")) #'wime-edit-delete-backward))
      (wime-edit-pause)
      (should (eq (key-binding (kbd "DEL")) #'ignore)))))

(ert-deftest wime-edit-neural-span-choice-invalidates-reused-candidates ()
  (wime-edit-neural-test--buffer
    ;; A cache may return the same records and a correction may keep the
    ;; displayed character.  Neither buffer tick nor record identity changes.
    (cl-letf (((symbol-function 'wime--refresh) #'ignore))
      (wime--choose-span 0 1 "甲"))
    (should (= (buffer-chars-modified-tick) (wime-neural--job-tick job)))
    (should (eq wime--candidate-records (wime-neural--job-records job)))
    (should (> wime-edit--generation (wime-neural--job-edit-generation job)))
    (should-not (wime-neural--valid-p job))))

;;; wime-edit-neural-test.el ends here
