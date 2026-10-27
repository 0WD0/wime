;;; wime-test.el --- Wime input method engine -*- lexical-binding: t; -*-

;;; Commentary:
;; The engine is pure Elisp, so these tests exercise the whole input path
;; without a module, a subprocess, or a graphical frame.

;;; Code:

(require 'ert)
(require 'wime)

(ert-deftest wime-context-uses-existing-text-at-point-not-only-ime-commits ()
  (with-temp-buffer
    (insert (propertize (concat (make-string 70 ?前) "这个函数") 'face 'bold) "后面的文字")
    (backward-char 5)
    (let ((wime--input "fhhv") (position (point)) (tick (buffer-chars-modified-tick)))
      (should (equal (wime--context) (concat (make-string 60 ?前) "这个函数")))
      (should-not (text-properties-at 0 (wime--context)))
      (should (= position (point)))
      (should (= tick (buffer-chars-modified-tick))))))

(ert-deftest wime-context-preserves-paragraphs-unless-explicitly-bounded ()
  (dolist (separator '("\n\n" "\n \t\n" "\f"))
    (with-temp-buffer
      (insert "上一段无关内容" separator "这个函数。\n返回")
      (should (equal (wime--context) (buffer-string)))
      (let ((wime-context-paragraph-only t))
        (should (equal (wime--context) "这个函数。\n返回"))
        (insert "\n\n")
        (should (equal (wime--context) ""))))))

(ert-deftest wime-context-respects-narrowing-and-field-edges ()
  (with-temp-buffer
    (insert "历史当前输入")
    (narrow-to-region 3 (point-max))
    (should (equal (wime--context) "当前输入"))
    (widen)
    (add-text-properties 1 3 '(field history rear-nonsticky t))
    (let ((inhibit-field-text-motion t))
      (should (equal (wime--context) "当前输入"))
      (goto-char 3)
      (should (equal (wime--context) "")))))

(ert-deftest wime-context-stops-at-read-only-text-and-overlays ()
  (dolist (overlay '(nil t))
    (with-temp-buffer
      (insert "旧内容|当前输入")
      (if overlay
          (overlay-put (make-overlay 4 5) 'read-only t)
        (put-text-property 4 5 'read-only t))
      (should (equal (wime--context) "当前输入"))
      (let ((buffer-read-only t) (inhibit-read-only t))
        (should (equal (wime--context) "")))))
  (with-temp-buffer
    (insert (propertize "提示>" 'read-only t 'rear-nonsticky t))
    (should (equal (wime--context) ""))
    (insert "输入")
    (should (equal (wime--context) "输入"))
    (goto-char 2)
    (should (equal (wime--context) ""))))

(ert-deftest wime-context-does-not-join-across-hidden-text ()
  (dolist (overlay '(nil t))
    (with-temp-buffer
      (insert "之前隐藏当前")
      (if overlay
          (overlay-put (make-overlay 3 5 nil t) 'invisible t)
        (put-text-property 3 5 'invisible t))
      (let ((buffer-invisibility-spec t))
        (should (equal (wime--context) "当前"))
        ;; Insertion before a non-front-sticky hidden span is still visible.
        (save-excursion
          (goto-char 3)
          (should (equal (wime--context) "之前"))))
      (let ((buffer-invisibility-spec nil))
        (should (equal (wime--context) "之前隐藏当前"))))))

(ert-deftest wime-context-excludes-even-an-unprotected-minibuffer-prompt ()
  (with-temp-buffer
    (insert "查找文字：当前输入")
    (cl-letf (((symbol-function 'minibuffer-prompt-end) (lambda () 6)))
      (should (equal (wime--context) "当前输入"))
      (goto-char 6)
      (should (equal (wime--context) "")))))

(ert-deftest wime-context-honors-current-comint-input-without-text-properties ()
  (require 'comint)
  (with-temp-buffer
    (comint-mode)
    (insert "历史输出\n提示> 输入\n第二行")
    (let ((process (make-pipe-process :name "wime-context-comint-test"
                                      :buffer (current-buffer) :noquery t)))
      (unwind-protect
          (progn
            (set-marker (process-mark process) 10)
            (should (equal (wime--context) "输入\n第二行"))
            (goto-char 5)
            (should (equal (wime--context) "")))
        (delete-process process)))))

(ert-deftest wime-context-honors-eshell-input-and-a-buffer-local-provider ()
  (with-temp-buffer
    (insert "历史输出 当前输入")
    (setq-local major-mode 'eshell-mode
                eshell-last-output-end (copy-marker 6))
    (should (equal (wime--context) "当前输入"))
    (setq-local wime-context-start-function (lambda () 8))
    (should (equal (wime--context) "输入"))
    (setq-local wime-context-start-function (lambda () (goto-char 8) (point)))
    (should (equal (wime--context) "输入"))
    (should (= (point) (point-max)))
    ;; A provider can narrow but cannot reopen the shell's history.
    (setq-local wime-context-start-function (lambda () 1))
    (should (equal (wime--context) "当前输入"))
    (setq-local wime-context-start-function (lambda () (error "unavailable")))
    (should (equal (wime--context) ""))
    (setq-local wime-context-start-function (lambda () (make-marker)))
    (should (equal (wime--context) ""))))

(defmacro wime-test--entries (entries &rest body)
  "Run BODY with a small in-memory dictionary ENTRIES."
  (declare (indent 1))
  `(cl-letf (((symbol-function 'wime--entries-for)
              (lambda (code) (cdr (assoc code ,entries))))
             ((symbol-function 'wime--grammar-score)
              (lambda (&rest _) 0.0)))
     ,@body))

(defmacro wime-test--memory (entries &rest body)
  "Run BODY with dictionary ENTRIES and an empty user database file."
  (declare (indent 1))
  `(let* ((wime-user-database (make-temp-file "wime-user" nil ".sqlite3"))
          (wime--user-db nil) (wime--memory (make-hash-table :test #'equal)))
     (unwind-protect
         (cl-letf (((symbol-function 'wime--database-entries)
                    (lambda (code) (cdr (assoc code ,entries))))
                   ((symbol-function 'wime--grammar-score) (lambda (&rest _) 0.0)))
           ,@body)
       (when wime--user-db (sqlite-close (cdr wime--user-db)))
       (delete-file wime-user-database))))

(defconst wime-test--homophones
  '(("dl" ("代" "a" 100) ("待" "b" 50)) ("xr" ("选" "d" 100)) ("xd" ("项" "e" 100))
    ("dlxr" ("代选" "a/d" 30) ("待选" "b/d" 20)) ("xrxd" ("选项" "d/e" 1000))))

(ert-deftest wime-correction-learns-the-covering-word-until-contradicted ()
  (wime-test--memory wime-test--homophones
    (let ((units (wime--parse "dlxrxd")) (tick wime--memory-tick))
      (should (equal (car (wime--sentence-candidates units)) "代选项"))
      (wime--learn "待选项" units "代选项")
      ;; One differing character becomes the dictionary word, never a lone 待.
      (should (equal (gethash "dlxr" wime--memory) '(("待选" . "b/d"))))
      (should-not (gethash "dl" wime--memory))
      (should (> wime--memory-tick tick))
      (should (equal (car (wime--sentence-candidates units)) "待选项"))
      (should (= (wime--memory-coverage "待选项" units) (/ 2.0 3)))
      (should (= (wime--memory-coverage "代选项" units) 0.0))
      ;; Persisted across sessions.
      (sqlite-close (cdr wime--user-db))
      (setq wime--user-db nil)
      (clrhash wime--memory)
      (should (equal (car (wime--sentence-candidates units)) "待选项"))
      ;; Committing another reading for the same keys replaces the preference.
      (wime--learn "代选项" units "待选项")
      (should (equal (gethash "dlxr" wime--memory) '(("代选" . "a/d"))))
      (should (equal (sqlite-select (wime--user-db) "SELECT code, text FROM memory")
                     '(("dlxr" "代选")))))))

(ert-deftest wime-span-pin-correction-is-learned-against-the-unpinned-offer ()
  (wime-test--memory wime-test--homophones
    (with-temp-buffer
      (setq wime--input "dlxrxd")
      (wime--refresh)
      (wime--choose-span 0 1 "待")
      (should (equal (car wime--candidates) "待选项"))
      (should (equal wime--default-reading "代选项"))
      (wime-input-method ?\s)
      (should (equal (gethash "dlxr" wime--memory) '(("待选" . "b/d")))))))

(ert-deftest wime-correction-learns-the-frequent-word-not-one-across-a-boundary ()
  (wime-test--memory '(("ru" ("入" "a" 100)) ("ui" ("是" "b" 100) ("时" "c" 90)) ("hb" ("候" "d" 100))
                       ("ruui" ("入时" "a/c" 50)) ("uihb" ("时候" "c/d" 9000)))
    (let ((units (wime--parse "ruuihb")))
      (wime--learn "入时候" units "入是候")
      (should (equal (gethash "uihb" wime--memory) '(("时候" . "c/d"))))
      (should-not (gethash "ruui" wime--memory)))))

(ert-deftest wime-commit-logs-the-offer-and-learns-only-corrections ()
  (wime-test--memory wime-test--homophones
    (with-temp-buffer
      (insert "按数字键选择")
      (setq wime--input "dlxrxd")
      (wime--refresh)
      (should (equal wime--default-reading "代选项"))
      (wime--commit "代选项")
      (should (zerop (hash-table-count wime--memory)))
      (setq wime--input "dlxrxd")
      (wime--refresh)
      (setq wime--candidate-records
            (append wime--candidate-records
                    (list (wime--make-candidate :text "待选项" :end 3 :score -1)))
            wime--candidates (mapcar #'wime-candidate-text wime--candidate-records))
      (wime--select-digit (+ ?1 (seq-position wime--candidates "待选项")))
      (should (equal (sqlite-select (wime--user-db)
                                    "SELECT input, keys, text, default_reading, context FROM commits")
                     '(("dlxrxd" "dlxrxd" "代选项" "代选项" "按数字键选择")
                       ("dlxrxd" "dlxrxd" "待选项" "代选项" "按数字键选择代选项"))))
      (should (equal (gethash "dlxr" wime--memory) '(("待选" . "b/d")))))))

(ert-deftest wime-search-cache-follows-editable-context-and-boundaries ()
  (wime-test--entries '(("aa" ("甲" "a" 100) ("乙" "b" 1))
                        ("bb" ("丙" "c" 100)))
    (cl-letf (((symbol-function 'wime--grammar-score)
               (lambda (context word _rear)
                 (if (and (equal context "前") (equal word "乙")) 20.0 0.0))))
      (with-temp-buffer
        (let ((units (wime--parse "aabb")))
          (should (equal (car (wime--candidates "aabb")) "甲丙"))
          (insert "前")
          (should (equal (car (wime--candidates "aabb")) "乙丙"))
          (should (equal (wime--sentence-records units)
                         (let ((wime--search-cache nil)) (wime--sentence-records units)))))
        ;; The text and point do not change, only what belongs to this input.
        (setq-local wime-context-start-function #'point)
        (should (equal (car (wime--candidates "aabb")) "甲丙"))))))

(ert-deftest wime-single-syllable-keeps-display-slots-and-auxiliary-constraints ()
  (with-temp-buffer
    (let ((wime-search-width 1) (wime-candidates-max 10))
      (should (= (length (wime--candidates "qd")) 10))
      (should (cl-every (lambda (c) (and (= (wime-candidate-end c) 1)
                                        (wime-candidate-dictionary c)))
                       wime--candidate-records))
      (insert "真刀真")
      (should (equal (car (wime--candidates "qdM")) "枪"))
      (should (equal (car (wime--candidates "qdG")) "强"))
      (should-not (member "枪" (wime--candidates "qdG")))
      (wime--candidates "q")
      (should-not wime--candidate-records))))

(ert-deftest wime-committed-prefix-continues-the-same-word ()
  (unless (and (wime--database) (wime--grammar-ready-p))
    (ert-skip "word database and octagram required"))
  (let ((context (concat "你模型水平咋样，到底是好是坏，推友们自己测一测就一清二楚了。"
                         "你模型水平好性价比又高的话，你都不用太宣传，推友们绝对会自发地介绍最好用最划算的模型。\n"
                         "你找一堆水军，强迫一堆大V去吹分数，跟 Opus 5.5 碰瓷，这叫自掘坟墓。"
                         "人家 Opus5.5 那水平那口碑是")))
    (dolist (split '(0 2 3))
      (with-temp-buffer
        (insert context)
        (let* ((word "真刀真枪") (code "vfdkvfqd")
               (prefix (substring word 0 split)))
          (when (> split 0)
            (mapc #'wime-input-method (string-to-list (substring code 0 (* 2 split))))
            (should (member prefix wime--candidates))
            (wime--commit prefix))
          (mapc #'wime-input-method (string-to-list (substring code (* 2 split))))
          (should (equal (car wime--candidates) (substring word split)))
          (wime-input-method ?\s)
          (should (equal (buffer-string) (concat context word))))))))

(ert-deftest wime-single-character-context-is-local-to-the-insertion-point ()
  (unless (and (wime--database) (wime--grammar-ready-p))
    (ert-skip "word database and octagram required"))
  (with-temp-buffer
    (insert "真刀真这面")
    (should (equal (car (wime--candidates "qd")) "墙"))
    (backward-char 2)
    (should (equal (car (wime--candidates "qd")) "枪"))
    (narrow-to-region (point) (point-max))
    (should (equal (car (wime--candidates "qd")) "强"))))

(ert-deftest wime-prefix-search-recovers-pruned-reading-without-changing-cache ()
  (wime-test--entries '(("aa" ("甲" "a" 100) ("乙" "b" 1))
                        ("bb" ("丙" "c" 100)))
    (let* ((wime-search-width 1) (wime--search-cache nil)
           (units (wime--parse "aabb"))
           (original (wime--sentence-records units)) (cache wime--search-cache))
      (should (equal (wime-candidate-text (car original)) "甲丙"))
      (should (equal (wime-candidate-text (car (wime--sentence-records units "乙"))) "乙丙"))
      (should (eq cache wime--search-cache))
      (should-not (seq-some (lambda (c) (= (wime-candidate-end c) 2))
                            (wime--sentence-records (wime--parse "aaAbb") "乙"))))))

(ert-deftest wime-prefix-search-handles-word-interior-and-impossible-prefix ()
  (wime-test--entries '(("aabb" ("甲乙" "a/b" 100) ("丙丁" "c/d" 1)))
    (let ((units (wime--parse "aabb")) (wime-search-width 1))
      (should (equal (wime-candidate-text (car (wime--sentence-records units "丙"))) "丙丁"))
      (should-not (wime--sentence-records units "甲乙丙"))
      (should-not (wime--sentence-records units "甲丁")))))

;; Character positions disagree where readings compete; marks follow the
;; posterior and alternatives come from every scored reading.
(ert-deftest wime-uncertain-spans-follow-the-reranker-posterior ()
  (let ((wime-confidence-temperature 3.0) (wime-confidence-threshold 0.9)
        (wime-span-alternatives 3))
    (should (equal (wime--uncertain-spans
                    "甲乙丙" '(("甲乙丙" . -0.1) ("甲丁丙" . -0.3) ("甲戊丙" . -2.0) ("短" . 0.0)))
                   '((1 2 ("丁" "戊")))))
    (should-not (wime--uncertain-spans "甲乙丙" '(("甲乙丙" . -0.1) ("甲丁丙" . -9.0))))
    (should-not (wime--uncertain-spans "甲乙丙" nil))))

(ert-deftest wime-span-choice-pins-characters-and-decodes-the-rest ()
  (wime-test--entries '(("aa" ("甲" "a" 100) ("乙" "b" 90))
                        ("bb" ("丙" "c" 100) ("丁" "d" 90))
                        ("aabb" ("乙丁" "bd" 50))
                        ("cc" ("戊" "e" 100)))
    (with-temp-buffer
      (let ((wime-candidate-style 'spans) (wime-search-width 8)
            (wime-candidates-updated-hook nil))
        (setq wime--input "aabb")
        (wime--refresh)
        (should (equal (car wime--candidates) "甲丙"))
        ;; Without reranker scores there is nothing to mark or number.
        (should (string-suffix-p " 甲丙" (wime--format)))
        (should-not wime--span-choices)
        (setq wime--candidate-scores '(("甲丙" . -0.1) ("乙丁" . -0.15) ("甲丁" . -2.0)))
        (let ((text (wime--format)))
          (should (equal (substring-no-properties text) "aabb 甲丙1乙丁2甲丁"))
          (should (eq (get-text-property 5 'face text) 'wime-uncertain-face))
          ;; Each alternative is boxed as one face run; its label is not.
          (should (equal (mapcar (lambda (i) (get-text-property i 'face text)) '(7 8 9 10 11 12))
                         '(wime-label-face wime-alternatives-face wime-alternatives-face
                           wime-label-face wime-alternatives-face wime-alternatives-face))))
        (should (equal wime--span-choices '((0 2 "乙丁") (0 2 "甲丁"))))
        ;; Digit 1 fixes the span; the search keeps it and scores are stale.
        (wime--select-digit ?1)
        (should (equal wime--input "aabb"))
        (should (equal (mapcar #'car wime--pins) '(0 1)))
        (should (equal (car wime--candidates) "乙丁"))
        (should (seq-every-p (lambda (c) (string-prefix-p (substring (wime-candidate-text c) 0 1) "乙"))
                             wime--candidate-records))
        (should-not wime--candidate-scores)
        ;; Typing on keeps the pins; changing a pinned syllable's keys drops it.
        (setq wime--input "aabbcc")
        (wime--refresh)
        (should (equal (car wime--candidates) "乙丁戊"))
        (should (eq (get-text-property 7 'face (wime--format)) 'wime-pinned-face))
        (setq wime--input "aa")
        (wime--refresh)
        (should (equal (mapcar #'car wime--pins) '(0)))
        (wime--reset)
        (should-not wime--pins)))))

(ert-deftest wime-span-view-toggles-to-the-sentence-list ()
  (wime-test--entries '(("aa" ("甲" "a" 100) ("乙" "b" 90))
                        ("bb" ("丙" "c" 100) ("丁" "d" 90)))
    (with-temp-buffer
      (let ((wime-candidate-style 'spans) (wime-candidates-updated-hook nil))
        (setq wime--input "aabb")
        (wime--refresh)
        (setq wime--candidate-scores '(("甲丙" . -0.1) ("乙丙" . -0.12)))
        (wime-toggle-candidate-list)
        (should (string-match-p "1\\.甲丙 2\\." (substring-no-properties (wime--format))))
        (wime--select-digit ?2)
        (should-not wime--input)))))

(ert-deftest wime-beam-keeps-alternative-full-readings ()
  (wime-test--entries '(("aa" ("甲" "a" 100) ("乙" "b" 90))
                        ("bb" ("丙" "c" 100) ("丁" "d" 90)))
    (let* ((wime-search-width 4)
           (records (wime--sentence-records (wime--parse "aabb")))
           (full (seq-filter (lambda (c) (= (wime-candidate-end c) 2)) records)))
      (should (= (length full) 4))
      (should (member "乙丁" (mapcar #'wime-candidate-text full)))
      (should (seq-some (lambda (c) (= (wime-candidate-end c) 1)) records))
      (should-not (seq-some #'wime-candidate-dictionary full)))))

(ert-deftest wime-beam-respects-auxiliary-codes-and-budget ()
  (wime-test--entries '(("aa" ("甲" "a" 100) ("乙" "b" 90))
                        ("bb" ("丙" "c" 100) ("丁" "d" 90)))
    (let ((wime-search-width 1))
      (should (equal (car (wime--sentence-candidates (wime--parse "aaBbbD")))
                     "乙丁"))
      (should (= 1 (length (seq-filter
                            (lambda (c) (= (wime-candidate-end c) 2))
                            (wime--sentence-records (wime--parse "aabb")))))))))

(ert-deftest wime-beam-retains-prefix-until-later-context-disambiguates ()
  (wime-test--entries '(("aa" ("甲" "a" 100) ("乙" "b" 90))
                        ("bb" ("丙" "c" 100)))
    (cl-letf (((symbol-function 'wime--grammar-score)
               (lambda (context _word _rear) (if (equal context "乙") 10.0 0.0))))
      (let ((wime-search-width 2))
        (should (equal (car (wime--sentence-candidates (wime--parse "aabb")))
                       "乙丙"))))))

(ert-deftest wime-beam-deduplicates-and-retains-dictionary-provenance ()
  (wime-test--entries '(("aa" ("甲" "a" 100)) ("bb" ("乙" "b" 100))
                        ("aabb" ("甲乙" "a/b" 1)))
    ;; Even when composition beats the dictionary path, retain exact-hit evidence.
    (let* ((wime-word-penalty 0)
           (records (wime--sentence-records (wime--parse "aabb")))
           (full (seq-filter (lambda (c) (= (wime-candidate-end c) 2)) records)))
      (should (= (length full) 1))
      (should (wime-candidate-dictionary (car full))))))

(ert-deftest wime-beam-does-not-label-partial-reading-as-full ()
  (wime-test--entries '(("aa" ("甲" "a" 100)))
    (let ((records (wime--sentence-records (wime--parse "aabb"))))
      (should records)
      (should (cl-every (lambda (c) (= (wime-candidate-end c) 1)) records)))))

(ert-deftest wime-incremental-search-matches-fresh-on-append-delete-and-aux ()
  (wime-test--entries '(("aa" ("甲" "a" 100) ("乙" "b" 90))
                        ("bb" ("丙" "c" 100) ("丁" "d" 90))
                        ("cc" ("戊" "e" 100)) ("aabb" ("甲丙" "a/c" 50)))
    (let ((wime--search-cache nil) (wime-search-width 4))
      (cl-letf (((symbol-function 'wime--grammar-score)
                 (lambda (_context word rear)
                   (if (and rear (equal word "丁")) 15.0 0.0))))
        (dolist (input '("aabb" "aabbcc" "aabbcc" "aabbDcc" "aabbD" "aaBbbD" "aabb"))
          (let* ((units (wime--parse input))
                 (incremental (wime--sentence-records units))
                 (fresh (let ((wime--search-cache nil)) (wime--sentence-records units))))
            (should (equal incremental fresh))))))))

(ert-deftest wime-search-cache-invalidates-on-scoring-settings ()
  (wime-test--entries '(("aa" ("甲" "a" 100)) ("bb" ("乙" "b" 100))
                        ("aabb" ("甲乙" "a/b" 1)))
    (let ((wime--search-cache nil))
      (wime--sentence-records (wime--parse "aabb"))
      (let* ((wime-word-penalty 0)
             (cached (wime--sentence-records (wime--parse "aabb")))
             (fresh (let ((wime--search-cache nil))
                      (wime--sentence-records (wime--parse "aabb")))))
        (should (equal cached fresh))))))

(declare-function wime-grammar-query "librime-grammar" (context word rear))

;; The database is a Bazel artifact.  Point at the build output when running
;; from the repository root so the batch suite works without a snapshot.
(unless wime-database-file
  (let ((candidate (expand-file-name "bazel-bin/emacs-packages/wime/wanxiang.db")))
    (when (file-readable-p candidate)
      (setq wime-database-file candidate))))

(defmacro wime-test--type (keys &rest body)
  "Type KEYS into a temporary buffer and evaluate BODY."
  `(with-temp-buffer
     (wime--reset)
     (mapc #'wime-input-method (string-to-list ,keys))
     ,@body))

(ert-deftest wime-single-character-lookup ()
  (should (equal (car (wime--candidates-for "hk" nil)) "好"))
  (should (equal (car (wime--candidates-for "ni" nil)) "你")))

(ert-deftest wime-ing-final-uses-semicolon ()
  (should (member "并" (wime--candidates-for "b;" nil)))
  (should (member "应" (wime--candidates-for "y;" nil))))

(ert-deftest wime-auxiliary-filter-is-case-insensitive ()
  (should (member "匿" (wime--candidates-for "ni" "k")))
  (should (equal (wime--candidates-for "ni" "k")
                 (wime--candidates-for "ni" "K"))))

(ert-deftest wime-uppercase-without-code-passes-through ()
  (with-temp-buffer
    (wime--reset)
    (should (equal (wime-input-method ?A) (list ?A)))))

(ert-deftest wime-parse-keeps-typed-order ()
  (should (equal (wime--parse "yrZt") '(("yr" . "Z") ("t" . ""))))
  (should (equal (wime--parse "nihk") '(("ni" . "") ("hk" . ""))))
  (should (equal (wime--parse "niR") '(("ni" . "R"))))
  (should (equal (wime--parse "y;") '(("y;" . "")))))

(ert-deftest wime-lowercase-after-aux-starts-next-syllable ()
  (wime-test--type "niKh"
    (should (equal wime--input "niKh"))
    (should (equal (wime--parse wime--input) '(("ni" . "K") ("h" . ""))))))

(ert-deftest wime-composition-consumes-uppercase ()
  (wime-test--type "niK"
    (should (equal wime--input "niK"))
    (should (equal (car wime--candidates) "匿"))))

(ert-deftest wime-space-commits-the-selected-candidate ()
  (wime-test--type "niK"
    (wime-input-method ?\s)
    (should (equal (buffer-string) "匿"))))

(ert-deftest wime-digit-selects-a-candidate ()
  (wime-test--type "hk"
    (wime-input-method ?2)
    (should (equal (buffer-string) "号"))))

(ert-deftest wime-backspace-removes-aux-then-code ()
  (wime-test--type "niK"
    (wime-delete-backward)
    (should (equal wime--input "ni"))
    (wime-delete-backward)
    (should (equal wime--input "n"))))

(ert-deftest wime-return-commits-raw-input ()
  (wime-test--type "niK"
    (wime-commit-raw)
    (should (equal (buffer-string) "niK"))))

(ert-deftest wime-escape-cancels-the-composition ()
  (wime-test--type "niK"
    (wime-cancel)
    (should-not wime--input)
    (should-not (wime--active-p))))

(ert-deftest wime-read-only-buffer-passes-through ()
  (with-temp-buffer
    (setq buffer-read-only t)
    (wime--reset)
    (should (equal (wime-input-method ?n) (list ?n)))
    (should-not (wime--active-p))))

(ert-deftest wime-word-lookup ()
  (should (equal (car (wime--candidates "nihk")) "你好"))
  (should (equal (car (wime--candidates "womf")) "我们")))

(ert-deftest wime-word-auxiliary-belongs-to-its-syllable ()
  (should (member "你好" (wime--candidates "niRhk")))
  (should-not (member "你好" (wime--candidates "niXhk"))))

(ert-deftest wime-space-commits-a-word ()
  (wime-test--type "nihk"
    (wime-input-method ?\s)
    (should (equal (buffer-string) "你好"))))

(ert-deftest wime-composes-unknown-codes ()
  (let ((result (wime--candidates "yrtk")))
    (should result)
    (should (= (length (car result)) 2))))

(ert-deftest wime-keeps-exact-words-before-composition ()
  (should (equal (car (wime--candidates "nihk")) "你好")))

(ert-deftest wime-sentence-composition ()
  (unless (wime--database)
    (ert-skip "word database not available"))
  (should (equal (car (wime--candidates "womfm;tm")) "我们明天"))
  (should (equal (car (wime--candidates "yrtk")) "袁涛")))

(ert-deftest wime-alternative-syllable-spellings ()
  (unless (wime--database)
    (ert-skip "word database not available"))
  ;; jue is both jt and jv; nüe is nv, nü is ny, gui is gv.
  (should (member "决" (wime--candidates "jv")))
  (should (member "决" (wime--candidates "jt")))
  (should (member "虐" (wime--candidates "nv")))
  (should (member "女" (wime--candidates "ny")))
  (should (member "贵" (wime--candidates "gv"))))

(ert-deftest wime-polyphonic-words-remain-within-the-typed-code ()
  (unless (wime--database) (ert-skip "word database not available"))
  ;; 便宜 has both biàn yí (便宜行事) and pián yi; 编译 only has bian yi.
  (should (member "便宜" (wime--candidates "bmyi")))
  (should (member "便宜" (wime--candidates "pmyi")))
  (should (member "编译" (wime--candidates "bmyi")))
  (should-not (member "编译" (wime--candidates "pmyi"))))

(ert-deftest wime-grammar-module ()
  (unless (wime--grammar-ready-p)
    (ert-skip "grammar module not available"))
  ;; 我们明天 is a stronger collocation than 我们今天.
  (should (< (wime-grammar-query "我们明" "天" nil)
             (wime-grammar-query "我们今" "天" nil))))

(ert-deftest wime-single-key-completion ()
  (unless (wime--database)
    (ert-skip "word database not available"))
  (should (member "你" (wime--candidates "n")))
  (should (member "就" (wime--candidates "j")))
  ;; A lone incomplete syllable completes to single characters only.
  (should (cl-every (lambda (word) (= (length word) 1))
                    (wime--candidates "n")))
  ;; A complete prefix allows longer words.
  (should (member "你好" (wime--candidates "nih"))))

(ert-deftest wime-full-width-punctuation ()
  (with-temp-buffer
    (wime--reset)
    (wime-input-method ?\,)
    (wime-input-method ?.)
    (should (equal (buffer-string) "，。")))
  (with-temp-buffer
    (wime--reset)
    (let ((wime-ascii-punctuation t))
      (should (equal (wime-input-method ?\,) (list 44))))))

(ert-deftest wime-lighter ()
  (with-temp-buffer
    (should (equal (wime-lighter) ""))
    (wime-mode 1)
    (let ((current-input-method "wime"))
      (should (equal (substring-no-properties (wime-lighter)) "wime")))))

(defmacro wime-test--with-database (file &rest body)
  "Evaluate BODY with the database FILE under bazel-bin, or skip."
  (declare (indent 1))
  `(let ((path (expand-file-name (concat "bazel-bin/emacs-packages/wime/" ,file)))
         (saved wime-database-file))
     (unless (file-readable-p path)
       (ert-skip (concat ,file " not built")))
     (unwind-protect
         (progn (wime-select-database path) ,@body)
       (wime-select-database saved))))

(ert-deftest wime-scheme-keys-come-from-the-database ()
  (wime-test--with-database "wanxiang.db"
    ;; 微软双拼 only uses ; for the final ing.
    (should-not (wime--code-char-p ?\; 'initial))
    (should (wime--code-char-p ?\; 'final))
    (should-not (wime--code-char-p ?\, 'final))))

(defun wime-test--liangti (&rest syllables)
  "Return the 量体双拼 code of SYLLABLES, each (INITIAL . FINAL)."
  (let ((table (with-temp-buffer
                 (insert-file-contents
                  (expand-file-name "schemes/liangti.json"
                                    (file-name-directory (locate-library "wime"))))
                 (json-parse-buffer :object-type 'hash-table))))
    (mapconcat (lambda (s)
                 (concat (gethash (car s) (gethash "initials" table))
                         (gethash (cdr s) (gethash "finals" table))))
               syllables "")))

(ert-deftest wime-liangti-scheme-lookup ()
  (wime-test--with-database "wanxiang-liangti.db"
    (should (equal (car (wime--candidates (wime-test--liangti '("n" . "i") '("h" . "ao"))))
                   "你好"))
    (should (equal (car (wime--candidates (wime-test--liangti '("w" . "o") '("m" . "en"))))
                   "我们"))
    (should (member "人" (wime--candidates (wime-test--liangti '("r" . "en")))))))

(ert-deftest wime-liangti-punctuation-depends-on-position ()
  (wime-test--with-database "wanxiang-liangti.db"
    ;; A punctuation key that only carries finals is a code key after an
    ;; initial and full-width punctuation where a syllable would start.
    (let* ((keys (wime--position-keys))
           (punct (seq-find (lambda (c) (and (not (string-search (string c) (car keys)))
                                             (string-search (string c) (cdr keys))))
                            '(?, ?.)))
           (nihao (wime-test--liangti '("n" . "i") '("h" . "ao"))))
      (unless punct (ert-skip "no final-only punctuation key"))
      (wime-test--type (concat (substring nihao 0 1) (string punct))
        (should (equal wime--input (concat (substring nihao 0 1) (string punct)))))
      (wime-test--type (concat nihao (string punct))
        (should (equal (buffer-string) (concat "你好" (wime--punctuation-for punct)))))
      (wime-test--type (string punct)
        (should (equal (buffer-string) (wime--punctuation-for punct)))))))

(provide 'wime-test)
;;; wime-test.el ends here
