;;; wime-test.el --- Wanxiang input method engine -*- lexical-binding: t; -*-

;;; Commentary:
;; The engine is pure Elisp, so these tests exercise the whole input path
;; without a module, a subprocess, or a graphical frame.

;;; Code:

(require 'ert)
(require 'wime)

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
      (should (equal (substring-no-properties (wime-lighter)) "万")))))

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
