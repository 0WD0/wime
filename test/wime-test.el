;;; wime-test.el --- Wanxiang input method engine -*- lexical-binding: t; -*-

;;; Commentary:
;; The engine is pure Elisp, so these tests exercise the whole input path
;; without a module, a subprocess, or a graphical frame.

;;; Code:

(require 'ert)
(require 'wime)

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

(provide 'wime-test)
;;; wime-test.el ends here
