;;; azik-from-ddskk.el --- Export ddskk's AZIK rules as wime's table -*- lexical-binding: t; -*-

;; Usage: DDSKK=/path/to/ddskk OUT=japanese/azik.eld \
;;   emacs -Q --batch -l gen/azik-from-ddskk.el < /dev/null
;;
;; Reads (does not load) skk-vars.el's base romaji rules and skk-azik.el's
;; additions, for an English keyboard, as SKK merges them: AZIK drops the
;; sh/ch/th/dh spellings and its rules win over the base ones.  Each row is
;; (KEYS KANA [PENDING]): typing KEYS emits KANA and leaves PENDING typed.
;; Punctuation SKK computes with functions gets fixed full-width forms.

;;; Code:

(require 'seq)
(require 'subr-x)

(defun azik-from-ddskk--forms (file coding names)
  "Values of the defvar/defcustom NAMES in ddskk's FILE, read with CODING."
  (let ((coding-system-for-read coding) result)
    (with-temp-buffer
      (insert-file-contents (expand-file-name file (getenv "DDSKK")))
      (goto-char (point-min))
      (condition-case nil
          (while t
            (let ((form (read (current-buffer))))
              (when (and (consp form) (memq (car form) '(defvar defcustom))
                         (memq (cadr form) names))
                (push (cons (cadr form) (eval (nth 2 form) t)) result))))
        (end-of-file nil)))
    result))

(let* ((base (alist-get 'skk-rom-kana-base-rule-list
                        (azik-from-ddskk--forms "skk-vars.el" nil '(skk-rom-kana-base-rule-list))))
       (azik (azik-from-ddskk--forms
              "skk-azik.el" 'iso-2022-jp
              '(skk-azik-unnecessary-base-rule-list skk-azik-additional-rom-kana-rule-list)))
       (dropped (alist-get 'skk-azik-unnecessary-base-rule-list azik))
       (rules (make-hash-table :test #'equal)) rows)
  (dolist (rule (append (seq-remove (lambda (r) (member (car r) dropped)) base)
                        ;; skk-azik-keyboard-type 'en, then fixed punctuation.
                        '(("'" nil "ー") ("x'" nil "'")
                          ("," nil "、") ("." nil "。") ("-" nil "ー") ("!" nil "！"))
                        (alist-get 'skk-azik-additional-rom-kana-rule-list azik)))
    (let ((out (nth 2 rule)))
      (when (or (stringp out) (consp out))
        (puthash (car rule)
                 (append (list (car rule)
                               (string-replace "う゛" "ゔ" (if (consp out) (cdr out) out)))
                         (when (nth 1 rule) (list (nth 1 rule))))
                 rules))))
  (maphash (lambda (_ row) (push row rows)) rules)
  (let ((coding-system-for-write 'utf-8-unix)
        (print-escape-newlines t)
        (print-escape-control-characters t)
        group)
    (with-temp-file (getenv "OUT")
      (insert ";;; AZIK rules, initially exported from ddskk for an English keyboard.\n"
              ";; Each row is (KEYS KANA [PENDING]); omitted PENDING means no trailing keys.\n"
              ";; Keep one rule per line, grouped by first key and sorted within each group.\n"
              ";; This file is the editable rule source; re-exporting replaces local edits.\n\n(\n")
      (dolist (row (sort rows (lambda (a b) (string< (car a) (car b)))))
        (let ((next-group (if (string-match-p "\\`[a-z]" (car row))
                              (substring (car row) 0 1)
                            "Punctuation")))
          (unless (equal group next-group)
            (when group (insert "\n"))
            (insert " ;; " next-group "\n")
            (setq group next-group)))
        (insert " " (string-replace "　" "\\u3000" (prin1-to-string row)) "\n"))
      (insert ")\n")))
  (message "%d rules" (length rows)))

;;; azik-from-ddskk.el ends here
