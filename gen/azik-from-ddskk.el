;;; azik-from-ddskk.el --- Export ddskk's AZIK rules as wime's table -*- lexical-binding: t; -*-

;; Usage: DDSKK=/path/to/ddskk OUT=japanese/azik.json \
;;   emacs -Q --batch -l gen/azik-from-ddskk.el < /dev/null
;;
;; Reads (does not load) skk-vars.el's base romaji rules and skk-azik.el's
;; additions, for an English keyboard, as SKK merges them: AZIK drops the
;; sh/ch/th/dh spellings and its rules win over the base ones.  Each row is
;; [KEYS PENDING KANA]: typing KEYS emits KANA and leaves PENDING typed.
;; Punctuation SKK computes with functions gets fixed full-width forms.

;;; Code:

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
                 (vector (car rule) (or (nth 1 rule) "")
                         (string-replace "う゛" "ゔ" (if (consp out) (cdr out) out)))
                 rules))))
  (maphash (lambda (_ row) (push row rows)) rules)
  (let ((coding-system-for-write 'utf-8-unix))
    (with-temp-file (getenv "OUT")
      (insert (json-serialize (vconcat (sort rows (lambda (a b) (string< (aref a 0) (aref b 0))))))
              "\n")))
  (message "%d rules" (length rows)))

;;; azik-from-ddskk.el ends here
