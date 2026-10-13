;;; wime.el --- Wanxiang double-pinyin input method -*- lexical-binding: t; -*-

;; Version: 0.1.0
;; Package-Requires: ((emacs "29.1"))

;;; Commentary:
;; A small, Rime-free input method for the wanxiang 微软双拼 scheme with
;; uppercase Moqi auxiliary codes.
;;
;; Lowercase keys build the double-pinyin code; the first uppercase key ends
;; that code and starts the auxiliary filter.  Because uppercase letters can
;; never be part of a pinyin syllable, the split is unambiguous and no
;; backtracking is needed.  Single-character candidates come from the
;; generated table in `wime-data'.
;;
;; The engine deliberately owns no C module and no process.  All lookup is
;; plain Elisp over the generated table, so the whole input path is testable
;; in batch.

;;; Code:

(require 'cl-lib)
(require 'seq)
(require 'wime-data)

(defgroup wime nil
  "Wanxiang double-pinyin input method."
  :group 'leim)

(defface wime-preedit-face
  '((t (:inherit font-lock-string-face)))
  "Face for the typed double-pinyin code."
  :group 'wime)

(defface wime-aux-face
  '((t (:inherit font-lock-keyword-face)))
  "Face for the typed auxiliary code."
  :group 'wime)

(defface wime-candidate-face
  '((t (:inherit default)))
  "Face for an unselected candidate."
  :group 'wime)

(defface wime-selected-face
  '((t (:inherit highlight)))
  "Face for the selected candidate."
  :group 'wime)

(defcustom wime-candidates-max 10
  "Maximum number of candidates shown at once."
  :type 'natnum
  :group 'wime)

(defcustom wime-show-candidates t
  "Whether to show the candidate list in the preedit."
  :type 'boolean
  :group 'wime)

(defvar wime--table nil
  "Hash table built from `wime-wanxiang-table' on first use.")

(defvar-local wime--code nil
  "Double-pinyin code typed so far, lowercase, including `;' for ing.")

(defvar-local wime--aux nil
  "Moqi auxiliary code typed so far, in the case the user typed it.")

(defvar-local wime--candidates nil
  "Characters matching the current code and auxiliary filter.")

(defvar-local wime--selected 0
  "Index into `wime--candidates' of the selected candidate.")

(defvar-local wime--overlay nil
  "Overlay showing the preedit and candidates.")

(defun wime--ensure-table ()
  "Return the lookup table, building it once."
  (unless wime--table
    (let ((table (make-hash-table :test #'equal :size (length wime-wanxiang-table))))
      (dolist (entry wime-wanxiang-table)
        (puthash (car entry) (cdr entry) table))
      (setq wime--table table)))
  wime--table)

(defun wime--aux-match-p (entry-aux typed)
  "Return non-nil if ENTRY-AUX matches the typed auxiliary code TYPED.

An entry's auxiliary field can hold several comma-separated groups; the
typed code matches when it is a prefix of any group.  TYPED may be nil.
Comparison is case-insensitive because the table stores lowercase codes."
  (cond
   ((or (null typed) (string-empty-p typed)) t)
   ((or (null entry-aux) (string-empty-p entry-aux)) nil)
   (t (let ((typed (downcase typed)))
        (seq-some (lambda (group) (string-prefix-p typed group))
                  (split-string entry-aux "," t))))))

(defun wime--candidates-for (code aux)
  "Return candidates for double-pinyin CODE filtered by auxiliary AUX."
  (let ((entries (and code (not (string-empty-p code))
                      (gethash code (wime--ensure-table)))))
    (when entries
      (let (result)
        (dolist (entry entries)
          (when (wime--aux-match-p (cadr entry) aux)
            (push (car entry) result)))
        (seq-take (nreverse result) wime-candidates-max)))))

(defun wime--active-p ()
  "Return non-nil when a composition is in progress."
  (and wime--code (not (string-empty-p wime--code))))

(defun wime--code-char-p (key)
  "Return non-nil if KEY is a double-pinyin code character."
  (and (integerp key)
       (or (and (>= key ?a) (<= key ?z))
           (= key ?\;))))

(defun wime--aux-char-p (key)
  "Return non-nil if KEY is an uppercase auxiliary code character."
  (and (integerp key) (>= key ?A) (<= key ?Z)))

(defun wime--digit-p (key)
  "Return non-nil if KEY is a digit."
  (and (integerp key) (>= key ?0) (<= key ?9)))

(defun wime--clear-overlay ()
  "Delete the preedit overlay, if any."
  (when (overlayp wime--overlay)
    (delete-overlay wime--overlay))
  (setq wime--overlay nil))

(defun wime--format ()
  "Return the preedit string, including candidates when enabled."
  (concat
   (propertize (or wime--code "") 'face 'wime-preedit-face)
   (propertize (or wime--aux "") 'face 'wime-aux-face)
   (when (and wime-show-candidates wime--candidates)
     (concat
      " "
      (cl-loop for candidate in wime--candidates
               for index from 1
               concat (propertize (format "%d.%s " index candidate)
                                  'face (if (= (1- index) wime--selected)
                                            'wime-selected-face
                                          'wime-candidate-face)))))))

(defun wime--display ()
  "Render the preedit and candidate list at point."
  (wime--clear-overlay)
  (when (wime--active-p)
    (let ((text (wime--format)))
      (unless (string-empty-p text)
        (setq wime--overlay (make-overlay (point) (point)))
        (overlay-put wime--overlay 'after-string text)))))

(defun wime--reset ()
  "Discard the current composition and its display."
  (setq wime--code nil
        wime--aux nil
        wime--candidates nil
        wime--selected 0)
  (wime-active-mode -1)
  (wime--clear-overlay))

(defun wime--refresh ()
  "Recompute candidates from the current code and refresh the display."
  (setq wime--candidates (wime--candidates-for wime--code wime--aux)
        wime--selected 0)
  (wime-active-mode (if (wime--active-p) 1 -1))
  (wime--display))

(defun wime--commit (text)
  "Commit TEXT, or the raw typed input when TEXT is nil."
  (let ((fallback (concat (or wime--code "") (or wime--aux ""))))
    (wime--reset)
    (insert (or text fallback))))

(defun wime--select-digit (key)
  "Select the candidate for digit KEY, or commit nothing when out of range."
  (let ((index (if (= key ?0) 9 (- key ?1))))
    (when (< index (length wime--candidates))
      (wime--commit (nth index wime--candidates)))))

(defun wime--text-read-only-p ()
  "Return non-nil when inserting at point would fail."
  (and (or buffer-read-only (get-pos-property (point) 'read-only))
       (not (or inhibit-read-only
                (get-pos-property (point) 'inhibit-read-only)))))

(defun wime-input-method (key)
  "Process KEY for the wanxiang input method."
  (if (wime--text-read-only-p)
      (list key)
    (cond
     ((wime--code-char-p key)
      (setq wime--code (concat wime--code (char-to-string key)))
      (wime--refresh)
      nil)
     ((wime--aux-char-p key)
      (if (wime--active-p)
          (progn
            (setq wime--aux (concat wime--aux (char-to-string key)))
            (wime--refresh)
            nil)
        (list key)))
     ((and (wime--active-p) (wime--digit-p key))
      (wime--select-digit key)
      nil)
     ((and (wime--active-p) (= key ?\s))
      (wime--commit (nth wime--selected wime--candidates))
      nil)
     (t
      (when (wime--active-p)
        (wime--reset))
      (list key)))))

(defun wime-delete-backward ()
  "Delete the last typed code or auxiliary character."
  (interactive)
  (cond
   ((and wime--aux (not (string-empty-p wime--aux)))
    (setq wime--aux (substring wime--aux 0 -1)))
   ((and wime--code (not (string-empty-p wime--code)))
    (setq wime--code (substring wime--code 0 -1))))
  (if (wime--active-p)
      (wime--refresh)
    (wime--reset)))

(defun wime-commit-raw ()
  "Commit the typed input as literal text."
  (interactive)
  (wime--commit nil))

(defun wime-cancel ()
  "Cancel the current composition."
  (interactive)
  (wime--reset))

(defun wime--clear-before-unrelated-command ()
  "Drop the composition before a command that does not belong to wime."
  (unless (or (not (symbolp this-command))
              (string-prefix-p "wime-" (symbol-name this-command))
              (string-match-p "self-insert" (symbol-name this-command)))
    (wime--reset)))

(defvar wime-active-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "DEL") #'wime-delete-backward)
    (define-key map (kbd "<backspace>") #'wime-delete-backward)
    (define-key map (kbd "RET") #'wime-commit-raw)
    (define-key map (kbd "<return>") #'wime-commit-raw)
    (define-key map (kbd "<escape>") #'wime-cancel)
    map)
  "Keymap installed while a wime composition is active.")

(define-minor-mode wime-active-mode
  "Minor mode installed while a wime composition is active."
  :lighter nil
  :init-value nil
  :keymap nil
  (if wime-active-mode
      (progn
        (internal-push-keymap wime-active-mode-map 'overriding-terminal-local-map)
        (add-hook 'pre-command-hook #'wime--clear-before-unrelated-command nil t))
    (internal-pop-keymap wime-active-mode-map 'overriding-terminal-local-map)
    (remove-hook 'pre-command-hook #'wime--clear-before-unrelated-command t)))

(define-minor-mode wime-mode
  "Minor mode enabled while the wime input method is active."
  :lighter nil
  :init-value nil)

(defun wime-activate (_name)
  "Activate the wanxiang input method."
  (wime--ensure-table)
  (setq-local input-method-function #'wime-input-method)
  (setq-local deactivate-current-input-method-function #'wime-deactivate)
  (wime-mode 1))

(defun wime-deactivate ()
  "Deactivate the wanxiang input method."
  (wime--reset)
  (wime-mode -1)
  (kill-local-variable 'input-method-function))

(defun wime-setup ()
  "Install the wanxiang input method as the default input method."
  (wime--ensure-table)
  (setq default-input-method "wime"))

(register-input-method "wime" "euc-cn" #'wime-activate "万象双拼")

(provide 'wime)
;;; wime.el ends here
