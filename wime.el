;;; wime.el --- Wanxiang double-pinyin input method -*- lexical-binding: t; -*-

;; Version: 0.1.0
;; Package-Requires: ((emacs "29.1"))

;;; Commentary:
;; A small, Rime-free input method for the wanxiang 微软双拼 scheme with
;; uppercase Moqi auxiliary codes.
;;
;; The typed string is parsed into syllables: two lowercase keys (``;'' counts
;; as a key, for ing) optionally followed by uppercase auxiliary keys.  The
;; auxiliary code belongs to the syllable it follows, so a lowercase key after
;; it starts the next syllable.  This keeps the typed order intact and never
;; has to guess where an auxiliary code ends.
;;
;; Candidate lookup is plain Elisp over the generated single-character and
;; word tables, so the whole input path is testable in batch.

;;; Code:

(require 'cl-lib)
(require 'seq)
(require 'wime-data)
(require 'wime-words)

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
  "Hash table built from the generated single-character and word tables.")

(defvar-local wime--input nil
  "Raw typed string, preserving the case and order of the keys.")

(defvar-local wime--candidates nil
  "Candidates matching the current input.")

(defvar-local wime--selected 0
  "Index into `wime--candidates' of the selected candidate.")

(defvar-local wime--overlay nil
  "Overlay showing the preedit and candidates.")

(defun wime--ensure-table ()
  "Return the lookup table, building it once."
  (unless wime--table
    (let ((table (make-hash-table :test #'equal
                                  :size (+ (length wime-wanxiang-table)
                                           (length wime-wanxiang-words)))))
      (dolist (source (list wime-wanxiang-table wime-wanxiang-words))
        (dolist (entry source)
          (puthash (car entry) (cdr entry) table)))
      (setq wime--table table)))
  wime--table)

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

(defun wime--parse (input)
  "Split INPUT into a list of (CODE . AUX) syllables.

CODE holds one or two lowercase keys (``;'' is a key); a lowercase key
after a complete syllable starts a new syllable.  AUX collects the
uppercase keys typed after that syllable, so the typed order is kept."
  (let (units code aux)
    (dolist (key (string-to-list (or input "")))
      (cond
       ((wime--code-char-p key)
        (when (= (length code) 2)
          (push (cons code (or aux "")) units)
          (setq code nil aux nil))
        (setq code (concat code (char-to-string key))))
       ((wime--aux-char-p key)
        (setq aux (concat aux (char-to-string key))))))
    (when (or code aux)
      (push (cons (or code "") (or aux "")) units))
    (nreverse units)))

(defun wime--full-code (units)
  "Return the concatenated double-pinyin code of UNITS."
  (mapconcat #'car units ""))

(defun wime--syllable-match-p (entry-aux typed)
  "Return non-nil if TYPED matches one comma-separated group of ENTRY-AUX."
  (or (string-empty-p typed)
      (seq-some (lambda (group) (string-prefix-p (downcase typed) group))
                (split-string entry-aux "," t))))

(defun wime--entry-match-p (entry-aux units)
  "Return non-nil if ENTRY-AUX matches the auxiliary codes in UNITS.

Word entries separate syllables with ``/''; single-character entries are
a single syllable.  A syllable matches when the typed auxiliary code is a
prefix of one of its comma-separated groups."
  (let ((syllables (split-string entry-aux "/")))
    (and (= (length syllables) (length units))
         (cl-every (lambda (unit syllable)
                     (wime--syllable-match-p syllable (cdr unit)))
                   units syllables))))

(defun wime--candidates (input)
  "Return candidates for the raw INPUT string."
  (let* ((units (wime--parse input))
         (code (wime--full-code units))
         (entries (and (not (string-empty-p code))
                       (gethash code (wime--ensure-table)))))
    (when entries
      (let (result)
        (dolist (entry entries)
          (when (wime--entry-match-p (cadr entry) units)
            (push (car entry) result)))
        (seq-take (nreverse result) wime-candidates-max)))))

(defun wime--candidates-for (code aux)
  "Return candidates for a single-syllable CODE filtered by AUX."
  (wime--candidates (concat code (upcase (or aux "")))))

(defun wime--active-p ()
  "Return non-nil when a composition is in progress."
  (and wime--input (not (string-empty-p wime--input))))

(defun wime--clear-overlay ()
  "Delete the preedit overlay, if any."
  (when (overlayp wime--overlay)
    (delete-overlay wime--overlay))
  (setq wime--overlay nil))

(defun wime--format ()
  "Return the preedit string, including candidates when enabled."
  (concat
   (apply #'concat
          (mapcar (lambda (key)
                    (propertize (char-to-string key)
                                'face (if (wime--aux-char-p key)
                                          'wime-aux-face
                                        'wime-preedit-face)))
                  (string-to-list (or wime--input ""))))
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

(defun wime--refresh ()
  "Recompute candidates from the raw input and refresh the display."
  (setq wime--candidates (wime--candidates wime--input)
        wime--selected 0)
  (wime-active-mode (if (wime--active-p) 1 -1))
  (wime--display))

(defun wime--reset ()
  "Discard the current composition and its display."
  (setq wime--input nil
        wime--candidates nil
        wime--selected 0)
  (wime-active-mode -1)
  (wime--clear-overlay))

(defun wime--commit (text)
  "Commit TEXT, or the raw typed input when TEXT is nil."
  (let ((fallback (or wime--input "")))
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
      (setq wime--input (concat wime--input (char-to-string key)))
      (wime--refresh)
      nil)
     ((wime--aux-char-p key)
      (if (wime--active-p)
          (progn
            (setq wime--input (concat wime--input (char-to-string key)))
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
  "Delete the last typed key."
  (interactive)
  (when (and wime--input (not (string-empty-p wime--input)))
    (setq wime--input (substring wime--input 0 -1)))
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
