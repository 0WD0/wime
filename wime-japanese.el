;;; wime-japanese.el --- Japanese input decoded by the CUDA worker -*- lexical-binding: t; -*-

;; SPDX-License-Identifier: GPL-3.0-or-later

;;; Commentary:
;; Keys convert to kana by editable romaji tables in japanese/:
;; a key sequence that may still grow waits, and
;; keys left over from a finished rule stay pending.  After a pause the
;; worker decodes the kana with the rescorer and a kana-conditioned model;
;; keys still pending restrict a provisional last kana.  Hiragana and
;; katakana remain candidates throughout.
;;
;; Keys typed with Shift held mark kana to be written in kanji or katakana
;; (KAku → 書く, I:SYOni → 一緒に).  Once a composition uses Shift, the kana
;; typed without it stay hiragana; without Shift decoding is automatic.  A
;; chord's first kana follows its first key, the rest its last key.

;;; Code:

(require 'cl-lib)
(require 'json)
(require 'lisp-mode)
(require 'japan-util)
(require 'wime)
(require 'wime-neural)

(defconst wime-japanese--directory
  (expand-file-name "japanese/" (file-name-directory (or load-file-name buffer-file-name)))
  "Directory containing the editable Japanese rule tables.")

(defcustom wime-japanese-azik-file
  (expand-file-name "azik.eld" wime-japanese--directory)
  "Romaji table: a data list of (KEYS KANA [PENDING]) rules.
Each rule contains two strings, with an optional third string for trailing
keys to keep pending.  Semicolon comments are allowed; forms are not evaluated."
  :type 'file :group 'wime)

(defcustom wime-japanese-model-file nil
  "Kana-conditioned model; with `wime-japanese-table-file' the worker decodes Japanese."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-japanese-table-file nil
  "Reading table of `wime-neural-rescorer''s tokens, from build-japanese-table.py."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-japanese-zenz-directory nil
  "zenz model from convert-zenz.py; the worker then reranks finished readings with it."
  :type '(choice (const nil) directory) :group 'wime)

(defcustom wime-japanese-shifted-keys
  '((?: . ?\;) (?\" . ?') (?< . ?,) (?> . ?.) (?_ . ?-) (?{ . ?\[) (?} . ?\]))
  "What Shift turns AZIK's punctuation keys into, mapped back to the key.
With Shift held, ; (っ) arrives as :, and still types っ, marked."
  :type '(alist :key-type character :value-type character) :group 'wime)

(defface wime-japanese-shift-face
  '((t (:underline t)))
  "Face for kana typed with Shift held, to be written in kanji or katakana."
  :group 'wime)

(defvar-local wime-japanese--scheme 'azik
  "Current scheme: base, azik, onishi, or onishi-azik.")

(defvar-local wime-japanese--table nil
  "Loaded AZIK table: (FILE RULES PREFIXES); RULES maps keys to (PENDING . KANA).")

(defvar-local wime-japanese--keymap nil
  "Cached (SCHEME . PHYSICAL-TO-LOGICAL-KEYS) for this buffer.")

(defvar-local wime-japanese--keys "" "Keys typed in this composition.")
(defvar-local wime-japanese--candidates nil "Candidate texts, decoded ones first.")
(defvar-local wime-japanese--selected 0 "Index of the selected candidate.")
(defvar-local wime-japanese--overlay nil "Overlay showing the composition.")

(defun wime-japanese--rule-file ()
  "Rule file for this buffer's input scheme."
  (pcase wime-japanese--scheme
    ('base (expand-file-name "base.eld" wime-japanese--directory))
    ('onishi (expand-file-name "romaji.eld" wime-japanese--directory))
    ('onishi-azik (expand-file-name "onishi-azik.eld" wime-japanese--directory))
    (_ wime-japanese-azik-file)))

(defun wime-japanese--rules ()
  "Return (RULES . PREFIXES) for the current scheme's rule file."
  (let ((file (wime-japanese--rule-file)))
    (unless (equal (car wime-japanese--table) file)
      (let ((rules (make-hash-table :test #'equal)) (prefixes (make-hash-table :test #'equal)))
        (let ((rows (with-temp-buffer
                      (insert-file-contents file)
                      (with-syntax-table emacs-lisp-mode-syntax-table
                        (prog1 (read (current-buffer))
                          (forward-comment (point-max))
                          (unless (eobp)
                            (error "Trailing data in AZIK table %s" file)))))))
          (unless (proper-list-p rows)
            (error "Expected a rule list in AZIK table %s" file))
          (dolist (row rows)
            (unless (and (memq (proper-list-p row) '(2 3))
                         (seq-every-p #'stringp row)
                         (not (string-empty-p (car row)))
                         (not (string-empty-p (cadr row))))
              (error "Invalid AZIK rule in %s: %S" file row))
            (let ((keys (car row)))
              (when (gethash keys rules)
                (error "Duplicate AZIK keys in %s: %S" file keys))
              (puthash keys (cons (or (nth 2 row) "") (cadr row)) rules)
              (dotimes (i (length keys))
                (puthash (substring keys 0 i) t prefixes)))))
        (remhash "" prefixes)
        (setq wime-japanese--table (list file rules prefixes))))
    (cdr wime-japanese--table)))

(defun wime-japanese--layout-keys ()
  "Read the selected scheme's generated physical-to-logical key map."
  (unless (eq (car wime-japanese--keymap) wime-japanese--scheme)
    (let* ((layouts (with-temp-buffer
                      (insert-file-contents (expand-file-name "layouts.eld" wime-japanese--directory))
                      (read (current-buffer))))
           (rows (cdr (assq wime-japanese--scheme layouts))) keys)
      (unless rows (error "No key layout for %s" wime-japanese--scheme))
      (dolist (row rows)
        (unless (and (proper-list-p row) (= (length row) 2)
                     (seq-every-p (lambda (s) (and (stringp s) (= (length s) 1))) row)
                     (not (assq (aref (car row) 0) keys)))
          (error "Invalid layout binding: %S" row))
        (push (cons (aref (car row) 0) (aref (cadr row) 0)) keys))
      (setq wime-japanese--keymap (cons wime-japanese--scheme keys))))
  (cdr wime-japanese--keymap))

(defun wime-japanese--translate-key (key)
  "Translate physical QWERTY KEY to the selected scheme's logical key.
Shift follows the physical key even when a letter moves to punctuation.
This runs only at the input-method boundary; stored keys are logical romaji."
  (if (and (characterp key) (< key 128) (memq wime-japanese--scheme '(base azik onishi onishi-azik)))
      (let* ((shifted '((?: . ?\;) (?< . ?,) (?> . ?.) (?? . ?/) (?_ . ?-)))
             (base (or (alist-get key shifted) (downcase key)))
             (mapped (alist-get base (wime-japanese--layout-keys))))
        (if (not mapped) key
          (if (= base key) mapped
            (or (car (rassq mapped shifted)) (upcase mapped)))))
    key))

(defun wime-japanese--base (key)
  "KEY as typed without Shift."
  (cond ((and (>= key ?A) (<= key ?Z)) (downcase key))
        ((alist-get key wime-japanese-shifted-keys))
        (t key)))

(defun wime-japanese--shifted-p (key)
  "Whether KEY was typed with Shift held."
  (or (and (>= key ?A) (<= key ?Z)) (and (assq key wime-japanese-shifted-keys) t)))

(defun wime-japanese--unshifted (keys)
  (concat (mapcar #'wime-japanese--base keys)))

(defun wime-japanese--mark (kana keys)
  "KANA spelled by KEYS, its first kana marked by the first key's Shift, the rest by the last's."
  (concat (propertize (substring kana 0 1) 'wime-japanese-shift (wime-japanese--shifted-p (aref keys 0)))
          (propertize (substring kana 1) 'wime-japanese-shift
                      (wime-japanese--shifted-p (aref keys (1- (length keys)))))))

(defun wime-japanese--convert (keys)
  "Return (KANA . PENDING) for KEYS: a sequence that may still grow waits;
one that cannot first emits what the pending keys already spell.  Kana typed
with Shift carry a non-nil `wime-japanese-shift' property; PENDING keeps the
keys as typed."
  (pcase-let ((`(,rules ,prefixes) (wime-japanese--rules)) (kana "") (pending ""))
    (dolist (key (string-to-list keys))
      (let ((done nil))
        (while (not done)
          (let* ((raw (concat pending (string key)))
                 (typed (wime-japanese--unshifted raw))
                 (rule (gethash typed rules))
                 (flushed (gethash (wime-japanese--unshifted pending) rules)))
            (cond ((gethash typed prefixes) (setq pending raw done t))
                  (rule (setq kana (concat kana (wime-japanese--mark (cdr rule) raw))
                              pending (substring raw (- (length raw) (length (car rule))))
                              done t))
                  ((string-empty-p pending)
                   (setq kana (concat kana (wime-japanese--mark (string key) (string key))) done t))
                  (flushed
                   (setq kana (concat kana (wime-japanese--mark (cdr flushed) pending))
                         pending (substring pending (- (length pending) (length (car flushed))))))
                  ;; Keys no rule continues stay visible as typed.
                  (t (setq kana (concat kana (mapconcat (lambda (c) (wime-japanese--mark (string c) (string c)))
                                                        pending ""))
                           pending "")))))))
    (cons kana pending)))

(defun wime-japanese--explicit-p (keys)
  "Whether KEYS use Shift, so unshifted kana stay hiragana."
  (seq-some #'wime-japanese--shifted-p keys))

(defun wime-japanese--starts (pending)
  "First kana of every reading PENDING keys can still complete to."
  (let ((pending (wime-japanese--unshifted pending)) starts)
    (unless (string-empty-p pending)
      (maphash (lambda (keys rule)
                 (when (and (string-prefix-p pending keys) (not (string-empty-p (cdr rule))))
                   (cl-pushnew (substring (cdr rule) 0 1) starts :test #'equal)))
               (car (wime-japanese--rules))))
    (sort starts #'string<)))

(defun wime-japanese--key-p (key)
  "Whether KEY starts an AZIK rule or continues the pending keys (k then l)."
  (pcase-let ((`(,rules ,prefixes) (wime-japanese--rules)))
    (and (characterp key)
         (cl-some (lambda (typed) (or (gethash typed rules) (gethash typed prefixes)))
                  (list (string (wime-japanese--base key))
                        (wime-japanese--unshifted
                         (concat (cdr (wime-japanese--convert wime-japanese--keys)) (string key))))))))

(defun wime-japanese--active-p ()
  (not (string-empty-p wime-japanese--keys)))

(defun wime-japanese--format ()
  "Preedit: kana, pending keys, then numbered candidates."
  (pcase-let ((`(,kana . ,pending) (wime-japanese--convert wime-japanese--keys)))
    (setq kana (copy-sequence kana))
    (add-face-text-property 0 (length kana) 'wime-preedit-face t kana)
    (dotimes (i (length kana))
      (when (get-text-property i 'wime-japanese-shift kana)
        (add-face-text-property i (1+ i) 'wime-japanese-shift-face nil kana)))
    (concat kana
            (propertize pending 'face 'wime-aux-face)
            (when wime-show-candidates
              (concat " " (cl-loop for candidate in wime-japanese--candidates for i from 1
                                   concat (propertize (format "%d.%s " i candidate)
                                                      'face (if (= (1- i) wime-japanese--selected)
                                                                'wime-selected-face 'wime-candidate-face))))))))

(defun wime-japanese--display ()
  (when (overlayp wime-japanese--overlay) (delete-overlay wime-japanese--overlay))
  (setq wime-japanese--overlay nil)
  (when (wime-japanese--active-p)
    (setq wime-japanese--overlay (make-overlay (point) (point)))
    (overlay-put wime-japanese--overlay 'after-string (wime-japanese--format))))

(defun wime-japanese--fallbacks ()
  "Hiragana and katakana of the typed keys, pending keys appended; with
Shift used, first the shifted kana in katakana and the rest in hiragana."
  (pcase-let ((`(,kana . ,pending) (wime-japanese--convert wime-japanese--keys)))
    ;; A complete prefix (notably final n) can be committed while still
    ;; allowing more keys to extend it during composition.
    (when-let* ((rule (gethash (wime-japanese--unshifted pending)
                              (car (wime-japanese--rules)))))
      (setq kana (concat kana (wime-japanese--mark (cdr rule) pending))
            pending (substring pending (- (length pending) (length (car rule))))))
    (delete-dups
     (append (when (wime-japanese--explicit-p wime-japanese--keys)
               (list (concat (mapconcat (lambda (i)
                                          (let ((c (substring-no-properties kana i (1+ i))))
                                            (if (get-text-property i 'wime-japanese-shift kana)
                                                (japanese-katakana c) c)))
                                        (number-sequence 0 (1- (length kana))) "")
                             pending)))
             (list (concat (substring-no-properties kana) pending)
                   (concat (japanese-katakana (substring-no-properties kana)) pending))))))

(defun wime-japanese--refresh ()
  (setq wime-japanese--candidates (wime-japanese--fallbacks) wime-japanese--selected 0)
  (wime-japanese-active-mode (if (wime-japanese--active-p) 1 -1))
  (wime-japanese--display)
  (wime-japanese--schedule))

(defun wime-japanese--reset ()
  (setq wime-japanese--keys "" wime-japanese--candidates nil wime-japanese--selected 0)
  (wime-neural--reset)
  (wime-japanese-active-mode -1)
  (wime-japanese--display))

(defun wime-japanese--commit (text)
  (wime-japanese--reset)
  (insert text))

;;; Worker

(defun wime-japanese--schedule ()
  "Ask the worker for readings after a pause, as wime does for Chinese."
  (wime-neural--reset)
  (when (and wime-neural-mode (not wime-neural--failed) wime-neural-rescorer
             wime-japanese-model-file wime-japanese-table-file (wime-japanese--active-p)
             (not (string-empty-p (car (wime-japanese--convert wime-japanese--keys)))))
    (let ((job (wime-neural--make-job
                :id (cl-incf wime-neural--serial) :buffer (current-buffer)
                :input wime-japanese--keys :position (point) :tick (buffer-chars-modified-tick)
                :context (wime--context) :records wime-japanese--candidates :method 'japanese
                :syllables (length (car (wime-japanese--convert wime-japanese--keys))))))
      (setq wime-neural--timer (run-with-timer wime-neural-delay nil #'wime-neural--enqueue job)))))

(defun wime-japanese--current-p (job)
  "Whether JOB still describes this buffer's unselected composition."
  (and (equal wime-japanese--keys (wime-neural--job-input job))
       (= wime-japanese--selected 0)
       (eq wime-japanese--candidates (wime-neural--job-records job))))

(defun wime-japanese--request (job)
  (let ((keys (wime-neural--job-input job)))
    (pcase-let ((`(,kana . ,pending) (wime-japanese--convert keys)))
      (json-serialize
       `(:id ,(wime-neural--job-id job) :operation "japanese" :context ,(wime-neural--job-context job)
         :kana ,(substring-no-properties kana) :pending ,(vconcat (wime-japanese--starts pending))
         :explicit ,(if (wime-japanese--explicit-p keys) t :false)
         :shift ,(mapconcat (lambda (i) (if (get-text-property i 'wime-japanese-shift kana) "1" "0"))
                            (number-sequence 0 (1- (length kana))) "")
         :pending_shift ,(if (and (> (length pending) 0) (wime-japanese--shifted-p (aref pending 0))) t :false)
         :syllables ,(length kana) :candidates [])))))

(defun wime-japanese--apply (_job response)
  "Show RESPONSE's readings before the kana fallbacks."
  (let ((decoded (cl-loop for pair across (or (alist-get 'candidates response) [])
                          when (and (vectorp pair) (stringp (aref pair 0)) (not (string-empty-p (aref pair 0))))
                          collect (aref pair 0))))
    (when decoded
      (setq wime-japanese--candidates
            (seq-take (delete-dups (append decoded (wime-japanese--fallbacks))) wime-candidates-max))
      (wime-japanese--display))))

;;; Input method

(defun wime-japanese-input-method (key)
  "Process physical KEY for the selected Japanese input scheme."
  (if (wime--text-read-only-p)
      (list key)
    (let ((key (wime-japanese--translate-key key)))
      (cond
       ((wime-japanese--key-p key)
        (setq wime-japanese--keys (concat wime-japanese--keys (string key)))
        (wime-japanese--refresh)
        nil)
       ((and (wime-japanese--active-p) (wime--digit-p key))
        (let ((index (if (= key ?0) 9 (- key ?1))))
          (when (< index (length wime-japanese--candidates))
            (wime-japanese--commit (nth index wime-japanese--candidates))))
        nil)
       ((and (wime-japanese--active-p) (= key ?\s))
        (wime-japanese--commit (nth wime-japanese--selected wime-japanese--candidates))
        nil)
       (t
        (when (wime-japanese--active-p)
          (wime-japanese--commit (nth wime-japanese--selected wime-japanese--candidates)))
        (list key))))))

(defun wime-japanese-delete-backward ()
  "Delete the last typed key."
  (interactive)
  (setq wime-japanese--keys (substring wime-japanese--keys 0 (max 0 (1- (length wime-japanese--keys)))))
  (if (wime-japanese--active-p) (wime-japanese--refresh) (wime-japanese--reset)))

(defun wime-japanese-commit-kana ()
  "Commit the typed keys as hiragana."
  (interactive)
  (wime-japanese--commit (car (wime-japanese--fallbacks))))

(defun wime-japanese-cancel ()
  "Cancel the composition."
  (interactive)
  (wime-japanese--reset))

(defun wime-japanese--clear-before-unrelated-command ()
  (unless (or (not (symbolp this-command))
              (string-prefix-p "wime-japanese-" (symbol-name this-command))
              (string-match-p "self-insert" (symbol-name this-command)))
    (wime-japanese--reset)))

(defvar wime-japanese-active-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "DEL") #'wime-japanese-delete-backward)
    (define-key map (kbd "<backspace>") #'wime-japanese-delete-backward)
    (define-key map (kbd "RET") #'wime-japanese-commit-kana)
    (define-key map (kbd "<return>") #'wime-japanese-commit-kana)
    (define-key map (kbd "<escape>") #'wime-japanese-cancel)
    map)
  "Keymap installed while a Japanese composition is active.")

(define-minor-mode wime-japanese-active-mode
  "Minor mode installed while a Japanese composition is active."
  :lighter nil :init-value nil :keymap nil
  (if wime-japanese-active-mode
      (progn
        (internal-push-keymap wime-japanese-active-mode-map 'overriding-terminal-local-map)
        (add-hook 'pre-command-hook #'wime-japanese--clear-before-unrelated-command nil t))
    (internal-pop-keymap wime-japanese-active-mode-map 'overriding-terminal-local-map)
    (remove-hook 'pre-command-hook #'wime-japanese--clear-before-unrelated-command t)))

(defun wime-japanese-activate (name)
  "Activate the Japanese input scheme registered as NAME."
  ;; Re-enabling the input method picks up regenerated rules and key positions.
  (setq wime-japanese--table nil wime-japanese--keymap nil)
  (setq wime-japanese--scheme
        (pcase name
          ("wime-japanese-base" 'base)
          ("wime-japanese-onishi" 'onishi)
          ("wime-japanese-onishi-azik" 'onishi-azik)
          (_ 'azik)))
  (setq-local input-method-function #'wime-japanese-input-method)
  (setq-local deactivate-current-input-method-function #'wime-japanese-deactivate))

(defun wime-japanese-deactivate ()
  "Deactivate Japanese input."
  (wime-japanese--reset)
  (kill-local-variable 'wime-japanese--scheme)
  (kill-local-variable 'input-method-function))

(register-input-method "wime-japanese" "Japanese" #'wime-japanese-activate "日"
                       "AZIK romaji, decoded by the local language models")
(register-input-method "wime-japanese-base" "Japanese" #'wime-japanese-activate "日B"
                       "QWERTY with ordinary romaji")
(register-input-method "wime-japanese-onishi" "Japanese" #'wime-japanese-activate "大"
                       "Onishi layout with ordinary romaji")
(register-input-method "wime-japanese-onishi-azik" "Japanese" #'wime-japanese-activate "大A"
                       "Onishi layout with Makiyuki AZIK (article version)")

(provide 'wime-japanese)
;;; wime-japanese.el ends here
