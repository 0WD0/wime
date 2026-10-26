;;; wime-japanese.el --- AZIK Japanese input decoded by the CUDA worker -*- lexical-binding: t; -*-

;;; Commentary:
;; Keys convert to kana by the AZIK table (japanese/azik.json, exported from
;; ddskk), with SKK's rules: a key sequence that may still grow waits, and
;; keys left over from a finished rule stay pending.  After a pause the
;; worker decodes the kana with the rescorer and a kana-conditioned model;
;; keys still pending restrict a provisional last kana.  Hiragana and
;; katakana remain candidates throughout.

;;; Code:

(require 'cl-lib)
(require 'json)
(require 'japan-util)
(require 'wime)
(require 'wime-neural)

(defcustom wime-japanese-azik-file
  (expand-file-name "japanese/azik.json" (file-name-directory (or load-file-name buffer-file-name)))
  "Romaji table: a JSON array of [KEYS PENDING KANA] rules."
  :type 'file :group 'wime)

(defcustom wime-japanese-model-file nil
  "Kana-conditioned model; with `wime-japanese-table-file' the worker decodes Japanese."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-japanese-table-file nil
  "Reading table of `wime-neural-rescorer''s tokens, from build-japanese-table.py."
  :type '(choice (const nil) file) :group 'wime)

(defvar wime-japanese--table nil
  "Loaded AZIK table: (FILE RULES PREFIXES); RULES maps keys to (PENDING . KANA).")

(defvar-local wime-japanese--keys "" "Keys typed in this composition.")
(defvar-local wime-japanese--candidates nil "Candidate texts, decoded ones first.")
(defvar-local wime-japanese--selected 0 "Index of the selected candidate.")
(defvar-local wime-japanese--overlay nil "Overlay showing the composition.")

(defun wime-japanese--rules ()
  "Return (RULES . PREFIXES) for `wime-japanese-azik-file'."
  (unless (equal (car wime-japanese--table) wime-japanese-azik-file)
    (let ((rules (make-hash-table :test #'equal)) (prefixes (make-hash-table :test #'equal)))
      (seq-doseq (row (with-temp-buffer
                        (insert-file-contents wime-japanese-azik-file)
                        (json-parse-buffer :array-type 'array)))
        (puthash (aref row 0) (cons (aref row 1) (aref row 2)) rules)
        (dotimes (i (length (aref row 0)))
          (puthash (substring (aref row 0) 0 i) t prefixes)))
      (remhash "" prefixes)
      (setq wime-japanese--table (list wime-japanese-azik-file rules prefixes))))
  (cdr wime-japanese--table))

(defun wime-japanese--convert (keys)
  "Return (KANA . PENDING) for KEYS: a sequence that may still grow waits;
one that cannot first emits what the pending keys already spell."
  (pcase-let ((`(,rules ,prefixes) (wime-japanese--rules)) (kana "") (pending ""))
    (dolist (key (string-to-list keys))
      (let ((done nil))
        (while (not done)
          (let* ((typed (concat pending (string key))) (rule (gethash typed rules)))
            (cond ((gethash typed prefixes) (setq pending typed done t))
                  (rule (setq kana (concat kana (cdr rule)) pending (car rule) done t))
                  ((string-empty-p pending) (setq kana (concat kana (string key)) done t))
                  ((gethash pending rules)
                   (setq kana (concat kana (cdr (gethash pending rules))) pending (car (gethash pending rules))))
                  ;; Keys no rule continues stay visible as typed.
                  (t (setq kana (concat kana pending) pending "")))))))
    (cons kana pending)))

(defun wime-japanese--starts (pending)
  "First kana of every reading PENDING keys can still complete to."
  (let (starts)
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
                  (list (string key)
                        (concat (cdr (wime-japanese--convert wime-japanese--keys)) (string key)))))))

(defun wime-japanese--active-p ()
  (not (string-empty-p wime-japanese--keys)))

(defun wime-japanese--format ()
  "Preedit: kana, pending keys, then numbered candidates."
  (pcase-let ((`(,kana . ,pending) (wime-japanese--convert wime-japanese--keys)))
    (concat (propertize kana 'face 'wime-preedit-face)
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
  "Hiragana and katakana of the typed keys, pending keys appended."
  (pcase-let ((`(,kana . ,pending) (wime-japanese--convert wime-japanese--keys)))
    (delete-dups (list (concat kana pending) (concat (japanese-katakana kana) pending)))))

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
  (pcase-let ((`(,kana . ,pending) (wime-japanese--convert (wime-neural--job-input job))))
    (json-serialize
     `(:id ,(wime-neural--job-id job) :operation "japanese" :context ,(wime-neural--job-context job)
       :kana ,kana :pending ,(vconcat (wime-japanese--starts pending))
       :syllables ,(length kana) :candidates []))))

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
  "Process KEY for AZIK Japanese input."
  (if (wime--text-read-only-p)
      (list key)
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
      (list key)))))

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

(defun wime-japanese-activate (_name)
  "Activate AZIK Japanese input."
  (setq-local input-method-function #'wime-japanese-input-method)
  (setq-local deactivate-current-input-method-function #'wime-japanese-deactivate))

(defun wime-japanese-deactivate ()
  "Deactivate AZIK Japanese input."
  (wime-japanese--reset)
  (kill-local-variable 'input-method-function))

(register-input-method "wime-japanese" "Japanese" #'wime-japanese-activate "日"
                       "AZIK romaji, decoded by the local language models")

(provide 'wime-japanese)
;;; wime-japanese.el ends here
