;;; japanese.el --- Generate wime's Japanese schemes from local definitions -*- lexical-binding: t; -*-

;; emacs -Q --batch -l emacs-packages/wime/gen/japanese.el -f wime-japanese-generate
;; Use -f wime-japanese-check to check the committed outputs without writing.

(require 'cl-lib)
(require 'seq)
(require 'subr-x)
(require 'lisp-mode)

(defconst wime-japanese-gen--directory
  (expand-file-name "../japanese/" (file-name-directory (or load-file-name buffer-file-name))))

(defun wime-japanese-gen--read (file)
  "Read one data form from FILE, without evaluating it."
  (with-temp-buffer
    (insert-file-contents file)
    (with-syntax-table emacs-lisp-mode-syntax-table
      (prog1 (read (current-buffer))
        (forward-comment (point-max))
        (unless (eobp) (error "Trailing data in %s" file))))))

(defun wime-japanese-gen--key (key &optional literal)
  "Check KEY against wime's candidate and Shift controls.
LITERAL permits explicitly preserved historical aliases and punctuation."
  (let ((case-fold-search nil))
    (unless (and (stringp key) (not (string-empty-p key))
                 (seq-every-p (lambda (c) (<= (if literal 32 33) c 126)) key)
                 (or literal (not (string-match-p "[A-Z0-9\"<>_:{}]" key))))
      (error "Invalid key %S: digits select candidates; Shift marks kana" key))))

(defun wime-japanese-gen--add (rules key kana &optional override literal)
  "Add KEY and KANA to RULES, rejecting collisions unless OVERRIDE."
  (wime-japanese-gen--key key literal)
  (unless (and (stringp kana) (not (string-empty-p kana)))
    (error "Invalid kana for %S: %S" key kana))
  (when (and (gethash key rules) (not override))
    (error "Conflicting key %S: %S and %S; use :overrides explicitly"
           key (gethash key rules) kana))
  (puthash key kana rules))

(defun wime-japanese-gen--syllables (source)
  "Validate and return SOURCE's named vowel rows."
  (let ((vowels (plist-get source :vowels)) (rows (plist-get source :syllables)) seen)
    (unless (equal vowels '("a" "i" "u" "e" "o"))
      (error ":vowels must name the a/i/u/e/o columns in order"))
    (dolist (row rows)
      (unless (and (proper-list-p row) (= (length row) 6)
                   (stringp (car row))
                   (seq-every-p (lambda (s) (or (null s) (and (stringp s) (not (string-empty-p s)))))
                                (cdr row)))
        (error "Invalid syllable row: %S" row))
      (when (member (car row) seen) (error "Duplicate syllable row: %s" (car row)))
      (push (car row) seen))
    rows))

(defun wime-japanese-gen--rows (source scheme)
  "Return (STEM KANA-COLUMNS EXTENSION-IDS) for each selected row."
  (let ((syllables (wime-japanese-gen--syllables source)) result)
    (dolist (binding (plist-get scheme :rows))
      (unless (or (stringp binding)
                  (and (proper-list-p binding) (>= (length binding) 2)
                       (stringp (car binding)) (stringp (cadr binding))
                       (zerop (% (length binding) 2))))
        (error "Invalid stem binding: %S" binding))
      (let* ((stem (if (stringp binding) binding (car binding)))
             (name (if (stringp binding) binding (cadr binding)))
             (row (assoc name syllables))
             (options (unless (stringp binding) (cddr binding)))
             (vowels (plist-get source :vowels))
             (enabled-vowels (if (plist-member options :vowels) (plist-get options :vowels) vowels))
             (ids (mapcar #'car (plist-get scheme :extensions)))
             (enabled-ids (if (plist-member options :extensions) (plist-get options :extensions) ids)))
        (unless row (error "Unknown syllable row: %s" name))
        (when (assoc stem result) (error "Duplicate stem binding: %s" stem))
        (let ((tail options) seen)
          (while tail
            (unless (and (memq (car tail) '(:vowels :extensions)) (not (memq (car tail) seen)))
              (error "Unknown or duplicate row option: %S" (car tail)))
            (push (car tail) seen)
            (setq tail (cddr tail))))
        (unless (and (proper-list-p enabled-vowels) (proper-list-p enabled-ids)
                     (seq-every-p (lambda (v) (member v vowels)) enabled-vowels)
                     (seq-every-p (lambda (id) (memq id ids)) enabled-ids))
          (error "Unknown vowel or extension in %S" binding))
        (push (list stem (cl-mapcar (lambda (v kana) (when (member v enabled-vowels) kana))
                                   vowels (cdr row)) enabled-ids) result)))
    (nreverse result)))

(defun wime-japanese-gen--layout (source scheme)
  "Build a physical-to-logical key map, rejecting ambiguous layouts."
  (let ((physical (plist-get source :physical-rows))
        (logical (plist-get scheme :layout-rows)))
    (unless (and (proper-list-p physical) (proper-list-p logical)
                 (= (length physical) (length logical))
                 (cl-every (lambda (a b) (and (stringp a) (stringp b)
                                             (= (length a) (length b)))) physical logical))
      (error "Physical and logical layout rows must have matching widths"))
    (let ((from (string-to-list (apply #'concat physical)))
          (to (string-to-list (apply #'concat logical))))
      (unless (and from (= (length from) (length (delete-dups (copy-sequence from))))
                   (equal (sort (copy-sequence from) #'<) (sort (copy-sequence to) #'<)))
        (error "Layout must be a permutation of distinct physical keys"))
      (mapcar (lambda (c) (wime-japanese-gen--key (string c))) from)
      (cl-mapcar (lambda (a b) (list (string a) (string b))) from to))))

(defun wime-japanese-gen--rules (source scheme)
  "Expand vowel rows, suffix recipes, and explicit exceptions."
  (let ((rows (wime-japanese-gen--rows source scheme))
        (vowels (plist-get source :vowels))
        (extensions (plist-get scheme :extensions))
        (rules (make-hash-table :test #'equal)) ids)
    (dolist (extension extensions)
      (unless (and (proper-list-p extension) (= (length extension) 4)
                   (symbolp (car extension)) (not (memq (car extension) ids))
                   (member (nth 2 extension) vowels) (stringp (nth 3 extension)))
        (error "Invalid or duplicate extension: %S" extension))
      (wime-japanese-gen--key (nth 1 extension))
      (push (car extension) ids))
    (dolist (row rows)
      (cl-mapc (lambda (v kana)
                 (when kana (wime-japanese-gen--add rules (concat (car row) v) kana)))
               vowels (nth 1 row))
      (unless (string-empty-p (car row))
        (dolist (extension extensions)
          (when-let* (((memq (car extension) (nth 2 row)))
                      (kana (nth (cl-position (nth 2 extension) vowels :test #'equal) (nth 1 row))))
            (wime-japanese-gen--add rules (concat (car row) (nth 1 extension))
                             (concat kana (nth 3 extension)))))))
    (dolist (field '(:extension-extras :extension-overrides))
      (dolist (extra (plist-get scheme field))
        (let ((extension (assq (car extra) extensions)))
          (unless (and (proper-list-p extra) (= (length extra) 3)
                       extension (assoc (nth 1 extra) rows))
            (error "Unknown extension or stem in %S" extra))
          (let ((key (concat (nth 1 extra) (nth 1 extension)))
                (override (eq field :extension-overrides)))
            (when (and override (not (gethash key rules)))
              (error "Cannot override absent extension key: %s" key))
            (wime-japanese-gen--add rules key (nth 2 extra) override)))))
    (dolist (key (plist-get scheme :nasal-keys))
      (wime-japanese-gen--add rules key "ん"))
    (wime-japanese-gen--add rules (plist-get source :sokuon-key) "っ")
    (dolist (row (append (plist-get source :punctuation) (plist-get scheme :extra-rules)))
      (unless (and (proper-list-p row) (= (length row) 2)) (error "Invalid rule: %S" row))
      (wime-japanese-gen--add rules (car row) (cadr row)))
    ;; Preserve explicitly listed historical punctuation and shifted aliases.
    ;; These remain subject to the runtime's existing Shift normalization.
    (dolist (row (plist-get scheme :literal-rules))
      (unless (and (proper-list-p row) (= (length row) 2)) (error "Invalid literal rule: %S" row))
      (wime-japanese-gen--add rules (car row) (cadr row) nil t))
    (dolist (key (plist-get scheme :remove))
      (unless (gethash key rules) (error "Cannot remove absent key: %s" key))
      (remhash key rules))
    (let (seen)
      (dolist (row (plist-get scheme :overrides))
        (unless (and (proper-list-p row) (= (length row) 2)
                     (gethash (car row) rules) (not (member (car row) seen)))
          (error "Override must name an existing key once: %S" row))
        (push (car row) seen)
        (wime-japanese-gen--add rules (car row) (cadr row) t
                                (assoc (car row) (plist-get scheme :literal-rules)))))
    (unless (equal (gethash (plist-get source :sokuon-key) rules) "っ")
      (error "The sokuon key must emit っ"))
    (maphash (lambda (key kana)
               (when (and (string-match-p "っ" kana) (not (equal key (plist-get source :sokuon-key))))
                 (error "Sokuon must use only %s, not %s" (plist-get source :sokuon-key) key))) rules)
    rules))

(defun wime-japanese-gen--romaji (source)
  "Canonical kana-to-romaji spellings, independent of scheme bindings."
  (let ((result (make-hash-table :test #'equal)))
    (dolist (row (wime-japanese-gen--syllables source))
      (when (string-match-p "\\`[a-z]*\\'" (car row))
        (cl-mapc
         (lambda (v kana)
           (when kana
             (let ((key (concat (car row) v)) (old (gethash kana result)))
               (when (or (null old) (< (length key) (length old))
                         (and (= (length key) (length old)) (string< key old)))
                 (puthash kana key result)))))
         (plist-get source :vowels) (cdr row))))
    (dolist (row '(("し" . "shi") ("ち" . "chi") ("つ" . "tsu") ("ふ" . "fu")
                   ("じ" . "ji") ("ん" . "n") ("ー" . "-")))
      (puthash (car row) (cdr row) result))
    result))

(defun wime-japanese-gen--comment (kana romaji)
  "Spell KANA using ROMAJI, or return nil for punctuation."
  (let ((rest kana) parts failed)
    (while (and (not failed) (not (string-empty-p rest)))
      (let ((n (length rest)))
        (while (and (> n 0) (not (gethash (substring rest 0 n) romaji))) (cl-decf n))
        (if (= n 0) (setq failed t)
          (push (gethash (substring rest 0 n) romaji) parts)
          (setq rest (substring rest n)))))
    (unless failed
      (concat (apply #'concat (nreverse parts))
              (when (member kana '("ヵ" "ヶ")) " (small katakana)")))))

(defun wime-japanese-gen--table-text (name rules romaji)
  "Render RULES with ordinary romaji comments."
  (with-temp-buffer
    (insert ";;; Generated " name " rules; edit rules.eld, then run gen/japanese.el.\n"
            ";; Each row is (LOGICAL-KEYS KANA); physical key positions are in layouts.eld.\n"
            ";; Comments give ordinary romaji, not this scheme's input keys.\n\n(\n")
    (let ((print-escape-newlines t) (print-escape-control-characters t) group)
      (dolist (key (sort (hash-table-keys rules) #'string<))
        (let* ((kana (gethash key rules))
               (next (if (string-match-p "\\`[a-z]" key) (substring key 0 1) "Punctuation"))
               (row (concat " " (string-replace "　" "\\u3000" (prin1-to-string (list key kana)))))
               (comment (wime-japanese-gen--comment kana romaji)))
          (unless (equal next group)
            (when group (insert "\n"))
            (insert " ;; " next "\n")
            (setq group next))
          (insert row)
          (when comment (insert (make-string (max 2 (- 24 (string-width row))) ?\s) "; " comment))
          (insert "\n"))))
    (insert ")\n")
    (buffer-string)))

(defun wime-japanese-gen--scheme (source name &optional ancestors)
  "Resolve NAME in SOURCE, with child fields replacing inherited fields.
An explicit nil overrides a parent value.  Reject missing parents and cycles."
  (when (memq name ancestors) (error "Scheme inheritance cycle: %S" (cons name ancestors)))
  (let ((scheme (plist-get source name)))
    (unless (and (keywordp name) (proper-list-p scheme) scheme
                 (zerop (% (length scheme) 2))
                 (cl-loop for (key _value) on scheme by #'cddr always (keywordp key)))
      (error "Unknown or invalid scheme: %S" name))
    (if-let* ((parent (plist-get scheme :inherits)))
        (append scheme (wime-japanese-gen--scheme source parent (cons name ancestors)))
      scheme)))

(defun wime-japanese-gen--outputs (source)
  "Build and validate all generated file contents before writing anything."
  (let ((romaji (wime-japanese-gen--romaji source)) outputs layouts)
    (dolist (entry '((:base base "base.eld") (:azik azik "azik.eld") (:onishi onishi "romaji.eld")
                     (:onishi-azik onishi-azik "onishi-azik.eld")))
      (let* ((scheme (wime-japanese-gen--scheme source (car entry)))
             (rules (wime-japanese-gen--rules source scheme)))
        (push (cons (nth 1 entry) (wime-japanese-gen--layout source scheme)) layouts)
        (push (cons (nth 2 entry) (wime-japanese-gen--table-text (symbol-name (nth 1 entry)) rules romaji)) outputs)))
    (push (cons "layouts.eld"
                (with-temp-buffer
                  (insert ";;; Generated layouts; edit :layout-rows in rules.eld.\n"
                          ";; Each pair is (PHYSICAL-QWERTY-KEY LOGICAL-KEY).\n(\n")
                  (dolist (layout (nreverse layouts))
                    (insert " (" (symbol-name (car layout)) "\n")
                    (dolist (pair (cdr layout)) (insert "  " (prin1-to-string pair) "\n"))
                    (insert " )\n"))
                  (insert ")\n")
                  (buffer-string))) outputs)
    (nreverse outputs)))

(defun wime-japanese-generate (&optional source directory)
  "Generate the Japanese schemes from SOURCE into DIRECTORY.
Defaults to japanese/rules.eld and the japanese directory."
  (interactive)
  (let* ((source (or source (expand-file-name "rules.eld" wime-japanese-gen--directory)))
         (directory (or directory wime-japanese-gen--directory))
         (outputs (wime-japanese-gen--outputs (wime-japanese-gen--read source)))
         (coding-system-for-write 'utf-8-unix))
    (make-directory directory t)
    (dolist (output outputs)
      (with-temp-file (expand-file-name (car output) directory) (insert (cdr output))))
    (message "Generated %d Japanese scheme files" (length outputs))))

(defun wime-japanese-check (&optional source directory)
  "Check that committed outputs match SOURCE without modifying files."
  (interactive)
  (let ((outputs (wime-japanese-gen--outputs
                  (wime-japanese-gen--read (or source (expand-file-name "rules.eld" wime-japanese-gen--directory))))))
    (dolist (output outputs)
      (let ((file (expand-file-name (car output) (or directory wime-japanese-gen--directory))))
        (unless (and (file-exists-p file)
                     (equal (cdr output) (with-temp-buffer (insert-file-contents file) (buffer-string))))
          (error "Stale generated file: %s; run wime-japanese-generate" file))))
    (message "Japanese generated files are current")))

(provide 'wime-japanese-generator)
;;; japanese.el ends here
