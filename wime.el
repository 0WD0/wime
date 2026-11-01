;;; wime.el --- Wime double-pinyin input method -*- lexical-binding: t; -*-

;; SPDX-License-Identifier: GPL-3.0-or-later

;; Version: 0.1.0
;; URL: https://github.com/0WD0/wime
;; Package-Requires: ((emacs "29.1"))

;;; Commentary:
;; A Rime-free double-pinyin input method using the Wanxiang word list and
;; uppercase Moqi auxiliary codes.  The scheme comes with the database: 微软双拼 by
;; default, or a table-generated one (see schemes/).
;;
;; The typed string is parsed into syllables: two code keys (letters, plus
;; whichever of ``;,./'' the scheme uses) optionally followed by uppercase
;; auxiliary keys.  The auxiliary code belongs to the syllable it follows, so
;; a lowercase key after it starts the next syllable.  This keeps the typed
;; order intact and never has to guess where an auxiliary code ends.
;;
;; Candidate lookup is plain Elisp over the generated single-character and
;; word tables, so the whole input path is testable in batch.

;;; Code:

(require 'cl-lib)
(require 'seq)
(require 'subr-x)
(require 'sqlite)

(declare-function wime-grammar-open "librime-grammar"
                  (shared-data-dir language))
(declare-function wime-grammar-query "librime-grammar" (context word rear))

(defgroup wime nil
  "Wime double-pinyin input method."
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

(defface wime-uncertain-face
  '((t (:underline (:style wave :color "orange"))))
  "Face for characters whose reading a reranker is unsure of."
  :group 'wime)

(defface wime-pinned-face
  '((t (:underline t)))
  "Face for characters fixed by span selection."
  :group 'wime)

(defface wime-alternatives-face
  '((t (:inherit wime-candidate-face)))
  "Face for the text of a numbered span alternative."
  :group 'wime)

(defface wime-label-face
  '((t (:inherit shadow)))
  "Face for the digit before an inline span alternative."
  :group 'wime)

(defface wime-indicator-face
  '((t (:inherit (mode-line-emphasis font-lock-keyword-face) :weight bold)))
  "Face for the wime indicator while it handles input."
  :group 'wime)

(defface wime-indicator-dim-face
  '((t (:inherit (mode-line shadow))))
  "Face for the wime indicator while input passes through."
  :group 'wime)

(defvar wime-title "wime"
  "The title shown in the mode line.")

(defvar wime-english-title "英"
  "The title shown while `wime-english-mode' inserts keys literally.")

(defcustom wime-candidates-max 10
  "Maximum number of candidates shown at once."
  :type 'natnum
  :group 'wime)

(defcustom wime-data-directory
  (expand-file-name "wime" (or (getenv "XDG_DATA_HOME") "~/.local/share"))
  "Directory holding the word database, models and worker environment."
  :type 'directory
  :group 'wime)

(defcustom wime-database-file (expand-file-name "wanxiang.db" wime-data-directory)
  "SQLite database holding the full wanxiang word list, or nil.

gen/wime_gen.py builds it; wdoom points this at its Bazel artifact.
A readable database and Emacs SQLite support are required for candidates."
  :type '(choice (const nil) file)
  :group 'wime)

(defcustom wime-database-limit 64
  "Maximum entries fetched from the database for one code."
  :type 'natnum
  :group 'wime)

(defcustom wime-word-penalty 20.0
  "Penalty subtracted per word when scoring a sentence path."
  :type 'number
  :group 'wime)

(defcustom wime-max-word-syllables 6
  "Longest dictionary word, in syllables, that the lattice considers."
  :type 'natnum
  :group 'wime)

(defcustom wime-search-width 8
  "Maximum distinct paths retained at each syllable boundary."
  :type '(integer :tag "Paths (at least 1)")
  :group 'wime)

(defcustom wime-context-paragraph-only nil
  "When non-nil, stop context at a blank line or form feed.
By default paragraph separators are preserved within the editable input
area and 64-character limit: an earlier paragraph can help disambiguation.
Modes can set this buffer-locally when paragraphs are independent inputs."
  :type 'boolean :group 'wime)

(defvar-local wime-context-start-function nil
  "Optional function returning the start of this buffer's editable input.
Called without arguments at the insertion point.  Return a position or a
marker in this buffer, or nil for no additional boundary.  Modes with custom
editors can set this in their mode hook.  It can only narrow the context;
field, prompt, visibility, read-only and length limits still apply.
Return point to disable context.  Errors or invalid markers also yield none.")

(defvar-local wime-context-prefix-function nil
  "Optional function returning text that precedes this buffer's input, or nil.
Called without arguments, for example by a chat buffer to supply its latest
messages before the composer.  The neural decoder reads the text right
before the buffer context; the lattice and the commit log never see it.
Errors yield no prefix.")

(defun wime--context-prefix ()
  "Return `wime-context-prefix-function' text, or the empty string."
  (or (and wime-context-prefix-function
           (condition-case nil
               (save-excursion (funcall wime-context-prefix-function))
             (error nil)))
      ""))

(defvar eshell-last-output-end)

(defun wime--input-start ()
  "Return the current input area's lower bound, clamped to accessible text."
  (let ((start (minibuffer-prompt-end))
        (mode-start
         (cond ((derived-mode-p 'comint-mode)
                (when-let* ((process (get-buffer-process (current-buffer))))
                  (process-mark process)))
               ((derived-mode-p 'eshell-mode)
                (and (boundp 'eshell-last-output-end) eshell-last-output-end))))
        (custom-start
         (when wime-context-start-function
           (condition-case nil
               (save-excursion
                 (save-restriction
                   (funcall wime-context-start-function)))
             (error (point))))))
    (dolist (bound (list mode-start custom-start))
      (cond ((null bound))
            ((integerp bound) (setq start (max start bound)))
            ((and (markerp bound) (eq (marker-buffer bound) (current-buffer)))
             (setq start (max start (marker-position bound))))
            (t (setq start (point)))))
    (max (point-min) (min (point) start))))

(defvar wime-edit--start)
(declare-function wime-edit--active-p "wime-edit")
(declare-function wime-edit--type-tone "wime-edit" (tone))
(declare-function wime-edit--type-literal "wime-edit" (key))

(defun wime--editing-p ()
  "Whether an editable run of `wime-edit-mode' holds the composition."
  (and (bound-and-true-p wime-edit-mode) (wime-edit--active-p)))

(defun wime--context (&optional limit)
  "Return committed context before the current composition."
  (save-excursion
    (when (wime--editing-p)
      (goto-char wime-edit--start))
    (wime--context-at-point limit)))

(defun wime--context-at-point (&optional limit)
  "Return the visible, editable suffix before the insertion point.
Honor narrowing, input-area and field boundaries; stop at protected or
hidden characters instead of joining text across them.  Paragraph limits
are optional; punctuation and single newlines are retained.
Keep at most LIMIT characters, 64 by default: the lattice and the syllable
model read no more."
  (save-excursion
    (save-match-data
      (if (or buffer-read-only (get-pos-property (point) 'read-only)
              (invisible-p (get-pos-property (point) 'invisible)))
          ""
        (let* ((end (point))
               (start (max (- end (or limit 64)) (wime--input-start)))
               (start (max start (field-beginning end nil start)))
               (position end))
          ;; At most LIMIT characters, including overlay properties.  A read-only
          ;; prompt can abut an editable insertion point via rear-nonsticky.
          (while (and (> position start)
                      (not (get-char-property (1- position) 'read-only))
                      (not (invisible-p (1- position))))
            (cl-decf position))
          (setq start position)
          (goto-char end)
          (when (and wime-context-paragraph-only
                     (re-search-backward "\n[ \t]*\n\\|\f" start t))
            (setq start (match-end 0)))
          (buffer-substring-no-properties start end))))))

(cl-defstruct (wime-candidate (:constructor wime--make-candidate))
  "A reading and the input it consumes; END counts parsed syllables."
  text score end dictionary)

(defvar-local wime--candidate-records nil
  "Candidates with scores, input coverage and dictionary provenance.")

(defvar-local wime--search-cache nil
  "Last complete search, as (SIGNATURE UNITS PATHS).")

(defvar-local wime--pins nil
  "Characters fixed by span selection, as (POSITION CHAR CODE) lists.
Every search keeps them; a pin is dropped once the keys at POSITION change.")

(defvar-local wime--typed-tones nil
  "Typed tones of the input being searched, as `wime--tones' returns them.")

(defvar wime--tone-cache nil
  "(DATABASE . TABLE) of single-character tone digits by (CODE . CHAR).")

(defun wime--char-tones (char code)
  "Tone digits CHAR has when read as CODE, from the database."
  (let ((db (wime--database)))
    (unless (eq (car wime--tone-cache) db)
      (setq wime--tone-cache (cons db (make-hash-table :test #'equal))))
    (with-memoization (gethash (cons code char) (cdr wime--tone-cache))
      (mapconcat #'car (sqlite-select db "SELECT tones FROM entry WHERE code = ? AND word = ?"
                                      (list code (string char)))
                 ""))))

(defun wime--tones-allow-p (word start units)
  "Whether WORD, read from syllable START of UNITS, has every typed tone it covers.
Character readings decide, so a typed 一 yi1 still reads 一个; the neutral
tone filters nothing."
  (cl-every (lambda (tone)
              (let ((offset (- (car tone) start)))
                (or (< offset 0) (>= offset (length word)) (= (cdr tone) 5)
                    (seq-contains-p (wime--char-tones (aref word offset) (car (nth (car tone) units)))
                                    (+ ?0 (cdr tone))))))
            wime--typed-tones))

(defun wime--pins-allow-p (word start)
  "Whether WORD, read from syllable START, keeps every pin it covers."
  (cl-every (lambda (pin)
              (let ((offset (- (car pin) start)))
                (or (< offset 0) (>= offset (length word))
                    (eq (aref word offset) (nth 1 pin)))))
            wime--pins))

(defvar wime-candidates-updated-hook nil
  "Hook run after refreshing the traditional candidates.")

(defvar wime-reset-hook nil
  "Hook run when a composition is discarded or committed.")

(defcustom wime-decoding 'hybrid
  "How Chinese compositions are converted.
`lattice': the dictionary lattice alone.
`hybrid': the lattice answers at once; after a typing pause the model's
  whole-input readings come first, then the lattice's words.
`neural': the model's readings alone.  New syllables show their keys until
  it answers, and SPC or a digit waits for it.  Characters the model does
  not know cannot be typed.
The last two load `wime-neural' and its local worker; while the worker or
its model files are unavailable, the lattice converts."
  :type '(choice (const lattice) (const hybrid) (const neural)) :group 'wime)

(defvar wime-lattice-inhibit-functions nil
  "Functions of the parsed units; if one returns non-nil, skip the lattice.")

(defvar wime-settle-hook nil
  "Hook run before a key commits or chooses from the candidates.")

(defcustom wime-grammar-module nil
  "Path to the librime-grammar module, or nil to disable the language model."
  :type '(choice (const nil) file)
  :group 'wime)

(defcustom wime-grammar-shared-dir nil
  "Directory holding the octagram ``.gram'' model."
  :type '(choice (const nil) directory)
  :group 'wime)

(defcustom wime-grammar-language "wanxiang-lts-zh-hans"
  "Octagram language name."
  :type 'string
  :group 'wime)

(defcustom wime-ascii-punctuation nil
  "When non-nil, insert ASCII punctuation instead of full-width."
  :type 'boolean
  :group 'wime)

(defcustom wime-punctuation
  '((?\, . "，")
    (?\\ . "、")
    (?. . "。")
    (?\? . "？")
    (?\! . "！")
    (?\: . "：")
    (?\x28 . "（")
    (?\x29 . "）")
    (?\[ . "【")
    (?\] . "】")
    (?\{ . "『")
    (?\} . "』")
    (?< . "《")
    (?> . "》")
    (?^ . "……")
    (?_ . "——"))
  "Full-width replacements for ASCII punctuation keys."
  :type '(alist :key-type character :value-type string)
  :group 'wime)

(defcustom wime-show-candidates t
  "Whether to show the candidate list in the preedit."
  :type 'boolean
  :group 'wime)

(defcustom wime-candidate-style 'list
  "How a complete multi-syllable composition shows its candidates.
`list' numbers whole sentences.  `spans' shows the best reading once,
marks characters a reranker is unsure of, and numbers alternatives for
those spans only; a digit fixes that span and the rest is decoded again.
\\<wime-active-mode-map>\\[wime-toggle-candidate-list] switches one
composition back to the sentence list."
  :type '(choice (const list) (const spans)) :group 'wime)

(defcustom wime-confidence-threshold 0.9
  "Characters with a lower estimated probability are marked uncertain."
  :type 'number :group 'wime)

(defcustom wime-confidence-temperature 3.0
  "Temperature turning per-character reranker scores into probabilities.
3.0 was calibrated on development pools for the earlier 25M conditional
model with the Qwen3.5-2B rescorer, not yet for the syllable transducer."
  :type 'number :group 'wime)

(defcustom wime-span-alternatives 3
  "Maximum inline alternatives shown for one uncertain span."
  :type 'natnum :group 'wime)

(defvar-local wime--input nil
  "Raw typed string, preserving the case and order of the keys.")

(defvar-local wime--candidates nil
  "Candidates matching the current input.")

(defvar-local wime--selected 0
  "Index into `wime--candidates' of the selected candidate.")

(defvar-local wime--model-ranked nil
  "Non-nil once a model result has been applied to the current candidates.")

(defvar-local wime--candidate-scores nil
  "Reranker scores for the current records, as (TEXT . PER-CHARACTER SCORE).")

(defvar-local wime--span-choices nil
  "Displayed span alternatives in label order, as (START END TEXT) lists.")

(defvar-local wime--list-view nil
  "Non-nil shows the sentence list despite `wime-candidate-style'.")

(defvar-local wime--overlay nil
  "Overlay showing the preedit and candidates.")

(defcustom wime-editable-preedit nil
  "Display Chinese compositions as editable buffer text.
Enable `wime-edit-mode' when activating wime.  Readings survive cursor
movement and switching to literal input until explicitly confirmed."
  :type 'boolean :group 'wime)

(autoload 'wime-edit-mode "wime-edit" nil t)
(defvar wime-edit--keymap)
(defvar wime-edit--generation)
(declare-function wime-edit--render "wime-edit")
(declare-function wime-edit--decodable-p "wime-edit")
(declare-function wime-edit--pending-refresh "wime-edit")
(declare-function wime-edit--input-method "wime-edit" (key))
(declare-function wime-edit-delete-backward "wime-edit")

(defconst wime--default-keys "abcdefghijklmnopqrstuvwxyz;"
  "Code keys assumed when the database does not describe its scheme.")

(defvar wime--position-keys nil
  "Cons (INITIAL-KEYS . FINAL-KEYS) read from the open database.")

(defun wime--position-keys ()
  "Return (INITIAL-KEYS . FINAL-KEYS): the keys a syllable starts and ends with."
  (or wime--position-keys
      (when-let* ((db (wime--database))
                  (rows (ignore-errors
                          (sqlite-select db "SELECT key, value FROM meta"))))
        (setq wime--position-keys
              (cons (cadr (assoc "initial_keys" rows))
                    (cadr (assoc "final_keys" rows)))))
      (cons wime--default-keys wime--default-keys)))

(defun wime--code-char-p (key &optional position)
  "Return non-nil if KEY is a double-pinyin code character.

Letters always are.  A punctuation key such as ``;'' or ``,'' is a code
key only where the scheme uses it: POSITION is `initial' or `final', and
nil accepts either.  Elsewhere it stays punctuation."
  (and (integerp key)
       (or (<= ?a key ?z)
           (let ((keys (wime--position-keys)))
             (string-search (char-to-string key)
                            (pcase position
                              ('initial (car keys))
                              ('final (cdr keys))
                              (_ (concat (car keys) (cdr keys)))))))))

(defun wime--aux-char-p (key)
  "Return non-nil if KEY is an uppercase auxiliary code character."
  (and (integerp key) (<= ?A key ?Z)))

(defun wime--digit-p (key)
  "Return non-nil if KEY is a digit."
  (and (integerp key) (<= ?0 key ?9)))

(defun wime--digit-index (key)
  "The candidate index digit KEY selects: 1 is the first, 0 the tenth."
  (if (= key ?0) 9 (- key ?1)))

(defconst wime--tone-marks "ˉˊˇˋ·"
  "Marks shown for typed tones 1 to 4 and the neutral tone 5.")

(defun wime--tone-key-p (key)
  "Return non-nil if KEY is a tone `wime-tone' typed; `wime--parse' skips it."
  (and (integerp key) (<= ?1 key ?5)))

(defun wime--tones (input)
  "Typed tones of INPUT as (SYLLABLE . TONE) pairs, in syllable order."
  (let ((index -1) (code 2) tones)
    (dolist (key (string-to-list (or input "")) (nreverse tones))
      (cond ((wime--code-char-p key)
             (when (= code 2) (setq index (1+ index) code 0))
             (setq code (1+ code)))
            ((and (wime--tone-key-p key) (>= index 0))
             (setq tones (cons (cons index (- key ?0)) (assq-delete-all index tones))))))))

(defun wime--retone (input tone)
  "INPUT with TONE typed for its last syllable in place of an earlier one."
  (let ((start 0) (code 2))
    (dotimes (i (length input))
      (when (wime--code-char-p (aref input i))
        (when (= code 2) (setq start i code 0))
        (setq code (1+ code))))
    (concat (substring input 0 start)
            (seq-remove #'wime--tone-key-p (substring input start))
            (number-to-string tone))))

(defun wime-tone ()
  "Type the tone of the syllable just typed: C-1 to C-4, C-5 for the neutral tone.
Tones 1 to 4 keep only characters with that reading; all reach the model."
  (interactive)
  (let ((tone (- (event-basic-type last-command-event) ?0)))
    (if (wime--editing-p)
        (wime-edit--type-tone tone)
      (when (wime--parse wime--input)
        (setq wime--input (wime--retone wime--input tone))
        (wime--refresh)))))

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

(defun wime--next-position ()
  "Return `final' when the next code key completes a syllable, else `initial'."
  (let ((last (car (last (wime--parse wime--input)))))
    (if (and last (= (length (car last)) 1) (string-empty-p (cdr last)))
        'final
      'initial)))

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

(defvar wime--db nil
  "SQLite connection to `wime-database-file'.")

(defvar wime--db-failed nil
  "Non-nil once opening the database has failed.")

(defun wime--database ()
  "Return the SQLite connection, or nil when unavailable."
  (when (and wime-database-file (file-readable-p wime-database-file))
    (unless (or wime--db wime--db-failed)
      (condition-case err
          (setq wime--db (sqlite-open wime-database-file))
        (error
         (setq wime--db-failed t)
         (message "wime: cannot open %s: %s" wime-database-file err))))
    wime--db))

(defun wime-select-database (file)
  "Switch to the word database FILE, and with it to its double-pinyin scheme."
  (interactive
   (list (read-file-name "wime database: "
                         (and wime-database-file (file-name-directory wime-database-file))
                         wime-database-file t)))
  (when (fboundp 'wime-neural--stop)
    (wime-neural--stop))
  (when wime--db
    (ignore-errors (sqlite-close wime--db)))
  (setq wime-database-file (and file (expand-file-name file))
        wime--db nil
        wime--db-failed nil
        wime--position-keys nil)
  (wime--reset))

(defun wime--database-entries (code)
  "Return the word entries for CODE from the database."
  (when-let* ((db (wime--database)))
    (sqlite-select db (concat "SELECT word, aux, weight FROM entry WHERE code = ? "
                              "ORDER BY weight DESC LIMIT ?")
                   (list code wime-database-limit))))

(defvar wime--grammar-ready nil)

(defvar wime--grammar-failed nil)

(defun wime--grammar-ready-p ()
  "Load the grammar module once and return non-nil when it is usable."
  (when (and wime-grammar-module wime-grammar-shared-dir
             (not wime--grammar-ready) (not wime--grammar-failed))
    (condition-case err
        (progn
          (module-load (expand-file-name wime-grammar-module))
          (setq wime--grammar-ready
                (and (fboundp 'wime-grammar-open)
                     (wime-grammar-open (expand-file-name wime-grammar-shared-dir)
                                        wime-grammar-language))))
      (error
       (setq wime--grammar-failed t)
       (message "wime: grammar module unavailable: %s" err))))
  wime--grammar-ready)

(defun wime--grammar-score (context word rear)
  "Return the octagram collocation score for WORD after CONTEXT."
  (if (wime--grammar-ready-p)
      (wime-grammar-query (or context "") word rear)
    0.0))

(defcustom wime-user-database nil
  "SQLite file of this user's commit log and learned readings, or nil.
Each committed complete reading records its raw input, keys, text, the
reading wime offered first and the context before it.  With `wime-learn',
corrections also become learned readings kept here."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-learn nil
  "Learn readings from corrections in `wime-user-database'.
Where committed text differs from the reading wime offered first, the
differing span, widened to the shortest dictionary word under the same keys,
becomes a learned reading that the lattice and the decoder strongly prefer.
Committing another reading for those keys forgets it.  Off by default: a
learned reading also wins where the context wants another word."
  :type 'boolean :group 'wime)

(defcustom wime-user-weight 1e7
  "Dictionary weight of a learned reading, above any wanxiang entry."
  :type 'number :group 'wime)

(defvar wime--user-db nil
  "Open `wime-user-database', as (FILE . CONNECTION).")

(defvar wime--memory (make-hash-table :test #'equal)
  "Learned readings: code -> list of (TEXT . AUX).")

(defvar wime--memory-tick 0
  "Incremented whenever learned readings change; part of the search cache key.")

(defvar-local wime--default-reading nil
  "First candidate offered for this input, before any pin or selection.")

(defun wime--reload-memory (db)
  "Reload learned readings from DB and invalidate earlier search results."
  (let ((rows (sqlite-select db "SELECT code, text, aux FROM memory")))
    (clrhash wime--memory)
    (pcase-dolist (`(,code ,text ,aux) rows)
      (push (cons text aux) (gethash code wime--memory)))
    (cl-incf wime--memory-tick)))

(defun wime--user-db ()
  "Return the user database connection, loading learned readings once."
  (when wime-user-database
    (unless (equal (car wime--user-db) wime-user-database)
      (make-directory (file-name-directory (expand-file-name wime-user-database)) t)
      (let ((db (sqlite-open wime-user-database)))
        (sqlite-execute db (concat "CREATE TABLE IF NOT EXISTS commits (time REAL, input TEXT, keys TEXT, "
                                   "text TEXT, default_reading TEXT, context TEXT)"))
        (sqlite-execute db (concat "CREATE TABLE IF NOT EXISTS memory (code TEXT, text TEXT, aux TEXT, "
                                   "count INTEGER, updated REAL, PRIMARY KEY (code, text))"))
        (setq wime--user-db (cons wime-user-database db))
        (wime--reload-memory db)))
    (cdr wime--user-db)))

(defun wime--learned-db ()
  "The user database when `wime-learn' applies its learned readings."
  (and wime-learn (wime--user-db)))

(defun wime--memory-entries (code)
  "Learned readings for CODE, as database entries."
  (when (wime--learned-db)
    (mapcar (lambda (m) (list (car m) (cdr m) wime-user-weight)) (gethash code wime--memory))))

(defun wime--memory-spans (units)
  "Learned readings within UNITS' leading complete syllables, as (START . TEXT)."
  (let ((n (cl-loop for unit in units while (= (length (car unit)) 2) count t)) spans)
    (when (and (> n 1) (wime--learned-db) (> (hash-table-count wime--memory) 0))
      (dotimes (s n)
        (cl-loop for e from (+ s 2) to (min n (+ s 16)) do
                 (dolist (m (gethash (wime--span-code units s e) wime--memory))
                   (push (cons s (car m)) spans)))))
    (nreverse spans)))

(defun wime--dictionary-entry (word units)
  "WORD's dictionary entry under the keys of UNITS, or nil."
  (seq-find (lambda (entry) (equal (car entry) word))
            (wime--database-entries (wime--span-code units 0 (length units)))))

(defun wime--learned-aux (word units)
  "Auxiliary codes of WORD read as UNITS, from the dictionary."
  (or (nth 1 (wime--dictionary-entry word units))
      (mapconcat (lambda (i)
                   (or (nth 1 (wime--dictionary-entry (string (aref word i)) (list (nth i units)))) ""))
                 (number-sequence 0 (1- (length word))) "/")))

(defun wime--learned-span (text units start end)
  "Shortest dictionary word of TEXT covering START..END under UNITS' keys.
Among equally short words the most frequent wins, so 输入时候 learns 时候
rather than 入时 across the word boundary.  Without one, at least two
characters, so a single-character correction never becomes a preference
for that syllable everywhere."
  (let ((n (length text)))
    (or (cl-loop for size from (max 2 (- end start)) to (min n wime-max-word-syllables)
                 thereis (cdr (car (sort (cl-loop for s from (max 0 (- end size)) to (min start (- n size))
                                                  for entry = (wime--dictionary-entry
                                                               (substring text s (+ s size))
                                                               (seq-subseq units s (+ s size)))
                                                  when entry collect (cons (nth 2 entry) (cons s (+ s size))))
                                         (lambda (a b) (> (car a) (car b)))))))
        (if (>= (- end start) 2) (cons start end)
          (let ((s (min start (- n 2)))) (cons s (+ s 2)))))))

(defun wime--learn (text units default)
  "Forget readings TEXT contradicts under UNITS, then learn where it left DEFAULT."
  (when-let* ((db (wime--learned-db)))
    (let ((n (length text)) (changed nil))
      (dotimes (s n)
        (cl-loop for e from (+ s 2) to (min n (+ s 16)) do
                 (let ((code (wime--span-code units s e)) (span (substring text s e)))
                   (dolist (m (copy-sequence (gethash code wime--memory)))
                     (unless (equal (car m) span)
                       (sqlite-execute db "DELETE FROM memory WHERE code = ? AND text = ?" (list code (car m)))
                       (setf (gethash code wime--memory) (delete m (gethash code wime--memory))
                             changed t))))))
      (when (and default (> n 1) (>= (length default) n) (not (equal (substring default 0 n) text)))
        (let ((i 0))
          (while (< i n)
            (if (eq (aref text i) (aref default i))
                (cl-incf i)
              (let ((j i))
                (while (and (< j n) (not (eq (aref text j) (aref default j))))
                  (cl-incf j))
                (pcase-let* ((`(,s . ,e) (wime--learned-span text units i j))
                             (word (substring text s e)) (code (wime--span-code units s e))
                             (aux (wime--learned-aux word (seq-subseq units s e))))
                  (sqlite-execute db (concat "INSERT INTO memory VALUES (?, ?, ?, 1, ?) ON CONFLICT (code, text) "
                                             "DO UPDATE SET count = count + 1, updated = excluded.updated")
                                  (list code word aux (float-time)))
                  (unless (assoc word (gethash code wime--memory))
                    (push (cons word aux) (gethash code wime--memory)))
                  (setq changed t))
                (setq i j))))))
      (when changed (cl-incf wime--memory-tick)))))

(defun wime--entries-for (code)
  "Return the learned and dictionary word entries for CODE."
  (and code (not (string-empty-p code))
       (append (wime--memory-entries code) (wime--database-entries code))))

(defun wime--matching-entries (entries units)
  "Return the entries of ENTRIES whose auxiliary codes match UNITS."
  (seq-filter (lambda (entry)
                (wime--entry-match-p (nth 1 entry) units))
              entries))

(defun wime--unit-words (unit)
  "Return the words for one UNIT that match its auxiliary code."
  (mapcar #'car (wime--matching-entries (wime--entries-for (car unit)) (list unit))))

(defun wime--span-code (units start end)
  "Return the concatenated code of UNITS from START to END."
  (wime--full-code (seq-subseq units start end)))

(defun wime--best-paths (paths)
  "Deduplicate PATHS by surface, preserving dictionary evidence, then prune."
  (let ((seen (make-hash-table :test #'equal)) result)
    (dolist (path (cl-stable-sort paths #'> :key #'wime-candidate-score))
      (let* ((text (wime-candidate-text path)) (previous (gethash text seen)))
        (if previous
            (when (wime-candidate-dictionary path)
              (setf (wime-candidate-dictionary previous) t))
          (puthash text path seen)
          (push path result))))
    (seq-take (nreverse result) (max 1 wime-search-width))))

(defun wime--sentence-records (units &optional required-prefix)
  "Search UNITS, optionally requiring the output to start with REQUIRED-PREFIX.
Filter extensions before beam pruning, including constraints ending inside
a dictionary word.  Constrained searches never replace the typing cache.
Include the editable text before point in collocation scores and the cache
signature."
  (let* ((n (length units)) (context (wime--context))
         (paths (make-vector (1+ n) nil))
         (signature (list context wime-database-file wime--db wime-database-limit
                          wime-word-penalty wime-max-word-syllables wime-search-width
                          wime-grammar-module wime-grammar-shared-dir wime-grammar-language
                          wime--grammar-ready wime--grammar-failed wime--pins wime--typed-tones
                          wime--memory-tick
                          (symbol-function 'wime--entries-for)
                          (symbol-function 'wime--grammar-score)))
         (cache (and (not required-prefix)
                     (equal signature (car wime--search-cache)) wime--search-cache))
         (old-units (nth 1 cache)) (old-paths (nth 2 cache))
         (common (cl-loop for old in old-units for new in units
                          while (equal old new) count t))
         ;; Only the last boundary uses octagram's rear score. When growing or
         ;; shrinking, rebuild both the old final boundary and the new one.
         (reuse (if (equal units old-units) n
                  (max 0 (min common (1- (length old-units)) (1- n))))))
    (aset paths 0 (list (wime--make-candidate :text "" :score 0.0 :end 0)))
    (cl-loop for end from 1 to reuse do (aset paths end (aref old-paths end)))
    (cl-loop for end from (1+ reuse) to n do
             (let (extensions)
               (cl-loop for start from (max 0 (- end wime-max-word-syllables))
                        to (1- end) do
                        (when (aref paths start)
                          (dolist (entry (wime--matching-entries
                                          (wime--entries-for
                                           (wime--span-code units start end))
                                          (seq-subseq units start end)))
                            (dolist (path (aref paths start))
                              (let* ((word (car entry))
                                     (prefix (wime-candidate-text path))
                                     (text (concat prefix word)))
                                (when (and (wime--pins-allow-p word start)
                                           (wime--tones-allow-p word start units)
                                           (or (not required-prefix)
                                               (string-prefix-p required-prefix text)
                                               (and (< end n)
                                                    (string-prefix-p text required-prefix))))
                                  (push (wime--make-candidate
                                       :text text :end end
                                       :dictionary (= start 0)
                                       :score (+ (wime-candidate-score path)
                                                 (log (max 1 (nth 2 entry)))
                                                 (- wime-word-penalty)
                                                 (wime--grammar-score (concat context prefix) word (= end n))))
                                      extensions)))))))
               (aset paths end (wime--best-paths (nreverse extensions)))))
    (unless required-prefix (setq wime--search-cache (list signature units paths)))
    ;; Full readings lead; shorter prefixes remain available after the beam.
    (let ((seen (make-hash-table :test #'equal)) result)
      (cl-loop for end from n downto 1 do
               (dolist (path (aref paths end))
                 (unless (gethash (wime-candidate-text path) seen)
                   (puthash (wime-candidate-text path) t seen)
                   (push path result))))
      (nreverse result))))

(defun wime--prefix-entries (code)
  "Return completion entries whose code starts with CODE."
  (when-let* ((db (wime--database)))
    (sqlite-select db
                   (concat "SELECT word, aux, weight FROM entry "
                           "WHERE code GLOB ? ORDER BY weight DESC LIMIT ?")
                   (list (concat code "*") (* 4 wime-candidates-max)))))

(defun wime--prefix-candidates (units)
  "Return completion candidates for an incomplete UNITS input.

A lone incomplete syllable can only complete to single characters; once
a complete syllable precedes it, longer words are allowed."
  (let ((entries (wime--prefix-entries (wime--full-code units))))
    (when (= (length units) 1)
      (setq entries (seq-filter (lambda (entry) (= (length (car entry)) 1))
                                entries)))
    (seq-uniq (mapcar #'car entries))))

(defun wime--candidates (input)
  "Return candidates for the raw INPUT string.

Complete input, including a single syllable, uses contextual word scores.
An incomplete trailing syllable falls back to a prefix search, the way
Rime completes partial pinyin."
  (setq wime--typed-tones (wime--tones input))
  (let* ((units (wime--parse input))
         (last (car (last units)))
         (complete (cl-every (lambda (unit) (= (length (car unit)) 2)) units))
         (skip (run-hook-with-args-until-success 'wime-lattice-inhibit-functions units)))
    (setq wime--candidate-records
          (when (and units complete (not skip))
            ;; Preserve all display slots for single characters even with a
            ;; narrower sentence beam (e.g. width 1 or the default width 8).
            (let ((wime-search-width (if (= (length units) 1)
                                        (max wime-candidates-max wime-search-width)
                                      wime-search-width)))
              (wime--sentence-records units))))
    (seq-take
     (cond
      ((or (null units) skip) nil)
      (wime--candidate-records (mapcar #'wime-candidate-text wime--candidate-records))
      ((and last (not complete)) (wime--prefix-candidates units))
      (t (seq-filter (lambda (word) (wime--tones-allow-p word 0 units))
                     (wime--unit-words (car units)))))
     wime-candidates-max)))

(defun wime--active-p ()
  "Return non-nil when a composition is in progress."
  (and wime--input (not (string-empty-p wime--input))))

(defun wime--clear-overlay ()
  "Delete the preedit overlay, if any."
  (when (overlayp wime--overlay)
    (delete-overlay wime--overlay))
  (setq wime--overlay nil))

(defun wime--uncertain-spans (top scores)
  "Return (START END ALTERNATIVES) for runs of TOP that SCORES leave uncertain.
SCORES are (TEXT . PER-CHARACTER SCORE).  Readings of TOP's length form a
posterior; a character is uncertain below `wime-confidence-threshold'.
ALTERNATIVES are other substrings at that span, most probable first."
  (let* ((n (length top))
         (aligned (seq-filter (lambda (s) (= (length (car s)) n)) scores))
         (best (and aligned (apply #'max (mapcar #'cdr aligned))))
         (weights (mapcar (lambda (s)
                            (cons (car s) (exp (/ (* n (- (cdr s) best))
                                                  wime-confidence-temperature))))
                          aligned))
         (total (apply #'+ (mapcar #'cdr weights)))
         (sure (lambda (i)
                 (>= (/ (cl-loop for (text . w) in weights
                                 when (eq (aref text i) (aref top i)) sum w)
                        total)
                     wime-confidence-threshold)))
         (i 0) spans)
    (when aligned
      (while (< i n)
        (if (funcall sure i)
            (cl-incf i)
          (let ((j i) (mass (make-hash-table :test #'equal)))
            (while (and (< j n) (not (funcall sure j))) (cl-incf j))
            (pcase-dolist (`(,text . ,w) weights)
              (let ((part (substring text i j)))
                (unless (equal part (substring top i j))
                  (puthash part (+ w (gethash part mass 0)) mass))))
            (push (list i j (seq-take (sort (hash-table-keys mass)
                                            (lambda (a b) (> (gethash a mass) (gethash b mass))))
                                      wime-span-alternatives))
                  spans)
            (setq i j)))))
    (nreverse spans)))

(defun wime--format-spans (top)
  "Return TOP with uncertain spans marked and their alternatives numbered.
Set `wime--span-choices' to the labeled alternatives, in label order."
  (let ((spans (wime--uncertain-spans top wime--candidate-scores))
        (pinned (mapcar #'car wime--pins))
        (label 0) parts choices)
    (dotimes (i (length top))
      (push (propertize (substring top i (1+ i)) 'face
                        (cond ((memq i pinned) 'wime-pinned-face)
                              ((seq-find (lambda (span) (and (<= (car span) i) (< i (nth 1 span))))
                                         spans)
                               'wime-uncertain-face)
                              (t 'wime-candidate-face)))
            parts)
      (pcase-dolist (`(,start ,end ,alternatives) spans)
        (when (= (1+ i) end)
          (dolist (alternative alternatives)
            (when (< label 10)
              (cl-incf label)
              ;; Each alternative is boxed, its label outside, so options stand
              ;; apart from each other and from the reading around them.
              (push (concat (propertize (number-to-string (% label 10)) 'face 'wime-label-face)
                            (propertize alternative 'face 'wime-alternatives-face))
                    parts)
              (push (list start end alternative) choices))))))
    (setq wime--span-choices (nreverse choices))
    (apply #'concat (nreverse parts))))

(defun wime--spans-p ()
  "Whether the current composition renders as one marked reading."
  (and (eq wime-candidate-style 'spans) (not wime--list-view)
       wime--candidate-records
       (> (wime-candidate-end (car wime--candidate-records)) 1)
       (= (wime-candidate-end (car wime--candidate-records))
          (length (wime--parse wime--input)))))

(defun wime--format ()
  "Return the preedit string, including candidates when enabled."
  (setq wime--span-choices nil)
  (concat
   (mapconcat (lambda (key)
                (if (wime--tone-key-p key)
                    (propertize (string (aref wime--tone-marks (- key ?1))) 'face 'wime-aux-face)
                  (propertize (string key) 'face (if (wime--aux-char-p key)
                                                     'wime-aux-face
                                                   'wime-preedit-face))))
              (or wime--input "") "")
   (cond
    ((not (and wime-show-candidates wime--candidates)) nil)
    ((wime--spans-p) (concat " " (wime--format-spans (car wime--candidates))))
    (t (concat " " (wime--format-list wime--candidates wime--selected))))))

(defun wime--format-list (candidates selected)
  "CANDIDATES numbered from 1, the one at index SELECTED highlighted."
  (cl-loop for candidate in candidates for i from 0
           concat (propertize (format "%d.%s " (1+ i) candidate)
                              'face (if (= i selected) 'wime-selected-face 'wime-candidate-face))))

(defun wime--display ()
  "Render the preedit and candidate list at point."
  (wime--clear-overlay)
  (if (wime--editing-p)
      (wime-edit--render)
    (when (wime--active-p)
      (let ((text (wime--format)))
        (unless (string-empty-p text)
          (setq wime--overlay (make-overlay (point) (point)))
          (overlay-put wime--overlay 'after-string text))))))

(defun wime--refresh ()
  "Recompute candidates from the raw input and refresh the display."
  (if (and (wime--editing-p) (not (wime-edit--decodable-p)))
      (wime-edit--pending-refresh)
    (wime--refresh-keys)))

(defun wime--refresh-keys ()
  "Refresh a composition whose flattened input retains syllable boundaries."
  (let ((units (wime--parse wime--input)))
    ;; A pin survives only while the syllable it fixed keeps its keys.
    (setq wime--pins (seq-filter (lambda (pin)
                                   (equal (car (nth (car pin) units)) (nth 2 pin)))
                                 wime--pins)))
  (setq wime--candidate-scores nil wime--model-ranked nil
        wime--candidates (wime--candidates wime--input)
        wime--selected 0)
  ;; A pinned refresh shows the user's own correction, not wime's offer.
  (unless wime--pins
    (setq wime--default-reading (car wime--candidates)))
  (wime-active-mode (if (wime--active-p) 1 -1))
  (wime--display)
  (run-hooks 'wime-candidates-updated-hook))

(defun wime--reset ()
  "Discard the current composition and its display."
  (setq wime--input nil
        wime--candidate-records nil
        wime--search-cache nil
        wime--candidates nil
        wime--candidate-scores nil
        wime--model-ranked nil
        wime--span-choices nil
        wime--pins nil
        wime--typed-tones nil
        wime--list-view nil
        wime--default-reading nil
        wime--selected 0)
  (wime-active-mode -1)
  (wime--clear-overlay)
  (run-hooks 'wime-reset-hook))

(defvar-local wime--committed-state nil
  "Confirmed input segment, endpoint, modification tick and dictionary.
The segment keeps actual keys and auxiliary input; ordinary history is text.")

(defun wime--composition-state (&optional limit)
  "Return original history and any still-valid confirmed input segment.
History keeps at most LIMIT characters, as `wime--context'."
  (or
   (pcase wime--committed-state
     (`(,segment ,end ,tick ,database)
      (when (and (= end (point)) (= tick (buffer-chars-modified-tick))
                 (equal database wime-database-file))
        (let* ((text (alist-get 'text segment))
               ;; Read past the confirmed segment so it cannot evict history.
               (visible (wime--context (+ (or limit 64) (length text)))))
          (when (string-suffix-p text visible)
            `((context . ,(substring visible 0 (- (length visible) (length text))))
              (committed . ,segment)))))))
   `((context . ,(wime--context limit)))))

(defun wime--commit (text)
  "Commit TEXT, or the raw typed input when TEXT is nil."
  (let* ((fallback (or wime--input ""))
         (units (wime--parse fallback))
         (typed (seq-take units (length text)))
         (record (seq-find (lambda (c) (equal text (wime-candidate-text c)))
                           wime--candidate-records))
         (old (alist-get 'committed (wime--composition-state)))
         (code (when (and record (string-match-p "\\`[一-鿿]+\\'" text)
                          (= (length text) (wime-candidate-end record) (length typed))
                          (cl-every (lambda (u) (= (length (car u)) 2)) typed))
                 (wime--full-code typed))))
    (when (and code (wime--user-db))
      (sqlite-execute (wime--user-db) "INSERT INTO commits VALUES (?, ?, ?, ?, ?, ?)"
                      (list (float-time) fallback code text (or wime--default-reading "") (wime--context)))
      (wime--learn text typed wime--default-reading))
    (wime--reset)
    (insert (or text fallback))
    (setq wime--committed-state
          (when code
            ;; Promote the complete old segment to history at this boundary,
            ;; rather than saving an arbitrary suffix of its keys.
            (when (> (+ (length (alist-get 'text old)) (length text)) 64)
              (setq old nil))
            (let ((aux (append (alist-get 'aux old)
                               (cl-loop for unit in typed for i from (length (alist-get 'text old))
                                        unless (string-empty-p (cdr unit))
                                        collect (vector i (downcase (cdr unit)))))))
              (list `((text . ,(concat (alist-get 'text old) text))
                      (input . ,(concat (alist-get 'input old) code))
                      (aux . ,(vconcat aux)))
                    (point) (buffer-chars-modified-tick) wime-database-file))))))

(defun wime--commit-selected ()
  "Commit the selected candidate once pending work has settled."
  (run-hooks 'wime-settle-hook)
  (wime--commit (nth wime--selected wime--candidates)))

(defun wime--select-digit (key)
  "Apply the span alternative or commit the candidate numbered KEY.
In the span view only numbered alternatives respond; others do nothing."
  (run-hooks 'wime-settle-hook)
  (let ((index (wime--digit-index key)))
    (if (wime--spans-p)
        (when-let* ((choice (nth index wime--span-choices)))
          (apply #'wime--choose-span choice))
      (when-let* ((text (nth index wime--candidates)))
        (wime--commit text)))))

(defun wime--choose-span (start end text)
  "Fix TEXT at syllables START..END and decode the rest of the input again."
  (when (wime--editing-p)
    (cl-incf wime-edit--generation))
  (let ((units (wime--parse wime--input)))
    (setq wime--pins
          (append (cl-loop for i from start below end
                           collect (list i (aref text (- i start)) (car (nth i units))))
                  (seq-remove (lambda (pin) (and (<= start (car pin)) (< (car pin) end)))
                              wime--pins))))
  (wime--refresh))

(defun wime-toggle-candidate-list ()
  "Switch the current composition between span view and the sentence list."
  (interactive)
  (setq wime--list-view (not wime--list-view))
  (wime--display))

(defun wime--text-read-only-p ()
  "Return non-nil when inserting at point would fail."
  (and (or buffer-read-only (get-pos-property (point) 'read-only))
       (not (or inhibit-read-only
                (get-pos-property (point) 'inhibit-read-only)))))

(defun wime--punctuation-for (key)
  "Return the full-width replacement for KEY, or nil."
  (unless wime-ascii-punctuation
    (cdr (assq key wime-punctuation))))

(defun wime-toggle-punctuation ()
  "Toggle full-width punctuation."
  (interactive)
  (setq wime-ascii-punctuation (not wime-ascii-punctuation))
  (message "wime punctuation: %s"
           (if wime-ascii-punctuation "ASCII" "中文")))

(define-minor-mode wime-english-mode
  "Insert keys literally while wime stays the input method.
With editable preedit, text typed inside a composition stays part of it:
the Chinese on both sides converts as one sentence, and SPC after switching
back confirms everything."
  :lighter nil
  (force-mode-line-update))

(defun wime-toggle-english (&optional arg)
  "Switch between Chinese and literal English without leaving composition.
Outside wime, or with prefix ARG, run `toggle-input-method' instead."
  (interactive "P")
  (if (and (equal current-input-method "wime") (not arg))
      (wime-english-mode 'toggle)
    (toggle-input-method arg t)))

(defun wime--english-input (key)
  "Insert KEY literally.
Inside an editable run KEY joins the run, which still converts as one
sentence.  The floating composition has no place for literal text, so it
commits its selected candidate first."
  (cond
   ((bound-and-true-p wime-edit-mode)
    (unless (wime-edit--type-literal key) (list key)))
   (t
    (when (wime--active-p)
      (wime--commit-selected))
    (list key))))

(defun wime-input-method (key)
  "Process KEY for the wanxiang input method."
  (cond
   ((wime--text-read-only-p) (list key))
   (wime-english-mode (wime--english-input key))
   ((bound-and-true-p wime-edit-mode) (wime-edit--input-method key))
   (t (wime--legacy-input-method key))))

(defun wime--legacy-input-method (key)
  "Process KEY with the floating composition interface."
  (cond
   ((or (wime--code-char-p key (wime--next-position))
        (and (wime--aux-char-p key) (wime--active-p)))
    (setq wime--input (concat wime--input (char-to-string key)))
    (wime--refresh)
    nil)
   ((and (wime--active-p) (wime--digit-p key))
    (wime--select-digit key)
    nil)
   ((and (wime--active-p) (= key ?\s))
    (wime--commit-selected)
    nil)
   ((wime--punctuation-for key)
    (when (wime--active-p)
      (wime--commit-selected))
    (insert (wime--punctuation-for key))
    nil)
   (t
    (when (wime--active-p)
      (wime--reset))
    (list key))))

(defun wime-delete-backward ()
  "Delete the last typed key."
  (interactive)
  (if (wime--editing-p)
      (wime-edit-delete-backward)
  (when (and wime--input (not (string-empty-p wime--input)))
    (setq wime--input (substring wime--input 0 -1)))
  (if (wime--active-p)
      (wime--refresh)
    (wime--reset))))

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
  (unless (or (bound-and-true-p wime-edit-mode)
              (not (symbolp this-command))
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
    (define-key map (kbd "TAB") #'wime-toggle-candidate-list)
    (dotimes (i 5)
      (define-key map (kbd (format "C-%d" (1+ i))) #'wime-tone))
    map)
  "Keymap installed while a wime composition is active.")

(defvar-local wime--active-emulation-map-alist nil
  "Editable composition bindings for this buffer, ahead of modal keymaps.")

(define-minor-mode wime-active-mode
  "Minor mode installed while a wime composition is active."
  :lighter nil
  :init-value nil
  :keymap nil
  (if wime-active-mode
      (progn
        (if (bound-and-true-p wime-edit-mode)
            (progn
              ;; Editable runs survive switching buffers.  A terminal-wide
              ;; override would keep intercepting keys in the destination.
              (setq wime--active-emulation-map-alist
                    `((wime-active-mode . ,wime-edit--keymap)))
              (setq emulation-mode-map-alists
                    (cons 'wime--active-emulation-map-alist
                          (remq 'wime--active-emulation-map-alist
                                emulation-mode-map-alists))))
          (setq wime--active-emulation-map-alist nil)
          (internal-push-keymap wime-active-mode-map 'overriding-terminal-local-map))
        (add-hook 'pre-command-hook #'wime--clear-before-unrelated-command nil t))
    (setq wime--active-emulation-map-alist nil)
    (internal-pop-keymap wime-active-mode-map 'overriding-terminal-local-map)
    (when (boundp 'wime-edit--keymap)
      (internal-pop-keymap wime-edit--keymap 'overriding-terminal-local-map))
    (remove-hook 'pre-command-hook #'wime--clear-before-unrelated-command t)))

(define-minor-mode wime-mode
  "Minor mode enabled while the wime input method is active."
  :lighter nil
  :init-value nil
  :keymap nil)

(defun wime-activate (_name)
  "Activate the wime input method."
  (when wime-editable-preedit (wime-edit-mode 1))
  (unless (eq wime-decoding 'lattice) (require 'wime-neural))
  (setq-local input-method-function #'wime-input-method)
  (setq-local deactivate-current-input-method-function #'wime-deactivate)
  (wime-english-mode -1)
  (wime-mode 1))

(defun wime-deactivate ()
  "Deactivate the wime input method."
  (wime--reset)
  (wime-english-mode -1)
  (wime-mode -1)
  (kill-local-variable 'input-method-function))

(defun wime-setup ()
  "Install wime as the default input method."
  (setq default-input-method "wime"))

(defun wime-lighter ()
  "Return a mode-line indicator for wime, or an empty string.

Bright while wime handles input, dim while the buffer is read-only and
keys pass through."
  (if (and (equal current-input-method "wime") (bound-and-true-p wime-mode))
      (propertize (if wime-english-mode wime-english-title wime-title)
                  'face (if (wime--text-read-only-p)
                            'wime-indicator-dim-face
                          'wime-indicator-face)
                  'help-echo (if wime-english-mode "wime: 英文，C-\\ 切回中文"
                               "wime: 中文双拼"))
    ""))

(register-input-method "wime" "euc-cn" #'wime-activate wime-title
                       "Wime double-pinyin input method.")

(autoload 'wime-memory "wime-memory" "Manage learned Wime readings." t)
(autoload 'wime-history "wime-memory" "Browse Wime commit history." t)
(autoload 'wime-memory-at-point "wime-memory" "Find learned readings near point." t)

(provide 'wime)
;;; wime.el ends here
