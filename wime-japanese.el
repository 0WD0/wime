;;; wime-japanese.el --- Japanese input decoded by the GPU worker -*- lexical-binding: t; -*-

;; SPDX-License-Identifier: GPL-3.0-or-later

;;; Commentary:
;; Keys convert to kana by editable romaji tables in japanese/:
;; a key sequence that may still grow waits, and
;; keys left over from a finished rule stay pending.
;;
;; Kana go straight into the buffer.  Every written segment keeps the kana
;; it reads as, so after a pause the worker decodes the kana around point
;; again and later kana can rewrite earlier segments.  Hiragana and
;; katakana remain candidates throughout.  SPC confirms, ESC suspends:
;; suspended text keeps its kana, and typing next to it resumes.
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
(require 'wime-edit)
(require 'wime-neural)

(declare-function evil-normal-state "evil-states" (&optional arg))

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

(defcustom wime-japanese-window 48
  "Kana around point that decoding reconsiders together."
  :type 'natnum :group 'wime)

(cl-defstruct (wime-japanese--segment (:constructor wime-japanese--make-segment (kana text)))
  "Written TEXT of KANA, whose characters carry their Shift marks."
  kana text)

;; Text typed next to a segment is not part of it.
(add-to-list 'text-property-default-nonsticky '(wime-japanese-segment . t))

(defvar-local wime-japanese--pending "" "Keys at point that spell no kana yet.")
(defvar-local wime-japanese--choices nil
  "Readings of the window at point, decoded ones first: (TEXT . SEGMENTS),
each segment a (KANA-COUNT . TEXT) pair.")
(defvar-local wime-japanese--candidates nil "Texts of `wime-japanese--choices'.")
(defvar-local wime-japanese--generation 0 "Counts edits; replies for older text are stale.")
(defvar-local wime-japanese--point nil "Point after the last conversion step.")
(defvar-local wime-japanese--overlay nil "Pending keys and candidates at point.")
(defvar-local wime-japanese--decorations nil "Overlays marking unconfirmed text.")
(defvar-local wime-japanese--internal nil "Non-nil while conversion itself edits the buffer.")

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
        (cond ((not mapped) key)
              ((= base key) mapped)
              (t (or (car (rassq mapped shifted)) (upcase mapped)))))
    key))

(defun wime-japanese--base (key)
  "KEY as typed without Shift."
  (cond ((<= ?A key ?Z) (downcase key))
        ((alist-get key wime-japanese-shifted-keys))
        (t key)))

(defun wime-japanese--shifted-p (key)
  "Whether KEY was typed with Shift held."
  (or (<= ?A key ?Z) (and (assq key wime-japanese-shifted-keys) t)))

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

(defun wime-japanese--key-p (key)
  "Whether KEY joins the composition: a digit, a key starting a rule, or one
continuing the pending keys (k then l)."
  (pcase-let ((`(,rules ,prefixes) (wime-japanese--rules)))
    (and (characterp key)
         (or (wime--digit-p key)
             (cl-some (lambda (typed) (or (gethash typed rules) (gethash typed prefixes)))
                      (list (string (wime-japanese--base key))
                            (wime-japanese--unshifted (concat wime-japanese--pending (string key)))))))))

;;; Segments

(defun wime-japanese--segment-at (position)
  "The segment written at accessible POSITION, when that text is editable."
  (when (and (<= (point-min) position) (< position (point-max))
             (not (get-char-property position 'read-only)) (not (invisible-p position)))
    (get-text-property position 'wime-japanese-segment)))

(defun wime-japanese--run ()
  "The unconfirmed run touching point: (START END SEGMENT) groups in order."
  (let ((field (get-char-property (point) 'field)) (start (point)) (end (point)) groups)
    (while (and (> start (wime--input-start)) (wime-japanese--segment-at (1- start))
                (equal field (get-char-property (1- start) 'field)))
      (cl-decf start))
    (while (and (wime-japanese--segment-at end) (equal field (get-char-property end 'field)))
      (cl-incf end))
    (while (< start end)
      (let ((next (next-single-property-change start 'wime-japanese-segment nil end)))
        (push (list start next (get-text-property start 'wime-japanese-segment)) groups)
        (setq start next)))
    (nreverse groups)))

(defun wime-japanese--window ()
  "Groups of the run that decoding reconsiders: up to `wime-japanese-window'
kana before point, then after it.  A segment holding point counts as before it."
  (let* ((groups (wime-japanese--run))
         (before (reverse (seq-filter (lambda (g) (< (car g) (point))) groups)))
         (after (seq-filter (lambda (g) (>= (car g) (point))) groups))
         (n 0) window)
    (cl-flet ((room-p (g) (and (< n wime-japanese-window)
                               (cl-incf n (length (wime-japanese--segment-kana (nth 2 g)))))))
      (cl-loop for g in before while (room-p g) do (push g window))
      (cl-loop for g in after while (room-p g) do (setq window (append window (list g)))))
    window))

(defun wime-japanese--kana (groups)
  (mapconcat (lambda (g) (wime-japanese--segment-kana (nth 2 g))) groups ""))

(defun wime-japanese--cursor (window)
  "Kana in WINDOW before point."
  (length (wime-japanese--kana (seq-filter (lambda (g) (< (car g) (point))) window))))

(defun wime-japanese--shift (kana)
  "KANA's Shift marks as a string of 1 and 0."
  (mapconcat (lambda (i) (if (get-text-property i 'wime-japanese-shift kana) "1" "0"))
             (number-sequence 0 (1- (length kana))) ""))

(defun wime-japanese--explicit-p (kana)
  "Whether KANA use Shift, so unshifted kana stay hiragana."
  (string-search "1" (wime-japanese--shift kana)))

(defun wime-japanese--edit (from to segments)
  "Replace FROM..TO with SEGMENTS as conversion's own edit."
  (let ((wime-japanese--internal t))
    (save-excursion
      (delete-region from to)
      (goto-char from)
      (dolist (segment segments)
        (insert (propertize (wime-japanese--segment-text segment) 'wime-japanese-segment segment)))))
  (cl-incf wime-japanese--generation))

(defun wime-japanese--provisional (kana)
  "A segment writing KANA as typed, until decoding rewrites it."
  (wime-japanese--make-segment kana (substring-no-properties kana)))

(defun wime-japanese--write (window segments)
  "Write SEGMENTS, (KANA-COUNT . TEXT) pairs, over WINDOW's kana.
Segments that stay the same keep their text; point keeps its kana position."
  (let* ((kana (wime-japanese--kana window)) (cursor (wime-japanese--cursor window))
         (start (car (car window))) (k 0)
         (new (mapcar (lambda (s) (prog1 (wime-japanese--make-segment (substring kana k (+ k (car s))) (cdr s))
                                    (cl-incf k (car s))))
                      segments))
         (old (mapcar #'caddr window))
         (same (lambda (a b) (and (equal (wime-japanese--segment-text a) (wime-japanese--segment-text b))
                                  (equal (wime-japanese--segment-kana a) (wime-japanese--segment-kana b)))))
         (prefix (cl-loop for a in old for b in new while (funcall same a b) count t))
         (suffix (cl-loop for a in (reverse (nthcdr prefix old)) for b in (reverse (nthcdr prefix new))
                          while (funcall same a b) count t)))
    (when (< prefix (length old))
      (wime-japanese--edit (car (nth prefix window)) (cadr (nth (- (length old) suffix 1) window))
                           (seq-subseq new prefix (- (length new) suffix))))
    ;; Point stays at the first segment boundary at or after its kana.
    (let ((position start) (n 0))
      (cl-loop for segment in new while (< n cursor)
               do (cl-incf n (length (wime-japanese--segment-kana segment)))
               do (cl-incf position (length (wime-japanese--segment-text segment))))
      (goto-char position))))

(defun wime-japanese--after-change (start end _length)
  "Turn segments an ordinary edit changed into ordinary text."
  (unless wime-japanese--internal
    (cl-incf wime-japanese--generation)
    (dolist (position (list (1- start) start (1- end) end))
      (when-let* ((segment (and (<= (point-min) position) (< position (point-max))
                                (get-text-property position 'wime-japanese-segment))))
        (let ((from (if (and (> position (point-min))
                             (eq (get-text-property (1- position) 'wime-japanese-segment) segment))
                        (previous-single-property-change position 'wime-japanese-segment nil (point-min))
                      position))
              (to (next-single-property-change position 'wime-japanese-segment nil (point-max))))
          (unless (equal (buffer-substring-no-properties from to) (wime-japanese--segment-text segment))
            (let ((wime-japanese--internal t))
              (remove-text-properties from to '(wime-japanese-segment nil)))))))))

;;; Display

(defun wime-japanese--converting-p ()
  (or (not (string-empty-p wime-japanese--pending)) wime-japanese--choices))

(defun wime-japanese--decorate ()
  "Mark unconfirmed text, the window being converted, and kana typed with Shift."
  (mapc #'delete-overlay wime-japanese--decorations)
  (setq wime-japanese--decorations nil)
  (cl-flet ((mark (from to face priority)
              (let ((overlay (make-overlay from to nil t nil)))
                (overlay-put overlay 'face face)
                (overlay-put overlay 'priority priority)
                (overlay-put overlay 'evaporate t)
                (push overlay wime-japanese--decorations))))
    (save-restriction
      (widen)
      (let ((position (point-min)))
        (while (setq position (text-property-not-all position (point-max) 'wime-japanese-segment nil))
          (let ((end (or (text-property-any position (point-max) 'wime-japanese-segment nil) (point-max))))
            (mark position end 'wime-edit-pending-face 1)
            (setq position end)))))
    (when-let* (((wime-japanese--converting-p)) (window (wime-japanese--window)))
      (mark (car (car window)) (cadr (car (last window))) 'wime-edit-active-face 2)
      (pcase-dolist (`(,from ,_ ,segment) window)
        ;; Kana still written as typed show their Shift marks.
        (let ((kana (wime-japanese--segment-kana segment)))
          (when (equal (substring-no-properties kana) (wime-japanese--segment-text segment))
            (dotimes (i (length kana))
              (when (get-text-property i 'wime-japanese-shift kana)
                (mark (+ from i) (+ from i 1) 'wime-japanese-shift-face 3)))))))))

(defun wime-japanese--display ()
  "Show pending keys and candidates at point, and mark unconfirmed text."
  (when (overlayp wime-japanese--overlay) (delete-overlay wime-japanese--overlay))
  (setq wime-japanese--overlay nil)
  (when (wime-japanese--converting-p)
    (setq wime-japanese--overlay (make-overlay (point) (point)))
    (overlay-put wime-japanese--overlay 'after-string
                 (concat (propertize wime-japanese--pending 'face 'wime-aux-face)
                         (when (and wime-show-candidates wime-japanese--candidates)
                           (concat " " (wime--format-list wime-japanese--candidates -1))))))
  (wime-japanese-active-mode (if (wime-japanese--converting-p) 1 -1))
  (wime-japanese--decorate))

(defun wime-japanese--fallbacks (kana)
  "Hiragana and katakana of KANA as choices covering all of it; with Shift
used, first the shifted kana in katakana and the rest in hiragana."
  (let ((plain (substring-no-properties kana)))
    (mapcar (lambda (text) (list text (cons (length kana) text)))
            (delete-dups
             (append (when (wime-japanese--explicit-p kana)
                       (list (concat (cl-loop for c across plain for i from 0
                                              collect (if (get-text-property i 'wime-japanese-shift kana)
                                                          (aref (japanese-katakana (string c)) 0) c)))))
                     (list plain (japanese-katakana plain)))))))

(defun wime-japanese--offer (choices)
  (setq wime-japanese--choices (seq-take choices wime-candidates-max)
        wime-japanese--candidates (mapcar #'car wime-japanese--choices)))

(defun wime-japanese--refresh ()
  "Offer the window's kana and ask the worker to decode them after a pause."
  (let ((window (wime-japanese--window)))
    (wime-japanese--offer (and window (wime-japanese--fallbacks (wime-japanese--kana window))))
    (setq wime-japanese--point (point))
    (wime-japanese--display)
    (wime-japanese--schedule window)))

(defun wime-japanese--reset ()
  "Stop converting, keeping unconfirmed text and its kana."
  (setq wime-japanese--pending "" wime-japanese--choices nil wime-japanese--candidates nil
        wime-japanese--point nil)
  (wime-neural--reset)
  (wime-japanese--display))

;;; Worker

(defun wime-japanese--schedule (window)
  "Ask the worker for readings of WINDOW after a pause, as wime does for Chinese."
  (wime-neural--reset)
  (when (and window (not wime-neural--failed) wime-neural-rescorer
             wime-japanese-model-file wime-japanese-table-file)
    (let* ((kana (wime-japanese--kana window)) (start (car (car window)))
           (job (wime-neural--make-job
                 :id (cl-incf wime-neural--serial) :buffer (current-buffer)
                 :input kana :position start :edit-generation wime-japanese--generation
                 :context (save-excursion
                            (goto-char start)
                            (concat (wime-neural--prefix) (wime--context-at-point wime-neural-context-limit)))
                 :method 'japanese :syllables (length kana))))
      (setq wime-neural--job job
            wime-neural--timer (run-with-timer wime-neural-delay nil #'wime-neural--enqueue job)))))

(defun wime-japanese--current-p (job)
  "Whether JOB decodes the window at point as it still stands."
  (and (= wime-japanese--generation (wime-neural--job-edit-generation job))
       (wime-japanese--converting-p)
       (let ((window (wime-japanese--window)))
         (and window (= (car (car window)) (wime-neural--job-position job))
              (equal (wime-japanese--kana window) (wime-neural--job-input job))))))

(defun wime-japanese--request (job)
  (let ((kana (wime-neural--job-input job)))
    (json-serialize
     `(:id ,(wime-neural--job-id job) :operation "japanese" :context ,(wime-neural--job-context job)
       :kana ,(substring-no-properties kana) :pending []
       :explicit ,(if (wime-japanese--explicit-p kana) t :false)
       :shift ,(wime-japanese--shift kana) :pending_shift :false
       :syllables ,(length kana) :candidates []))))

(defun wime-japanese--segments (value count)
  "VALUE as (KANA-COUNT . TEXT) segments covering COUNT kana, or nil."
  (let ((segments (cl-loop for pair across (if (vectorp value) value [])
                           when (and (vectorp pair) (= (length pair) 2) (natnump (aref pair 0))
                                     (> (aref pair 0) 0) (stringp (aref pair 1))
                                     (not (string-empty-p (aref pair 1))))
                           collect (cons (aref pair 0) (aref pair 1)))))
    (and (= (length segments) (length value)) (= (apply #'+ (mapcar #'car segments)) count)
         segments)))

(defun wime-japanese--apply (job response)
  "Write RESPONSE's best reading over the window, keeping the rest as choices."
  (let* ((kana (wime-neural--job-input job))
         (decoded (cl-loop for candidate across (or (alist-get 'candidates response) [])
                           for segments = (and (vectorp candidate) (= (length candidate) 3)
                                               (stringp (aref candidate 0))
                                               (wime-japanese--segments (aref candidate 2) (length kana)))
                           when (and segments (equal (apply #'concat (mapcar #'cdr segments)) (aref candidate 0)))
                           collect (cons (aref candidate 0) segments))))
    (when decoded
      (wime-japanese--write (wime-japanese--window) (cdar decoded))
      (wime-japanese--offer (append decoded (seq-remove (lambda (c) (assoc (car c) decoded))
                                                        (wime-japanese--fallbacks kana))))
      (setq wime-japanese--point (point))
      (wime-japanese--display))))

(defun wime-japanese--settle ()
  "Send a scheduled decode at once and wait for every outstanding one."
  (when-let* ((job wime-neural--job))
    (when (memq wime-neural--timer timer-list)
      (cancel-timer wime-neural--timer)
      (wime-neural--enqueue job))
    (while (memq job (list wime-neural--busy wime-neural--pending))
      (accept-process-output wime-neural--process 0.05))))

;;; Input method

(defun wime-japanese--to-boundary ()
  "Move point off the inside of a segment, to its end."
  (when-let* ((segment (wime-japanese--segment-at (point))))
    (when (eq segment (wime-japanese--segment-at (1- (point))))
      (goto-char (next-single-property-change (point) 'wime-japanese-segment nil (point-max))))))

(defun wime-japanese--insert (kana)
  "Insert KANA at point as typed, joining the run there."
  (wime-japanese--edit (point) (point) (list (wime-japanese--provisional kana)))
  (forward-char (length kana)))

(defun wime-japanese--type (key)
  (wime-japanese--to-boundary)
  (pcase-let ((`(,kana . ,pending) (wime-japanese--convert (concat wime-japanese--pending (string key)))))
    (setq wime-japanese--pending pending)
    (if (string-empty-p kana)
        (progn (setq wime-japanese--point (point)) (wime-japanese--display))
      (wime-japanese--insert kana)
      (wime-japanese--refresh))))

(defun wime-japanese--flush ()
  "Write the kana a complete pending rule spells, such as a final n; return
the keys left over."
  (let* ((pending wime-japanese--pending)
         (rule (gethash (wime-japanese--unshifted pending) (car (wime-japanese--rules)))))
    (setq wime-japanese--pending "")
    (if (not rule) pending
      (wime-japanese--insert (wime-japanese--mark (cdr rule) pending))
      (substring pending (- (length pending) (length (car rule)))))))

(defun wime-japanese--uncommitted-p ()
  (text-property-not-all (point-min) (point-max) 'wime-japanese-segment nil))

(defun wime-japanese--confirm ()
  "Make every unconfirmed segment in the accessible buffer ordinary text."
  (let ((wime-japanese--internal t))
    (remove-text-properties (point-min) (point-max) '(wime-japanese-segment nil)))
  (wime-neural--forget-prefix)
  (wime-japanese--reset))

(defun wime-japanese-input-method (key)
  "Process physical KEY for the selected Japanese input scheme."
  (if (wime--text-read-only-p)
      (list key)
    (let ((key (wime-japanese--translate-key key)))
      (cond
       ((wime-japanese--key-p key) (wime-japanese--type key) nil)
       ((and (= key ?\s) (or (wime-japanese--converting-p) (wime-japanese--uncommitted-p)))
        (wime-japanese-finish) nil)
       (t (wime-japanese-pause) (list key))))))

(defun wime-japanese-finish ()
  "Confirm the readings shown, after the outstanding decode."
  (interactive)
  (let ((generation wime-japanese--generation) (rest (wime-japanese--flush)))
    (unless (string-empty-p rest)
      (wime-japanese--insert rest))
    (unless (= generation wime-japanese--generation)
      (wime-japanese--refresh)))
  (wime-japanese--settle)
  (wime-japanese--confirm))

(defun wime-japanese-choose ()
  "Confirm the candidate the digit of this C-digit key numbers."
  (interactive)
  (let* ((digit (event-basic-type last-command-event))
         (choice (and (wime--digit-p digit) (nth (wime--digit-index digit) wime-japanese--choices)))
         (window (wime-japanese--window)))
    (when (and choice window)
      (wime-japanese--write window (cdr choice))
      (goto-char (cadr (car (last (wime-japanese--window)))))
      (wime-japanese--confirm))))

(defun wime-japanese-delete-backward ()
  "Delete the last pending key, else the last kana before point."
  (interactive)
  (if (not (string-empty-p wime-japanese--pending))
      (progn (setq wime-japanese--pending (substring wime-japanese--pending 0 -1))
             (wime-japanese--display))
    (wime-japanese--to-boundary)
    (if-let* ((group (car (last (seq-filter (lambda (g) (< (car g) (point))) (wime-japanese--run)))))
              ((= (cadr group) (point))))
        (let ((kana (substring (wime-japanese--segment-kana (nth 2 group)) 0 -1)))
          (wime-japanese--edit (car group) (cadr group)
                               (unless (string-empty-p kana) (list (wime-japanese--provisional kana))))
          (goto-char (+ (car group) (length kana)))
          (wime-japanese--refresh))
      (backward-delete-char-untabify 1))))

(defun wime-japanese-commit-kana ()
  "Confirm the run at point as the kana typed, those typed with Shift in katakana."
  (interactive)
  (let ((rest (wime-japanese--flush)))
    (pcase-dolist (`(,from ,to ,segment) (reverse (wime-japanese--run)))
      (let ((kana (wime-japanese--segment-kana segment)))
        (wime-japanese--edit from to (list (wime-japanese--make-segment
                                            kana (car (car (wime-japanese--fallbacks kana))))))))
    (goto-char (if-let* ((run (wime-japanese--run))) (cadr (car (last run))) (point)))
    (insert rest))
  (wime-japanese--confirm))

(defun wime-japanese-pause ()
  "Stop converting; unconfirmed text keeps its kana for later typing."
  (interactive)
  ;; A complete rule, such as a final n, is written; other pending keys go.
  (wime-japanese--flush)
  (wime-japanese--reset))

(defun wime-japanese-escape ()
  "Suspend conversion and leave Evil insertion with the same key."
  (interactive)
  (wime-japanese-pause)
  (when (bound-and-true-p evil-local-mode)
    (evil-normal-state)))

(defun wime-japanese-cancel ()
  "Delete the run at point and stop converting."
  (interactive)
  (setq wime-japanese--pending "")
  (when-let* ((run (wime-japanese--run)))
    (wime-japanese--edit (car (car run)) (cadr (car (last run))) nil))
  (wime-japanese--reset))

(defun wime-japanese--post-command ()
  "Stop converting once point leaves the place conversion left it."
  (when (and (wime-japanese--converting-p) (not (eql (point) wime-japanese--point)))
    (wime-japanese-pause))
  (wime-japanese--decorate))

(defvar wime-japanese-active-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "DEL") #'wime-japanese-delete-backward)
    (define-key map (kbd "<backspace>") #'wime-japanese-delete-backward)
    (define-key map (kbd "RET") #'wime-japanese-commit-kana)
    (define-key map (kbd "<return>") #'wime-japanese-commit-kana)
    (define-key map (kbd "<escape>") #'wime-japanese-escape)
    (dotimes (i 10)
      (define-key map (kbd (format "C-%d" i)) #'wime-japanese-choose))
    map)
  "Keymap installed while Japanese text is being converted.")

(define-minor-mode wime-japanese-active-mode
  "Minor mode installed while Japanese text is being converted."
  :lighter nil :init-value nil :keymap nil
  (if wime-japanese-active-mode
      (internal-push-keymap wime-japanese-active-mode-map 'overriding-terminal-local-map)
    (internal-pop-keymap wime-japanese-active-mode-map 'overriding-terminal-local-map)))

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
  (setq-local deactivate-current-input-method-function #'wime-japanese-deactivate)
  (add-hook 'post-command-hook #'wime-japanese--post-command nil t)
  (add-hook 'after-change-functions #'wime-japanese--after-change nil t))

(defun wime-japanese-deactivate ()
  "Deactivate Japanese input, confirming the text shown."
  (when (or (wime-japanese--converting-p) (wime-japanese--uncommitted-p))
    (wime-japanese-finish))
  (remove-hook 'post-command-hook #'wime-japanese--post-command t)
  (remove-hook 'after-change-functions #'wime-japanese--after-change t)
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
