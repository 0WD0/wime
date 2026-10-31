;;; wime-edit.el --- Editable Chinese compositions -*- lexical-binding: t; -*-

;; SPDX-License-Identifier: GPL-3.0-or-later

;;; Commentary:
;; Buffer characters carry immutable reading records, so ordinary edits and
;; undo retain the readings of untouched characters.  Literal text separates
;; Chinese runs, except punctuation and `wime-english-mode' text typed inside
;; a run: they become literal units of it, and the run converts as one
;; sentence around them.  Only the run at point participates in conversion;
;; switching input methods suspends conversion without throwing away those
;; records.

;;; Code:

(require 'wime)

(declare-function evil-normal-state "evil-states" (&optional arg))

(defcustom wime-edit-max-syllables 24
  "Maximum Chinese characters reconsidered together around point.
Literal text inside the run does not count."
  :type 'natnum :group 'wime)

(defface wime-edit-pending-face
  '((t (:inherit secondary-selection :underline t)))
  "Face for suspended Chinese text whose readings are still editable."
  :group 'wime)

(defface wime-edit-active-face
  '((t (:inherit highlight :underline t)))
  "Face for the Chinese text currently being converted."
  :group 'wime)

(defface wime-edit-aux-face
  '((t (:inherit wime-aux-face :height 0.75 :weight bold)))
  "Face for auxiliary keys attached to their Chinese character."
  :group 'wime)

(cl-defstruct (wime-edit--unit (:constructor wime-edit--make-unit))
  keys pin original (database wime-database-file) literal)

(defvar-local wime-edit--start nil)
(defvar-local wime-edit--end nil)
(defvar-local wime-edit--units nil)
(defvar-local wime-edit--generation 0)
(defvar-local wime-edit--decoration nil)
(defvar-local wime-edit--pending-decorations nil)
(defvar-local wime-edit--aux-decorations nil)
(defvar-local wime-edit--correction-decorations nil)
(defvar-local wime-edit--correction-layout nil)
(defvar-local wime-edit--internal nil)
(defvar-local wime-edit--undoing nil)
(defvar-local wime-edit--paused-point nil)
(defvar wime-edit-mode)

;; Plain insertions next to a Chinese character must not inherit its reading.
(add-to-list 'text-property-default-nonsticky '(wime-edit-unit . t))

(defun wime-edit--unit-at (position)
  "Return the reading at accessible POSITION, if any."
  (when (and (<= (point-min) position) (< position (point-max)))
    (let ((unit (get-text-property position 'wime-edit-unit)))
      (and (wime-edit--unit-p unit)
           (equal (wime-edit--unit-database unit) wime-database-file) unit))))

(defun wime-edit--editable-unit-at (position)
  (and (wime-edit--unit-at position)
       (not (get-char-property position 'read-only))
       (not (invisible-p position))
       (wime-edit--unit-at position)))

(defun wime-edit--active-p ()
  "Whether an editable Chinese run is currently selected."
  (and wime-edit-mode (markerp wime-edit--start) (marker-buffer wime-edit--start)
       (markerp wime-edit--end) (marker-buffer wime-edit--end)
       (eq (marker-buffer wime-edit--start) (current-buffer))
       (eq (marker-buffer wime-edit--end) (current-buffer))
       (cl-every (lambda (u) (equal (wime-edit--unit-database u) wime-database-file))
                 wime-edit--units)))

(defun wime-edit--writable-p ()
  "Whether the active run is accessible, visible and editable."
  (and (wime-edit--active-p) (not buffer-read-only)
       (<= (point-min) wime-edit--start wime-edit--end (point-max))
       (let ((p (marker-position wime-edit--start)) (ok t))
         (while (and ok (< p wime-edit--end))
           (setq ok (not (or (get-char-property p 'read-only) (invisible-p p))))
           (cl-incf p))
         ok)))

(defun wime-edit--clear-decoration ()
  (when (overlayp wime-edit--decoration) (delete-overlay wime-edit--decoration))
  (mapc #'delete-overlay wime-edit--correction-decorations)
  (setq wime-edit--decoration nil wime-edit--correction-decorations nil
        wime-edit--correction-layout nil wime--span-choices nil))

(defun wime-edit--refresh-aux-decorations ()
  "Show auxiliary keys and typed tones beside each reading-bearing character."
  (mapc #'delete-overlay wime-edit--aux-decorations)
  (setq wime-edit--aux-decorations nil)
  (when wime-edit-mode
    (save-restriction
      (widen)
      (let ((position (point-min)))
        (while (< position (point-max))
          (let* ((unit (get-text-property position 'wime-edit-unit))
                 (end (next-single-property-change position 'wime-edit-unit nil (point-max)))
                 (keys (and (wime-edit--unit-p unit) (not (wime-edit--unit-literal unit))
                            (wime-edit--unit-keys unit)))
                 (aux (and keys (concat (upcase (cdar (wime--parse keys)))
                                        (mapconcat (lambda (tone) (string (aref wime--tone-marks (1- (cdr tone)))))
                                                   (wime--tones keys) "")))))
            (when (and aux (not (string-empty-p aux)))
              (cl-loop for p from position below end
                       do (let ((overlay (make-overlay p (1+ p))))
                            (overlay-put overlay 'priority 2)
                            (overlay-put overlay 'evaporate t)
                            (overlay-put overlay 'after-string
                                         (propertize aux 'face 'wime-edit-aux-face
                                                     'display '(raise 0.3)))
                            (push overlay wime-edit--aux-decorations))))
            (setq position end)))))))

(defun wime-edit--refresh-pending-decorations ()
  "Mark every suspended reading span without changing buffer text.
Rebuild from reading properties so ordinary edits and undo remain the source
of truth.  Widen only for display; confirmation still respects narrowing."
  (wime-edit--refresh-aux-decorations)
  (mapc #'delete-overlay wime-edit--pending-decorations)
  (setq wime-edit--pending-decorations nil)
  (when wime-edit-mode
    (save-restriction
      (widen)
      (let* ((position (point-min))
             (active (and (wime-edit--active-p) (< wime-edit--start wime-edit--end)))
             (active-start (and active (marker-position wime-edit--start)))
             (active-end (and active (marker-position wime-edit--end))))
        (while (< position (point-max))
          (cond
           ((and active-start (<= active-start position) (< position active-end))
            (setq position active-end))
           ((not (wime-edit--unit-p (get-text-property position 'wime-edit-unit)))
            (setq position (next-single-property-change position 'wime-edit-unit nil (point-max))))
           (t
            (let ((start position))
              (while (and (< position (point-max))
                          (not (and active-start (= position active-start)))
                          (wime-edit--unit-p
                           (get-text-property position 'wime-edit-unit)))
                (setq position (next-single-property-change position 'wime-edit-unit nil (point-max))))
              (let ((overlay (make-overlay start position nil nil nil)))
                (overlay-put overlay 'face 'wime-edit-pending-face)
                (overlay-put overlay 'priority 1)
                (overlay-put overlay 'help-echo "暂存文字；移动光标继续修改，空格确认")
                (overlay-put overlay 'evaporate t)
                (push overlay wime-edit--pending-decorations))))))))))

(defun wime-edit--reset ()
  "Forget active conversion state, retaining buffer text and readings."
  (cl-incf wime-edit--generation)
  (dolist (marker (list wime-edit--start wime-edit--end))
    (when (markerp marker) (set-marker marker nil)))
  (setq wime-edit--start nil wime-edit--end nil wime-edit--units nil)
  (wime-edit--clear-decoration)
  (wime-edit--refresh-pending-decorations))

(defun wime-edit-pause ()
  "Hide conversion choices, visibly marking the retained editable readings."
  (interactive)
  (wime--reset)
  (setq wime-edit--paused-point (point)))

(defun wime-edit-escape ()
  "Suspend editable input and leave Evil insertion with the same key."
  (interactive)
  (wime-edit-pause)
  (when (bound-and-true-p evil-local-mode)
    (evil-normal-state)))

(defun wime-edit--bounds ()
  "Find the bounded Chinese run touching point, within the input area."
  (let* ((p (cond ((wime-edit--editable-unit-at (point)) (point))
                  ((wime-edit--editable-unit-at (1- (point))) (1- (point))))))
    (when p
      (let ((start p) (end (1+ p)) (floor (wime--input-start))
            (field (get-char-property p 'field))
            (limit (max 1 wime-edit-max-syllables)))
        (while (and (> start floor) (wime-edit--editable-unit-at (1- start))
                    (equal field (get-char-property (1- start) 'field)))
          (cl-decf start))
        (while (and (< end (point-max)) (wime-edit--editable-unit-at end)
                    (equal field (get-char-property end 'field)))
          (cl-incf end))
        ;; A bounded window keeps idle decoding practical in long paragraphs.
        (let ((p (min end (point))) (n 0))
          (while (and (> p start)
                      (or (wime-edit--unit-literal (wime-edit--unit-at (1- p))) (< n limit)))
            (cl-decf p)
            (unless (wime-edit--unit-literal (wime-edit--unit-at p)) (cl-incf n)))
          (setq start p))
        (let ((p start) (n 0))
          (while (and (< p end)
                      (or (wime-edit--unit-literal (wime-edit--unit-at p)) (< n limit)))
            (unless (wime-edit--unit-literal (wime-edit--unit-at p)) (cl-incf n))
            (cl-incf p))
          (setq end p))
        ;; A run starts with a syllable; earlier literal text is context.
        (while (and (< start end) (wime-edit--unit-literal (wime-edit--unit-at start)))
          (cl-incf start))
        (when (and (< start end) (<= start (point)))
          (cons start end))))))

(defun wime-edit--read-units (start end)
  (cl-loop for p from start below end collect (wime-edit--unit-at p)))

(defun wime-edit--syllables (units)
  "UNITS without literal text: what the engine converts."
  (seq-remove #'wime-edit--unit-literal units))

(defun wime-edit--literals ()
  "Literal text in the run as (SYLLABLES-BEFORE . TEXT) groups."
  (let ((n 0) groups)
    (dolist (unit wime-edit--units (nreverse groups))
      (cond ((not (wime-edit--unit-literal unit)) (cl-incf n))
            ((eql (caar groups) n)
             (setcdr (car groups) (concat (cdar groups) (wime-edit--unit-keys unit))))
            (t (push (cons n (wime-edit--unit-keys unit)) groups))))))

(defun wime-edit--merge (text)
  "Run text: converted TEXT, one character per syllable, around the literals."
  (let ((i 0))
    (mapconcat (lambda (unit)
                 (if (wime-edit--unit-literal unit) (wime-edit--unit-keys unit)
                   (prog1 (substring text i (1+ i)) (cl-incf i))))
               wime-edit--units "")))

(defun wime-edit--offset (index)
  "Run offset of syllable INDEX, or the run's end past the last one."
  (or (nth index (cl-loop for unit in wime-edit--units for offset from 0
                          unless (wime-edit--unit-literal unit) collect offset))
      (length wime-edit--units)))

(defun wime-edit--load-input ()
  "Derive the existing engine's input and hard pins from reading records."
  (let ((syllables (wime-edit--syllables wime-edit--units)))
    (setq wime--input (mapconcat #'wime-edit--unit-keys syllables "")
          wime--pins
          (cl-loop for unit in syllables for i from 0
                   when (wime-edit--unit-pin unit)
                   collect (list i (wime-edit--unit-pin unit)
                                 (caar (wime--parse (wime-edit--unit-keys unit))))))))

(defun wime-edit--adopt (start end)
  "Select the reading-bearing buffer run START..END."
  (wime--reset)
  (setq wime-edit--start (copy-marker start)
        wime-edit--end (copy-marker end t)
        wime-edit--units (wime-edit--read-units start end))
  (wime-edit--load-input))

(defun wime-edit--sync ()
  "Select the run at point, leaving literal text and suspended input alone."
  (when (and wime-edit-mode wime-mode (not wime-edit--internal))
    (let ((bounds (wime-edit--bounds)))
      (cond
       ((null bounds) (when wime-edit--start (wime--reset)))
       ((and (wime-edit--active-p)
             (= (car bounds) wime-edit--start) (= (cdr bounds) wime-edit--end)) nil)
       (t
        (wime-edit--adopt (car bounds) (cdr bounds))
        (if (wime-edit--writable-p) (wime--refresh) (wime--reset)))))))

(defun wime-edit--post-command ()
  (unless (memq this-command '(undo undo-only undo-redo))
    (setq wime-edit--undoing nil))
  (unless (or wime-edit--internal wime-edit--undoing
              (eq (point) wime-edit--paused-point)
              (memq this-command '(wime-edit-pause wime-edit-escape wime-edit-finish
                                   wime-edit-return toggle-input-method wime-toggle-english
                                   wime-dictionary-mode
                                   undo undo-only undo-redo)))
    (wime-edit--sync))
  (when (and (wime-edit--active-p)
             (not (equal wime-edit--correction-layout (wime-edit--layout-signature))))
    (wime-edit--render)))

(defun wime-edit--before-change (_start _end)
  (unless wime-edit--internal
    ;; All outstanding responses refer to the previous buffer contents.
    (when (wime-edit--active-p) (wime--reset))))

(defun wime-edit--after-change (start end _old-length)
  (unless wime-edit--internal
    (setq wime-edit--undoing undo-in-progress)
    ;; Undo restores exact reading properties.  Other inserted/replaced text,
    ;; including yanked text, is literal and cannot acquire a stale reading.
    (unless undo-in-progress
      (let ((wime-edit--internal t))
        (with-silent-modifications
          (remove-text-properties start end '(wime-edit-unit nil)))))
    (wime-edit--refresh-pending-decorations)))

(defun wime-edit--candidate-text ()
  "Return a full reading, with a one-character fallback for incomplete keys.
While a model decode is pending, characters already shown keep their text:
dictionary guesses would replace the last decoded reading, and a character
leaving the window at that moment would keep the guess."
  (let* ((syllables (wime-edit--syllables wime-edit--units))
         (n (length syllables))
         (candidate (seq-find
                     (lambda (c) (and (= (wime-candidate-end c) n)
                                      (= (length (wime-candidate-text c)) n)))
                     wime--candidate-records))
         (guess (and candidate (wime-candidate-text candidate)))
         (provisional (and (not wime--model-ranked)
                           (bound-and-true-p wime-neural-mode)
                           (not (bound-and-true-p wime-neural--failed))
                           (bound-and-true-p wime-neural-rescorer)
                           (eq (bound-and-true-p wime-neural-search-method) 'decode)))
         (i -1))
    (cond
     ((not (wime-edit--decodable-p))
      (mapconcat #'wime-edit--unit-text syllables ""))
     ((and guess (not provisional)) guess)
     (t
      (let (wime--candidate-records wime--search-cache wime--pins)
        (mapconcat (lambda (unit)
                     (cl-incf i)
                     (or (and (wime-edit--unit-pin unit)
                              (char-to-string (wime-edit--unit-pin unit)))
                         (and provisional (wime-edit--shown unit))
                         (and guess (substring guess i (1+ i)))
                         (car (wime--candidates-for
                               (caar (wime--parse (wime-edit--unit-keys unit)))
                               (cdar (wime--parse (wime-edit--unit-keys unit)))))
                         (substring (wime-edit--unit-keys unit) 0 1)))
                   syllables ""))))))

(defun wime-edit--decodable-p ()
  "Whether flattening this run preserves its syllable boundaries."
  (cl-every (lambda (u) (= (length (caar (wime--parse (wime-edit--unit-keys u)))) 2))
            (butlast (wime-edit--syllables wime-edit--units))))

(defun wime-edit--pending-refresh ()
  "Display an incomplete interior syllable without changing its neighbors."
  (setq wime--candidate-records nil wime--candidates nil
        wime--candidate-scores nil wime--span-choices nil)
  (wime-active-mode 1)
  (wime--display)
  (run-hooks 'wime-candidates-updated-hook))

(defun wime-edit--shown (unit)
  "UNIT's character in the run, or nil when UNIT is not shown yet."
  (cl-loop for p from (marker-position wime-edit--start) below wime-edit--end
           when (eq (wime-edit--unit-at p) unit) return (char-to-string (char-after p))))

(defun wime-edit--unit-text (unit)
  "Keep an existing UNIT's displayed text while an interior key is pending."
  (or (wime-edit--shown unit)
      (and (wime-edit--unit-pin unit) (char-to-string (wime-edit--unit-pin unit)))
      (substring (wime-edit--unit-keys unit) 0 1)))

(defun wime-edit--replace (text units &optional cursor)
  "Replace the selected run with TEXT and UNITS, preserving point and mark.
CURSOR, when provided, is the desired offset after a user input operation.
All real edits remain undoable; ordinary buffer modification hooks still run."
  (when (wime-edit--writable-p)
    (let* ((start (marker-position wime-edit--start))
           (end (marker-position wime-edit--end))
           (old-length (- end start))
           (position (point)) (mark-position (mark t))
           (old (buffer-substring start end))
           (old-units (wime-edit--read-units start end))
           (replacement (copy-sequence text))
           (wime-edit--internal t))
      (cl-loop for unit in units for i from 0
               for source = (or (cl-position unit old-units :test #'eq)
                                (and (= old-length (length text)) i))
               do (when source
                    (set-text-properties i (1+ i) (text-properties-at source old) replacement))
               do (put-text-property i (1+ i) 'wime-edit-unit unit replacement))
      (unless (equal-including-properties old replacement)
        (atomic-change-group
          (save-excursion
            (goto-char start)
            ;; Retain unrelated mode-owned properties on existing characters.
            (if (= old-length (length text))
                (cl-loop for i from 0 below old-length
                         for unit in units
                         do (unless (= (char-after (+ start i)) (aref text i))
                              (goto-char (+ start i))
                              (let ((props (text-properties-at (point))))
                                (delete-char 1)
                                (insert (apply #'propertize (substring text i (1+ i)) props))))
                         do (put-text-property (+ start i) (+ start i 1) 'wime-edit-unit unit))
              (let ((prefix 0) (suffix 0) (new-length (length text)))
                (while (and (< prefix (min old-length new-length))
                            (equal-including-properties (substring old prefix (1+ prefix))
                                                        (substring replacement prefix (1+ prefix))))
                  (cl-incf prefix))
                (while (and (< suffix (- (min old-length new-length) prefix))
                            (equal-including-properties (substring old (- old-length suffix 1) (- old-length suffix))
                                                        (substring replacement (- new-length suffix 1) (- new-length suffix))))
                  (cl-incf suffix))
                (goto-char (+ start prefix))
                (delete-region (point) (- end suffix))
                (insert (substring replacement prefix (- new-length suffix))))))
          (set-marker wime-edit--start start)
          (set-marker wime-edit--end (+ start (length text)))))
      (setq wime-edit--units units)
      (goto-char (cond (cursor (+ start cursor))
                       ((< position start) position)
                       ((<= position end) (+ start (min (- position start) (length text))))
                       (t (+ position (- (length text) old-length)))))
      (when (and (= old-length (length text)) mark-position (<= start mark-position end))
        (set-marker (mark-marker) (+ start (min (- mark-position start) (length text))))))))

(defun wime-edit--layout-signature ()
  "Return the window geometry relevant to correction rows."
  (list (point-min) (point-max) (copy-tree face-remapping-alist)
        (mapcar (lambda (window)
                  (list window (window-body-width window) (window-hscroll window)
                        display-line-numbers
                        (with-selected-window window (line-number-display-width t))))
                (get-buffer-window-list (current-buffer) nil t))))

(defun wime-edit--correction-groups (text)
  "Return (START END OPTIONS) groups, where OPTIONS contain (LABEL . TEXT).
Span labels share one sequence across the whole run.  Whole-sentence labels
keep their original indices so the existing digit selection remains valid."
  (setq wime--span-choices nil)
  (when (and wime-show-candidates wime--candidates)
    (if (wime--spans-p)
        (let ((label 0) groups choices)
          (pcase-dolist (`(,start ,end ,alternatives)
                         (wime--uncertain-spans text wime--candidate-scores))
            (let (options)
              (dolist (alternative alternatives)
                (when (< label 10)
                  (cl-incf label)
                  (push (cons (% label 10) alternative) options)
                  (push (list start end alternative) choices)))
              (when options (push (list start end (nreverse options)) groups))))
          (setq wime--span-choices (nreverse choices))
          (nreverse groups))
      (let ((label 0) options)
        (dolist (candidate wime--candidates)
          (cl-incf label)
          (when (and (<= label 10) (= (length candidate) (length text))
                     (not (equal candidate text)))
            (push (cons (% label 10) candidate) options)))
        (when options (list (list 0 (length text) (nreverse options))))))))

(defun wime-edit--correction-label (option &optional window)
  "Return OPTION's number and dot, fitting one Han character in WINDOW."
  (let ((label (propertize (format "%d." (car option)) 'face 'wime-label-face)))
    (when (and window (display-graphic-p (window-frame window))
               (> (length (cdr option)) 0)
               (> (char-width (aref (cdr option) 0)) 1))
      (let* ((width (wime-edit--display-width label window))
             (limit (wime-edit--display-width
                     (propertize (substring (cdr option) 0 1) 'face 'wime-alternatives-face)
                     window)))
        ;; Rasterization can make two Latin glyphs a pixel wider than one Han
        ;; glyph even in the same monospaced font.  Shrink only that label,
        ;; remeasuring because font sizes are rounded to pixels too.
        (when (and (> limit 0) (> width limit))
          (cl-loop for scale = (/ (float limit) width) then (* scale 0.95)
                   repeat 20
                   do (put-text-property 0 (length label) 'face
                                         `(:inherit wime-label-face :height ,scale) label)
                   until (<= (wime-edit--display-width label window) limit)))))
    label))

(defun wime-edit--correction-cell (option &optional window)
  (concat (wime-edit--correction-label option window)
          (propertize (cdr option) 'face 'wime-alternatives-face)))

(defun wime-edit--correction-rows (columns &optional window)
  "Pack (COLUMN . OPTIONS) COLUMNS into the earliest free display rows.
Keep each group's options consecutive, measuring each cell in WINDOW so
fonts use the same units as the source-text alignment."
  (let (rows)
    (dolist (column columns)
      (let* ((left (car column))
             (cells (mapcar (lambda (option)
                              (let ((text (wime-edit--correction-cell option window)))
                                (list left (wime-edit--display-width text window) text)))
                            (cdr column)))
             (start 0))
        ;; A neighboring group can share these rows whenever its actual cells
        ;; fit.  Do not reserve a full-width band or an extra two-character gap.
        (while (cl-loop for cell in cells for row from start
                        thereis (cl-some
                                 (lambda (other)
                                   (and (< (car cell) (- (+ (car other) (cadr other)) 1e-6))
                                        (< (car other) (- (+ (car cell) (cadr cell)) 1e-6))))
                                 (nth row rows)))
          (cl-incf start))
        (setq rows (nconc rows (make-list (max 0 (- (+ start (length cells)) (length rows))) nil)))
        (cl-loop for cell in cells for row from start do (push cell (nth row rows)))))
    (mapconcat
     (lambda (row)
       (concat "\n"
               (mapconcat
                (lambda (cell)
                  (concat (propertize " " 'display `(space :align-to ,(car cell)))
                          (nth 2 cell)))
                (sort row (lambda (a b) (< (car a) (car b)))) "")))
     rows "")))

(defun wime-edit--correction-location (position window)
  "Return (ANCHOR COLUMN SOFT-WRAP) for a correction at POSITION in WINDOW."
  (save-excursion
    (goto-char position)
    (let ((column (current-column)) (end (line-end-position)))
      (if (not window) (list end column nil)
        ;; Measure without our virtual candidate rows.  Normal text to the
        ;; right stays on its original screen line, above the candidates.
        (vertical-motion 0 window)
        (setq column
              (if (display-graphic-p (window-frame window))
                  ;; Pixel measurement includes the number gutter, while
                  ;; display-space alignment starts after it.
                  (/ (float (- (car (window-text-pixel-size window (point) position t))
                               (with-selected-window window (line-number-display-width t))))
                     (frame-char-width (window-frame window)))
                (- column (current-column))))
        (goto-char position)
        (vertical-motion 1 window)
        (list (min end (point)) column (< (point) end))))))

(defun wime-edit--display-width (text window)
  "Measure TEXT in the units used by display-space alignment."
  (if (and window (display-graphic-p (window-frame window)))
      (with-selected-window window
        (/ (float
            (if (>= emacs-major-version 31)
                (string-pixel-width text (window-buffer window))
              ;; Emacs 29/30's string-pixel-width cannot inherit the source
              ;; buffer's text scaling.  Measure with those face remappings.
              (let ((remapping face-remapping-alist))
                (with-temp-buffer
                  (setq-local face-remapping-alist remapping
                              display-line-numbers nil line-prefix nil wrap-prefix nil)
                  (insert text)
                  (car (buffer-text-pixel-size nil nil t))))))
           (frame-char-width (window-frame window))))
    (string-width text)))

(defun wime-edit--show-corrections (text)
  "Display correction columns underneath their source text."
  (let ((groups (wime-edit--correction-groups text))
        (windows (or (get-buffer-window-list (current-buffer) nil t) '(nil))))
    (dolist (group groups)
      (let ((overlay (make-overlay (+ wime-edit--start (wime-edit--offset (car group)))
                                   (+ wime-edit--start 1 (wime-edit--offset (1- (nth 1 group)))))))
        (overlay-put overlay 'face 'wime-uncertain-face)
        (overlay-put overlay 'priority 1)
        (push overlay wime-edit--correction-decorations)))
    (dolist (window windows)
      (let (lines)
        ;; Measure every column before inserting any virtual lines.
        (dolist (group groups)
          (pcase-let* ((`(,anchor ,column ,soft)
                        (wime-edit--correction-location
                         (+ wime-edit--start (wime-edit--offset (car group))) window))
                       (options (nth 2 group))
                       (width (apply #'max (mapcar (lambda (option)
                                                    (wime-edit--display-width
                                                     (wime-edit--correction-cell option window) window))
                                                  options)))
                       ;; Put the alternative's characters under the source;
                       ;; its number occupies the small gutter to the left.
                       (column (max 0 (- column (wime-edit--display-width
                                                 (wime-edit--correction-label (car options) window)
                                                 window))))
                       (column (if window
                                   (max 0 (min column (- (window-body-width window)
                                                        (with-selected-window window
                                                          (line-number-display-width 'columns))
                                                        width 1)))
                                 column))
                       (line (assq anchor lines)))
            (unless line
              (setq line (list anchor soft nil))
              (push line lines))
            (setf (nth 2 line) (append (nth 2 line) (list (cons column options))))))
        (dolist (line lines)
          (let ((overlay (make-overlay (car line) (car line))))
            (when window (overlay-put overlay 'window window))
            (overlay-put overlay 'after-string
                         (concat (wime-edit--correction-rows (nth 2 line) window)
                                 (when (nth 1 line) "\n")))
            (push overlay wime-edit--correction-decorations))))))
  (setq wime-edit--correction-layout (wime-edit--layout-signature)))

(defun wime-edit--render ()
  "Publish a reading in the buffer and show non-textual conversion details."
  (when (and (wime-edit--active-p) (wime-edit--writable-p))
    (let* ((text (wime-edit--candidate-text))
           (i -1)
           (units (mapcar
                   (lambda (unit)
                     (if (wime-edit--unit-literal unit) unit
                       (let ((pin (nth 1 (assq (cl-incf i) wime--pins))))
                         (if (and (wime-edit--unit-original unit)
                                  (equal pin (wime-edit--unit-pin unit))) unit
                           (wime-edit--make-unit
                            :keys (wime-edit--unit-keys unit) :pin pin
                            :original (or (wime-edit--unit-original unit) (aref text i)))))))
                   wime-edit--units)))
      (wime-edit--replace (wime-edit--merge text) units)
      (wime-edit--clear-decoration)
      (setq wime-edit--decoration (make-overlay wime-edit--start wime-edit--end))
      (overlay-put wime-edit--decoration 'face 'wime-edit-active-face)
      (overlay-put wime-edit--decoration 'help-echo
                   "空格确认，ESC 暂存；TAB 切换整句候选")
      (wime-edit--refresh-pending-decorations)
      (wime-edit--show-corrections text))))

(defun wime-edit--refresh (units cursor)
  "Update reading UNITS and place point at the syllable boundary CURSOR."
  (cl-incf wime-edit--generation)
  (setq wime-edit--units units wime--search-cache nil)
  (wime-edit--load-input)
  (cond
   ((wime-edit--syllables units)
    (wime--refresh)
    (goto-char (+ wime-edit--start cursor)))
   (t
    ;; Without syllables, remaining literal units are ordinary text.
    (wime-edit--replace (mapconcat #'wime-edit--unit-keys units "") units cursor)
    (let ((wime-edit--internal t))
      (with-silent-modifications
        (remove-text-properties wime-edit--start wime-edit--end '(wime-edit-unit nil))))
    (wime--reset))))

(defun wime-edit--type-code (key)
  (wime-edit--sync)
  (unless (wime-edit--active-p) (wime-edit--adopt (point) (point)))
  (let* ((offset (- (point) wime-edit--start))
         (previous (and (> offset 0) (nth (1- offset) wime-edit--units)))
         (partial (and previous (not (wime-edit--unit-literal previous))
                       (= (length (caar (wime--parse (wime-edit--unit-keys previous)))) 1)))
         (index (if partial (1- offset) offset))
         (unit (wime-edit--make-unit
                :keys (concat (and partial (wime-edit--unit-keys previous)) (char-to-string key))))
         (units (append (seq-take wime-edit--units index) (list unit)
                        (nthcdr (if partial offset index) wime-edit--units))))
    (wime-edit--refresh units (1+ index))))

(defun wime-edit--type-aux (key)
  "Attach KEY to the character immediately before point."
  (wime-edit--sync)
  (when (and (wime-edit--active-p) (> (point) wime-edit--start))
    (let* ((index (1- (- (point) wime-edit--start)))
           (unit (nth index wime-edit--units)))
      (when (and (not (wime-edit--unit-literal unit))
                 (= (length (caar (wime--parse (wime-edit--unit-keys unit)))) 2))
        (wime-edit--refresh
         (append (seq-take wime-edit--units index)
                 (list (wime-edit--make-unit
                        :keys (concat (wime-edit--unit-keys unit) (char-to-string key))
                        :original (wime-edit--unit-original unit)))
                 (nthcdr (1+ index) wime-edit--units)) (1+ index))
        t))))

(defun wime-edit--type-tone (tone)
  "Type TONE for the syllable immediately before point."
  (wime-edit--sync)
  (when (and (wime-edit--active-p) (> (point) wime-edit--start))
    (let* ((index (1- (- (point) wime-edit--start)))
           (unit (nth index wime-edit--units)))
      (unless (wime-edit--unit-literal unit)
        (wime-edit--refresh
         (append (seq-take wime-edit--units index)
                 (list (wime-edit--make-unit :keys (wime--retone (wime-edit--unit-keys unit) tone)
                                             :original (wime-edit--unit-original unit)))
                 (nthcdr (1+ index) wime-edit--units))
         (1+ index))))))

(defun wime-edit--type-literal (text)
  "Insert TEXT, a string or one key, as literal units of the run at point.
Return nil without a run; text before its first syllable is context."
  (setq wime-edit--undoing nil wime-edit--paused-point nil)
  (wime-edit--sync)
  (when (and (wime-edit--active-p) (> (point) wime-edit--start))
    (let ((offset (- (point) wime-edit--start)))
      (wime-edit--refresh
       (append (seq-take wime-edit--units offset)
               (mapcar (lambda (char) (wime-edit--make-unit :keys (string char) :literal t))
                       (if (stringp text) text (string text)))
               (nthcdr offset wime-edit--units))
       (+ offset (if (stringp text) (length text) 1)))
      t)))

(defun wime-edit--choose (text)
  "Fix the explicitly chosen full TEXT without ending its editable run."
  (when (= (length text) (length (wime-edit--syllables wime-edit--units)))
    (cl-incf wime-edit--generation)
    (setq wime--pins
          (cl-loop for unit in (wime-edit--syllables wime-edit--units) for i from 0
                   collect (list i (aref text i) (caar (wime--parse (wime-edit--unit-keys unit))))))
    (wime--refresh)))

(defun wime-edit--input-method (key)
  (setq wime-edit--undoing nil wime-edit--paused-point nil)
  (cond
   ((wime--code-char-p
     key (if (let ((unit (wime-edit--unit-at (1- (point)))))
               (and unit (not (wime-edit--unit-literal unit))
                    (= (length (caar (wime--parse (wime-edit--unit-keys unit)))) 1)))
             'final 'initial))
    (wime-edit--type-code key) nil)
   ((and (wime--aux-char-p key) (wime-edit--type-aux key)) nil)
   ((and (wime-edit--active-p) (wime--digit-p key))
    (when wime-show-candidates
      (if (wime--spans-p) (wime--select-digit key)
        (let ((text (nth (if (= key ?0) 9 (- key ?1)) wime--candidates)))
          (when text (wime-edit--choose text))))) nil)
   ((and (= key ?\s)
         (text-property-not-all (point-min) (point-max) 'wime-edit-unit nil))
    (wime-edit-finish) nil)
   ((wime--punctuation-for key)
    ;; Punctuation continues the sentence; only SPC or ESC ends conversion.
    (unless (wime-edit--type-literal (wime--punctuation-for key))
      (insert (wime--punctuation-for key)))
    nil)
   (t (wime-edit-pause) (list key))))

(defun wime-edit-delete-backward ()
  "Remove one auxiliary/key input at the character before point."
  (interactive)
  (wime-edit--sync)
  (if (and (wime-edit--active-p) (> (point) wime-edit--start))
      (let* ((index (1- (- (point) wime-edit--start)))
             (unit (nth index wime-edit--units))
             (keys (substring (wime-edit--unit-keys unit) 0 -1)))
        (wime-edit--refresh
         (append (seq-take wime-edit--units index)
                 (unless (string-empty-p keys) (list (wime-edit--make-unit :keys keys)))
                 (nthcdr (1+ index) wime-edit--units))
         (+ index (if (string-empty-p keys) 0 1))))
    (backward-delete-char-untabify 1)))

(defun wime-edit-finish ()
  "Confirm pending readings in the accessible buffer, retaining its text."
  (interactive)
  (wime-edit-pause)
  (let ((wime-edit--internal t) (inhibit-read-only t) (position (point-min)))
    ;; Literal units become ordinary text; the Chinese around them is
    ;; recorded and learned as separate runs.
    (while (< position (point-max))
      (let ((end (next-single-property-change position 'wime-edit-unit nil (point-max)))
            (unit (get-text-property position 'wime-edit-unit)))
        (when (and (wime-edit--unit-p unit) (wime-edit--unit-literal unit))
          (remove-text-properties position end '(wime-edit-unit nil)))
        (setq position end))))
  (let ((wime-edit--internal t))
    ;; An unmatched or unfinished syllable is still input.  Do not discard
    ;; its remaining keys when removing the per-character reading record.
    (save-excursion
      (goto-char (point-min))
      (while (< (point) (point-max))
        (let ((unit (wime-edit--unit-at (point))))
          (if (and unit (not (<= #x3400 (char-after) #x9fff))
                   (not (get-char-property (point) 'read-only))
                   (not buffer-read-only))
              (let ((keys (wime-edit--unit-keys unit)))
                (atomic-change-group (delete-char 1) (insert keys)))
            (forward-char)))))
    (save-excursion
      (goto-char (point-min))
      (while (< (point) (point-max))
        (if (not (wime-edit--unit-at (point))) (forward-char)
          (let ((start (point)))
            (while (and (< (point) (point-max)) (wime-edit--unit-at (point)))
              (forward-char))
            (let* ((end (point))
                   (units (wime-edit--read-units start end))
                   (raw (mapconcat #'wime-edit--unit-keys units ""))
                   (parsed (wime--parse raw))
                   (text (buffer-substring-no-properties start end))
                   (original (concat (cl-loop for u in units for i from 0
                                              collect (or (wime-edit--unit-original u) (aref text i))))))
              (when (and (= (length parsed) (length text))
                         (cl-every (lambda (u) (= (length (car u)) 2)) parsed)
                         (string-match-p "\\`[一-鿿]+\\'" text)
                         (wime--user-db))
                (sqlite-execute (wime--user-db) "INSERT INTO commits VALUES (?, ?, ?, ?, ?, ?)"
                                (list (float-time) raw (wime--full-code parsed) text original
                                      (save-excursion (goto-char start) (wime--context-at-point))))
                ;; Model revisions are not user corrections.  Only explicitly
                ;; pinned/aux-filtered choices teach the reading preference.
                (let ((baseline
                       (concat
                        (cl-loop for u in units for i from 0
                                 collect (if (or (wime-edit--unit-pin u)
                                                 (not (string-empty-p
                                                       (cdar (wime--parse (wime-edit--unit-keys u))))))
                                             (aref original i) (aref text i))))))
                  (unless (equal text baseline)
                    (wime--learn text parsed baseline)))))))))
    (atomic-change-group
      (let ((position (point-min)))
        (while (< position (point-max))
          (let ((end (next-single-property-change position 'wime-edit-unit nil (point-max))))
            (when (get-text-property position 'wime-edit-unit)
              ;; Only discard our annotation, including on text a mode has
              ;; since protected.  The text and its protection stay unchanged.
              (let ((inhibit-read-only t))
                (remove-text-properties position end '(wime-edit-unit nil))))
            (setq position end))))))
  (wime-edit--refresh-pending-decorations))

(defun wime-edit-return ()
  "Commit the active run as raw keys, or confirm and run normal Return."
  (interactive)
  (if (wime-edit--active-p)
      (when (wime-edit--writable-p)
        (let ((start (marker-position wime-edit--start))
              (end (marker-position wime-edit--end))
              (raw (mapconcat #'wime-edit--unit-keys wime-edit--units "")))
          (wime-edit-pause)
          (atomic-change-group
            (delete-region start end)
            (goto-char start)
            (insert raw))))
    (wime-edit-finish)
    (let* ((wime-edit-mode nil) (command (key-binding (kbd "RET"))))
      (when (and (commandp command) (not (eq command #'wime-edit-return)))
        (call-interactively command)))))

(defvar wime-edit--keymap
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "DEL") #'wime-edit-delete-backward)
    (define-key map (kbd "<backspace>") #'wime-edit-delete-backward)
    (define-key map (kbd "RET") #'wime-edit-return)
    (define-key map (kbd "<return>") #'wime-edit-return)
    (define-key map (kbd "TAB") #'wime-toggle-candidate-list)
    (define-key map (kbd "<escape>") #'wime-edit-escape)
    (dotimes (i 5)
      (define-key map (kbd (format "C-%d" (1+ i))) #'wime-tone))
    map))

(defvar wime-edit-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "RET") #'wime-edit-return)
    (define-key map (kbd "<return>") #'wime-edit-return)
    map))

;;;###autoload
(define-minor-mode wime-edit-mode
  "Keep Chinese readings attached to editable buffer characters.
SPC confirms pending text; ESC suspends it with a visible marker.
Moving back to a character resumes conversion; uppercase auxiliary codes
apply to the character before point.  Switching input methods preserves
readings for later edits."
  :lighter " 半"
  (if wime-edit-mode
      (progn
        (add-hook 'wime-reset-hook #'wime-edit--reset nil t)
        (add-hook 'post-command-hook #'wime-edit--post-command nil t)
        (add-hook 'before-change-functions #'wime-edit--before-change nil t)
        (add-hook 'after-change-functions #'wime-edit--after-change nil t)
        (wime-edit--refresh-pending-decorations))
    (wime-edit-finish)
    (remove-hook 'wime-reset-hook #'wime-edit--reset t)
    (remove-hook 'post-command-hook #'wime-edit--post-command t)
    (remove-hook 'before-change-functions #'wime-edit--before-change t)
    (remove-hook 'after-change-functions #'wime-edit--after-change t)))

(provide 'wime-edit)
;;; wime-edit.el ends here
