;;; wime-neural.el --- Local asynchronous neural decoding -*- lexical-binding: t; -*-

;; SPDX-License-Identifier: GPL-3.0-or-later

;; Package-Requires: ((emacs "29.1"))

;;; Commentary:
;; A persistent worker loads the character model once.  There is at most one
;; running request and one pending request; typing replaces pending work.
;; Decoded readings precede or replace dictionary candidates, per
;; `wime-decoding', while the input stays current.

;;; Code:

(require 'cl-lib)
(require 'json)
(require 'wime)

(defconst wime-neural--directory
  (expand-file-name "neural" (file-name-directory (or load-file-name buffer-file-name)))
  "This package's worker sources and environment project.")

;; The defaults are the evaluated decode setup, laid out as the README describes.
(defcustom wime-neural-program (expand-file-name "gpu-worker.py" wime-neural--directory)
  "Local worker executable, by default this package's gpu-worker.py."
  :type 'string :group 'wime)

(defcustom wime-neural-model-file
  (expand-file-name "wime-syllable-40m/model.safetensors" wime-data-directory)
  "Syllable transducer (wime-syllable-transducer-v1) in safetensors format."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-neural-interpreter (expand-file-name "python/bin/python" wime-data-directory)
  "Interpreter that runs `wime-neural-program', or nil to execute it directly.
The worker needs PyTorch with GPU support and transformers."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-neural-rescorer (expand-file-name "rescorers/Qwen3.5-2B-Base" wime-data-directory)
  "Directory of a general causal language model for the worker.
Its log probability, times `wime-neural-rescorer-weight', joins the
syllable model score.  Nil disables it."
  :type '(choice (const nil) directory) :group 'wime)

(defcustom wime-neural-rescorer-weight 1.5
  "Weight of `wime-neural-rescorer' relative to the syllable model score."
  :type 'number :group 'wime)

(defcustom wime-neural-memory-weight 10.0
  "Score bonus for decoded text covered by learned readings.
The bonus keeps corrections preferred for the same keys; see
`wime-user-database'."
  :type 'number :group 'wime)

(defcustom wime-neural-beam 32
  "Hypotheses the decoder keeps per character."
  :type 'natnum :group 'wime)

(defcustom wime-neural-delay 0.04
  "Seconds without a new input key before sending a neural request."
  :type 'number :group 'wime)

(defcustom wime-neural-timeout 6.0
  "Seconds allowed for a response from an already running worker."
  :type 'number :group 'wime)

(defcustom wime-neural-startup-timeout 30.0
  "Seconds allowed for the first response, including model and GPU setup.
Traditional candidates remain usable during initialization.  Later requests
use `wime-neural-timeout'."
  :type 'number :group 'wime)

(defcustom wime-neural-single-character t
  "Let the model rank single-character compositions too.
When disabled, single-character input retains dictionary ranking."
  :type 'boolean :group 'wime)

(cl-defstruct (wime-neural--job (:constructor wime-neural--make-job))
  id buffer input position context records syllables pins method
  history committed literals
  edit-generation (memory-tick wime--memory-tick))

(declare-function wime-japanese--current-p "wime-japanese" (job))
(declare-function wime-japanese--request "wime-japanese" (job))
(declare-function wime-japanese--apply "wime-japanese" (job response))
(declare-function wime-edit--active-p "wime-edit" ())
(declare-function wime-edit--writable-p "wime-edit" ())
(declare-function wime-edit--decodable-p "wime-edit" ())
(declare-function wime-edit--literals "wime-edit" ())
(defvar wime-edit-mode)
(defvar wime-edit--generation)
(defvar wime-neural--process nil)
(defvar wime-neural--busy nil)
(defvar wime-neural--pending nil)
(defvar wime-neural--timeout-timer nil)
(defvar wime-neural--serial 0)
(defvar wime-neural--failed nil)
(defvar-local wime-neural--timer nil)
(defvar-local wime-neural--job nil
  "This buffer's latest scheduled decode, which a choice may wait for.")
(defvar-local wime-neural-last-latency nil
  "Milliseconds spent in the most recently accepted model evaluation.")

(defconst wime-neural-context-limit 1024
  "Characters of left context sent to the worker.
The language model reads all of them; the syllable model its last 64.")

(defvar-local wime-neural--prefix nil
  "This composition's `wime-context-prefix-function' text, taken once.
A chat timeline that changes while the user types must not void the decode.")

(defun wime-neural--prefix ()
  (or wime-neural--prefix (setq wime-neural--prefix (wime--context-prefix))))

(defun wime-neural--forget-prefix ()
  (setq wime-neural--prefix nil))

(defun wime-neural--context ()
  (concat (wime-neural--prefix) (wime--context wime-neural-context-limit)))

(defun wime-neural--state ()
  "The composition state, its history led by `wime-context-prefix-function' text."
  (let ((state (wime--composition-state wime-neural-context-limit)))
    (cons (cons 'context (concat (wime-neural--prefix) (alist-get 'context state)))
          (assq-delete-all 'context state))))

(defun wime-neural--valid-p (job)
  "Whether JOB still belongs to the current unselected composition."
  (and (buffer-live-p (wime-neural--job-buffer job))
       (with-current-buffer (wime-neural--job-buffer job)
         (and (or (eq (wime-neural--job-method job) 'japanese)
                  (= (wime-neural--job-memory-tick job) wime--memory-tick))
              (if (eq (wime-neural--job-method job) 'japanese)
                  (wime-japanese--current-p job)
                (and (wime--active-p) (= wime--selected 0)
                     (equal wime--input (wime-neural--job-input job))
                     (eq wime--candidate-records (wime-neural--job-records job))))
              (if (and (not (eq (wime-neural--job-method job) 'japanese))
                       (wime-neural--job-edit-generation job))
                  ;; Moving within an editable run changes neither its keys
                  ;; nor its context.  Re-adoption and edits do, even when
                  ;; they happen to restore identical text and candidates.
                  (and (wime--editing-p) (wime-edit--writable-p)
                       (= wime-edit--generation
                          (wime-neural--job-edit-generation job)))
                (= (point) (wime-neural--job-position job)))
              ;; The decode read only this context: edits after point or past
              ;; its window, such as a chat timeline's new rows, keep it valid.
              ;; Overlay properties and narrowing can change it without edits.
              (equal (wime-neural--context) (wime-neural--job-context job))
              (or (null (wime-neural--job-history job))
                  (let ((state (wime-neural--state)))
                    (and (equal (alist-get 'context state) (wime-neural--job-history job))
                         (equal (alist-get 'committed state) (wime-neural--job-committed job)))))))))

(defun wime-neural--reset ()
  "Cancel this buffer's scheduled work.  In-flight replies become stale."
  (when (timerp wime-neural--timer) (cancel-timer wime-neural--timer))
  (setq wime-neural--timer nil wime-neural--job nil)
  (when (and wime-neural--pending
             (eq (current-buffer) (wime-neural--job-buffer wime-neural--pending)))
    (setq wime-neural--pending nil)))

(defun wime-neural--features-p ()
  "Whether the database records the syllable features the model reads."
  (condition-case nil
      (equal (caar (sqlite-select (wime--database)
                                  "SELECT value FROM meta WHERE key = 'syllable_encoding'"))
             "wime-syllable-features-v1")
    (error nil)))

(defun wime-neural--model-p (units)
  "Whether `wime-decoding' gives UNITS to a usable model.
Single characters go to it only with `wime-neural-single-character'."
  (and (not (eq wime-decoding 'lattice)) (not wime-neural--failed)
       wime-neural-model-file wime-neural-rescorer
       (or (cdr units) wime-neural-single-character)
       (wime-neural--features-p)))

(defun wime-neural--exclusive-p (units)
  "Whether `neural' decoding leaves UNITS to the model, without the lattice.
An unfinished first syllable shows its keys too: the model decodes it once
complete."
  (and (eq wime-decoding 'neural) (wime-neural--model-p units)))

(defun wime-neural--schedule ()
  "Decode after typing pauses, keeping the dictionary usable immediately."
  (wime-neural--reset)
  (let ((units (wime--parse wime--input)))
    (when (and (= (length (caar units)) 2) (wime-neural--model-p units)
               (or (not (bound-and-true-p wime-edit-mode)) (wime-edit--decodable-p)))
      (let* ((state (wime-neural--state))
             (editing (wime--editing-p))
             (job (wime-neural--make-job
                   :id (cl-incf wime-neural--serial) :buffer (current-buffer)
                   :input wime--input :position (point-marker)
                   :edit-generation (and editing wime-edit--generation)
                   :context (wime-neural--context)
                   :history (alist-get 'context state) :committed (alist-get 'committed state)
                   :literals (and editing (wime-edit--literals))
                   :records wime--candidate-records :syllables (length units)
                   :method 'decode :pins wime--pins)))
        (setq wime-neural--job job
              wime-neural--timer
              (run-with-timer wime-neural-delay nil #'wime-neural--enqueue job))))))

(defun wime-neural--await ()
  "Under `neural' decoding, finish this composition's decode before a choice.
A scheduled request goes out at once.  If the worker has failed, even
between requests, the lattice converts instead of committing keys."
  (when-let* ((job (and (eq wime-decoding 'neural) (not wime--model-ranked) wime-neural--job)))
    (when (memq wime-neural--timer timer-list)
      (cancel-timer wime-neural--timer)
      (wime-neural--enqueue job))
    (while (and (not wime--model-ranked)
                (memq job (list wime-neural--busy wime-neural--pending)))
      (accept-process-output wime-neural--process 0.05))
    (when wime-neural--failed (wime--refresh))))

(defun wime-neural--stop ()
  "Stop the worker and clear global work without changing candidates."
  (when (timerp wime-neural--timeout-timer)
    (cancel-timer wime-neural--timeout-timer))
  (setq wime-neural--timeout-timer nil
        wime-neural--busy nil wime-neural--pending nil)
  (when (processp wime-neural--process)
    (set-process-sentinel wime-neural--process #'ignore)
    (delete-process wime-neural--process))
  (setq wime-neural--process nil))

(defun wime-neural--fail (reason &optional job)
  "Disable failed worker attempts, retaining traditional results, with REASON.
A nil REASON fails quietly.  Under `neural' decoding, JOB's composition (by
default the busy one's), which showed its keys while waiting, converts with
the lattice instead."
  (let ((job (or job wime-neural--busy)))
    (wime-neural--stop)
    (unless (or wime-neural--failed (null reason))
      (message "wime neural unavailable: %s (M-x wime-neural-restart retries)" reason))
    (setq wime-neural--failed t)
    (when (and job (eq wime-decoding 'neural) (buffer-live-p (wime-neural--job-buffer job)))
      (with-current-buffer (wime-neural--job-buffer job)
        (when (wime--active-p) (wime--refresh))))))

(defun wime-neural--sentinel (process _event)
  "Handle PROCESS exiting without logging any typed text."
  (when (and (eq process wime-neural--process) (not (process-live-p process)))
    (wime-neural--fail "worker exited")))

(defun wime-neural--command (model)
  "Worker command line for MODEL; the model stays the last argument."
  (append (and wime-neural-interpreter (list wime-neural-interpreter))
          (list (or (executable-find wime-neural-program)
                    (expand-file-name wime-neural-program)))
          (and wime-neural-rescorer
               (list "--rescorer" (expand-file-name wime-neural-rescorer)
                     "--rescorer-weight" (number-to-string wime-neural-rescorer-weight)))
          (and wime-neural-rescorer
               (list "--database" (expand-file-name wime-database-file)
                     "--beam" (number-to-string wime-neural-beam)))
          (and wime-neural-rescorer (bound-and-true-p wime-japanese-model-file)
               (bound-and-true-p wime-japanese-table-file)
               (append (list "--japanese-model" (expand-file-name wime-japanese-model-file)
                             "--japanese-table" (expand-file-name wime-japanese-table-file))
                       (and (bound-and-true-p wime-japanese-zenz-directory)
                            (list "--japanese-zenz" (expand-file-name wime-japanese-zenz-directory)))))
          (list (expand-file-name model))))

(defun wime-neural--start ()
  "Return a running local worker or signal a startup error."
  (unless (process-live-p wime-neural--process)
    (unless (and wime-neural-model-file (file-readable-p wime-neural-model-file))
      (error "Model file is not readable"))
    (let ((default-directory temporary-file-directory))
      (setq wime-neural--process
            (make-process :name "wime-neural" :buffer nil :noquery t
                          :connection-type 'pipe :coding 'utf-8-unix
                          :stderr " *wime-neural-errors*"
                          :command (wime-neural--command wime-neural-model-file)
                          :filter #'wime-neural--filter
                          :sentinel #'wime-neural--sentinel))))
  wime-neural--process)

(defun wime-neural--request (job)
  "Encode a Chinese or Japanese decoding JOB."
  (pcase (wime-neural--job-method job)
    ('decode (wime-neural--decode-request job))
    ('japanese (wime-japanese--request job))
    (_ (error "Unknown decoding method"))))

(defun wime-neural--decode-request (job)
  "Encode JOB's whole input, pins and learned readings for decoding."
  (let ((units (wime--parse (wime-neural--job-input job))))
    (json-serialize
     `(:id ,(wime-neural--job-id job) :operation "decode" :context ,(or (wime-neural--job-history job) (wime-neural--job-context job))
       :syllables ,(length units) :input ,(wime--full-code units)
       :encoding "wime-composition-v2"
       :committed ,(or (wime-neural--job-committed job) :null)
       :pins ,(vconcat (mapcar (lambda (pin) (vector (car pin) (string (nth 1 pin))))
                               (wime-neural--job-pins job)))
       :learned ,(vconcat (mapcar (lambda (span) (vector (car span) (cdr span)))
                                  (wime--memory-spans units)))
       :aux ,(vconcat (cl-loop for unit in units for i from 0
                               unless (string-empty-p (cdr unit))
                               collect (vector i (downcase (cdr unit)))))
       :literals ,(vconcat (mapcar (lambda (literal) (vector (car literal) (cdr literal)))
                                   (wime-neural--job-literals job)))
       :tones ,(vconcat (mapcar (lambda (tone) (vector (car tone) (cdr tone)))
                                (wime--tones (wime-neural--job-input job))))
       :memory_weight ,wime-neural-memory-weight))))

(defun wime-neural--installed-p ()
  "Whether the model, language model and interpreter are where the options say."
  (and wime-neural-model-file (file-readable-p wime-neural-model-file)
       (file-directory-p wime-neural-rescorer)
       (or (null wime-neural-interpreter) (file-executable-p wime-neural-interpreter))))

(defun wime-neural--enqueue (job)
  "Send JOB or replace pending work while the worker is busy.
Without the model files, the default `hybrid' decoding stays on the lattice
without a message."
  (when (and (not wime-neural--failed) (wime-neural--valid-p job))
    (cond
     (wime-neural--busy (setq wime-neural--pending job))
     ((not (or (process-live-p wime-neural--process) (wime-neural--installed-p)))
      (wime-neural--fail (unless (eq wime-decoding 'hybrid) "model files are missing") job))
     (t
      (condition-case nil
          (let ((process (wime-neural--start)))
            (setq wime-neural--busy job
                  wime-neural--timeout-timer
                  (run-with-timer (+ (if (process-get process 'wime-initialized)
                                         wime-neural-timeout wime-neural-startup-timeout)
                                     ;; Decoding from scratch, as after a new
                                     ;; composition's pause, costs per syllable.
                                     (* 0.1 (wime-neural--job-syllables job)))
                                  nil #'wime-neural--fail "worker timed out"))
            (process-send-string process (concat (wime-neural--request job) "\n")))
        (error (wime-neural--fail "worker could not start or receive input" job)))))))

(defun wime-neural--apply (job response)
  "Apply RESPONSE only if JOB is still current."
  (when (wime-neural--valid-p job)
    (with-current-buffer (wime-neural--job-buffer job)
      (setq wime-neural-last-latency (alist-get 'elapsed_ms response)
            wime--model-ranked t)
      (pcase (wime-neural--job-method job)
        ('decode (wime-neural--decode-apply job response))
        ('japanese (wime-japanese--apply job response))))))

(defun wime-neural--decode-apply (job response)
  "Show RESPONSE's readings of JOB's whole input ahead of the lattice's.
Each must cover every syllable, be Chinese and keep the pins; the lattice's
words and completions stay after them, so a prefix can still be committed."
  (let* ((n (wime-neural--job-syllables job))
         (decoded (cl-loop for pair across (or (alist-get 'candidates response) [])
                           for text = (and (vectorp pair) (= (length pair) 2) (aref pair 0))
                           when (and (stringp text) (= (length text) n) (numberp (aref pair 1))
                                     (string-match-p "\\`[㐀-鿿]+\\'" text)
                                     (wime--pins-allow-p text 0))
                           collect (wime--make-candidate :text text :score (aref pair 1) :end n)))
         (texts (mapcar #'wime-candidate-text decoded)))
    (when decoded
      (setq wime--candidate-scores (mapcar (lambda (c) (cons (wime-candidate-text c) (wime-candidate-score c)))
                                           decoded)
            wime--candidate-records
            (append decoded (seq-remove (lambda (c) (member (wime-candidate-text c) texts))
                                        (wime-neural--job-records job)))
            wime--candidates (seq-take (seq-uniq (append texts wime--candidates)) wime-candidates-max))
      (unless wime--pins
        (setq wime--default-reading (car wime--candidates)))
      (wime--display))))

(defun wime-neural--reply (response)
  "Complete the matching request with RESPONSE and dispatch latest pending work."
  (when (and wime-neural--busy
             (equal (alist-get 'id response) (wime-neural--job-id wime-neural--busy)))
    (when (processp wime-neural--process)
      (process-put wime-neural--process 'wime-initialized t))
    (when (timerp wime-neural--timeout-timer) (cancel-timer wime-neural--timeout-timer))
    (setq wime-neural--timeout-timer nil)
    (wime-neural--apply wime-neural--busy response)
    (setq wime-neural--busy nil)
    (when wime-neural--pending
      (let ((job wime-neural--pending))
        (setq wime-neural--pending nil)
        (wime-neural--enqueue job)))))

(defun wime-neural--filter (process chunk)
  "Decode possibly split JSON lines from PROCESS CHUNK."
  (when (eq process wime-neural--process)
    (let* ((data (concat (process-get process 'partial) chunk))
           (lines (split-string data "\n")))
      (process-put process 'partial (car (last lines)))
      (condition-case nil
          (dolist (line (butlast lines))
            (wime-neural--reply (json-parse-string line :object-type 'alist :null-object nil)))
        (error (wime-neural--fail "invalid worker response"))))))

(defun wime-neural-restart ()
  "Stop the worker and forget a failure; the next decode starts it again."
  (interactive)
  (dolist (buffer (buffer-list))
    (with-current-buffer buffer (wime-neural--reset)))
  (wime-neural--stop)
  (setq wime-neural--failed nil))

;; `wime-decoding' decides, at each composition, whether these do anything.
(add-hook 'wime-candidates-updated-hook #'wime-neural--schedule)
(add-hook 'wime-lattice-inhibit-functions #'wime-neural--exclusive-p)
(add-hook 'wime-settle-hook #'wime-neural--await)
(add-hook 'wime-reset-hook #'wime-neural--reset)
(add-hook 'wime-reset-hook #'wime-neural--forget-prefix)
(add-hook 'kill-buffer-hook #'wime-neural--reset)

(provide 'wime-neural)
;;; wime-neural.el ends here
