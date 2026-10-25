;;; wime-neural.el --- Local asynchronous candidate reranking -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "29.1"))

;;; Commentary:
;; A persistent worker loads the character model once.  There is at most one
;; running request and one pending request; typing replaces pending work.
;; Conditional verification can ask the dictionary lattice for new prefixes;
;; only complete, verified readings in the same composition may be promoted.

;;; Code:

(require 'cl-lib)
(require 'json)
(require 'wime)

(defcustom wime-neural-program "wime-neural"
  "Local reranking worker executable."
  :type 'string :group 'wime)

(defcustom wime-neural-model-file nil
  "Local safetensors model, downloaded separately at build time."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-neural-interpreter nil
  "Interpreter that runs `wime-neural-program', or nil to execute it directly.
The CUDA worker's rescorer needs a Python with a recent transformers."
  :type '(choice (const nil) file) :group 'wime)

(defcustom wime-neural-rescorer nil
  "Directory of a general causal language model for the CUDA worker.
Its mean per-character log probability, times `wime-neural-rescorer-weight',
is added to the model score of every verified candidate.  Nil disables it."
  :type '(choice (const nil) directory) :group 'wime)

(defcustom wime-neural-rescorer-weight 1.5
  "Weight of `wime-neural-rescorer' relative to the conditional model score."
  :type 'number :group 'wime)

(defcustom wime-neural-pool-width nil
  "Lattice width of the verify pool and prefix searches, or nil for the display.
Typing keeps `wime-search-width'; this wider search runs once per pause, when
the request is sent, with its own incremental cache."
  :type '(choice (const nil) integer) :group 'wime)

(defcustom wime-neural-memory-weight 10.0
  "Score bonus for a verified reading fully covered by learned readings.
Partial coverage earns the covered fraction of syllables.  The bonus joins
CUDA seed selection and draft verification, so the models cannot outvote a
correction the user made for the same keys; see `wime-user-database'."
  :type 'number :group 'wime)

(defcustom wime-neural-delay 0.04
  "Seconds without a new input key before requesting neural reranking."
  :type 'number :group 'wime)

(defcustom wime-neural-timeout 2.0
  "Seconds allowed for a response from an already running worker."
  :type 'number :group 'wime)

(defcustom wime-neural-startup-timeout 10.0
  "Seconds allowed for the first response, including model and CUDA setup.
Traditional candidates remain usable during initialization.  Later requests
use `wime-neural-timeout'."
  :type 'number :group 'wime)

(defcustom wime-neural-search-method 'rerank
  "Use candidate reranking, conditional draft verification or decoding.
`verify' requires a wime-ms-pinyin-v1 model and a Microsoft double-pinyin
database.  Auxiliary codes continue to constrain the dictionary search.
`decode' also needs `wime-neural-rescorer': the CUDA worker spells the keys
with the rescorer's tokens, fused with the model, without the lattice, one
syllable at a time and including a trailing half syllable.  Input with
auxiliary codes, or without a rescorer, is verified instead."
  :type '(choice (const rerank) (const verify) (const decode)) :group 'wime)

(defcustom wime-neural-single-character nil
  "Allow single-character compositions in conditional `verify' mode.
Enable only with a model trained and evaluated on single-character spans.
Older models retain contextual octagram ranking by default."
  :type 'boolean :group 'wime)

(defcustom wime-neural-context-anchor 0
  "Committed characters retained as a fixed decoder prefix by the CUDA worker.
Zero uses the original context/composition separation.  Actual recently
committed keys override inferred context readings; only new text is scored."
  :type 'integer :group 'wime)

(defcustom wime-neural-inference-budget 4
  "Maximum verification rounds per composition in `verify' mode.
The optional CUDA seed selection scores the initial candidate pool in one
round; subsequent rounds each verify one prefix-constrained draft."
  :type 'integer :group 'wime)

(defcustom wime-neural-seed-candidates nil
  "Let the CUDA worker score the initial candidate pool to choose a draft.
This avoids relying on single-character corrections to reach a better word.
Requires gpu-worker.py; the Rust worker retains its traditional draft."
  :type 'boolean :group 'wime)

(defcustom wime-neural-prior-weight 0.1
  "Weight of the traditional path score in conditional draft selection.
Both model and traditional scores are normalized by syllable count.  Zero
selects by the model alone.  Prefix proposals still come from the model;
this weight selects among complete, verified drafts, including CUDA seeds."
  :type 'number :group 'wime)

(defcustom wime-neural-prior-confidence 1.0
  "Full-candidate model probability that disables traditional score mixing.
One disables this optional CUDA seed-selection rule.  Lower values must be
calibrated for the model.  The selected weight stays fixed for all rounds
of one composition; dictionary and auxiliary constraints remain mandatory."
  :type 'number :group 'wime)

(cl-defstruct (wime-neural--job (:constructor wime-neural--make-job))
  id buffer input position tick context records pool syllables pins
  method draft best best-score (round 0) visited scored
  context-reading (context-anchor wime-neural-context-anchor)
  (seed-candidates wime-neural-seed-candidates)
  (prior-confidence wime-neural-prior-confidence)
  (prior-weight wime-neural-prior-weight))

(defvar wime-neural-mode)
(defvar wime-neural--process nil)
(defvar wime-neural--busy nil)
(defvar wime-neural--pending nil)
(defvar wime-neural--timeout-timer nil)
(defvar wime-neural--serial 0)
(defvar wime-neural--failed nil)
(defvar-local wime-neural--timer nil)
(defvar-local wime-neural--pool-cache nil
  "`wime--search-cache' of the wider pool search, apart from typing's.")
(defvar-local wime-neural-last-latency nil
  "Milliseconds spent in the most recently accepted model evaluation.")

(defalias 'wime-neural--context #'wime--context)

(defun wime-neural--valid-p (job)
  "Whether JOB still belongs to the current unselected composition."
  (and wime-neural-mode (buffer-live-p (wime-neural--job-buffer job))
       (with-current-buffer (wime-neural--job-buffer job)
         (and (wime--active-p) (= wime--selected 0)
              (= (point) (wime-neural--job-position job))
              (= (buffer-chars-modified-tick) (wime-neural--job-tick job))
              (equal wime--input (wime-neural--job-input job))
              (eq wime--candidate-records (wime-neural--job-records job))
              ;; Overlay properties and narrowing can change the context
              ;; without advancing buffer-chars-modified-tick.
              (equal (wime-neural--context) (wime-neural--job-context job))))))

(defun wime-neural--reset ()
  "Cancel this buffer's scheduled work.  In-flight replies become stale."
  (when (timerp wime-neural--timer) (cancel-timer wime-neural--timer))
  (setq wime-neural--timer nil)
  (when (and wime-neural--pending
             (eq (current-buffer) (wime-neural--job-buffer wime-neural--pending)))
    (setq wime-neural--pending nil)))

(defun wime-neural--decodable-p (units)
  "Whether `decode' reads UNITS: plain keys, one complete syllable or more."
  (and wime-neural-rescorer
       (cl-every (lambda (unit) (string-empty-p (cdr unit))) units)
       (= (length (caar units)) 2)
       (or (> (length units) 1) wime-neural-single-character)))

(defun wime-neural--schedule ()
  "Schedule a bounded comparison after the traditional candidates appear."
  (wime-neural--reset)
  (when (and wime-neural-mode (not wime-neural--failed) wime-neural-model-file)
    (let* ((records wime--candidate-records)
           (units (wime--parse wime--input))
           (n (length units))
           (method (if (eq wime-neural-search-method 'decode)
                       (if (wime-neural--decodable-p units) 'decode 'verify)
                     wime-neural-search-method))
           (full (seq-filter (lambda (c) (= n (wime-candidate-end c))) records)))
      (when (and (or (eq method 'decode)
                     (and (or (> n 1) (and (= n 1) wime-neural-single-character (eq method 'verify)))
                          full (eq (car full) (car records))
                          (or (eq method 'verify)
                              (and (> (length full) 1)
                                   (not (wime-candidate-dictionary (car records)))))))
                 (or (eq method 'rerank)
                     (condition-case nil
                         (equal (caar (sqlite-select (wime--database)
                                                    "SELECT value FROM meta WHERE key = 'scheme'"))
                                "mspy")
                       (error nil))))
        (let ((job (wime-neural--make-job
                    :id (cl-incf wime-neural--serial) :buffer (current-buffer)
                    :input wime--input :position (point)
                    :tick (buffer-chars-modified-tick) :context (wime-neural--context)
                    :context-reading (wime--context-reading (wime-neural--context))
                    :records records :syllables n :method method :pins wime--pins
                    :draft (car full))))
          (setq wime-neural--timer
                (run-with-timer wime-neural-delay nil #'wime-neural--enqueue job)))))))

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

(defun wime-neural--fail (reason)
  "Disable failed worker attempts, retaining traditional results, with REASON."
  (wime-neural--stop)
  (unless wime-neural--failed
    (message "wime neural unavailable: %s (toggle wime-neural-mode to retry)" reason))
  (setq wime-neural--failed t))

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
          (and wime-neural-rescorer (eq wime-neural-search-method 'decode) wime-database-file
               (list "--database" (expand-file-name wime-database-file)))
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
  "Encode JOB with explicit coverage and dictionary provenance."
  (if (eq (wime-neural--job-method job) 'decode)
      (wime-neural--decode-request job)
    (wime-neural--verify-request job)))

(defun wime-neural--decode-request (job)
  "Encode JOB's whole input, pins and learned readings for decoding."
  (let ((units (wime--parse (wime-neural--job-input job))))
    (json-serialize
     `(:id ,(wime-neural--job-id job) :operation "decode" :context ,(wime-neural--job-context job)
       :syllables ,(length units) :input ,(wime--full-code units)
       :context_anchor ,(wime-neural--job-context-anchor job)
       :context_reading ,(or (wime-neural--job-context-reading job) :null)
       :pins ,(vconcat (mapcar (lambda (pin) (vector (car pin) (string (nth 1 pin))))
                               (wime-neural--job-pins job)))
       :learned ,(vconcat (mapcar (lambda (span) (vector (car span) (cdr span)))
                                  (wime--memory-spans units)))
       :memory_weight ,wime-neural-memory-weight :candidates []))))

(defun wime-neural--verify-request (job)
  "Encode JOB's candidates and draft for verification or reranking."
  (json-serialize
   `(:id ,(wime-neural--job-id job) :context ,(wime-neural--job-context job)
     :syllables ,(wime-neural--job-syllables job)
     :operation ,(if (eq (wime-neural--job-method job) 'verify) "verify" "rerank")
     :input ,(wime--full-code (wime--parse (wime-neural--job-input job)))
     :draft ,(if (wime-neural--job-draft job)
                 (wime-candidate-text (wime-neural--job-draft job)) "")
     :seed_candidates ,(if (and (wime-neural--job-seed-candidates job)
                               (= (wime-neural--job-round job) 0)) t :false)
     :prior_weight ,(wime-neural--job-prior-weight job)
     :prior_confidence ,(wime-neural--job-prior-confidence job)
     :context_anchor ,(wime-neural--job-context-anchor job)
     :context_reading ,(or (wime-neural--job-context-reading job) :null)
     :candidates ,(vconcat
                   (mapcar (lambda (c)
                             `(:text ,(wime-candidate-text c) :end ,(wime-candidate-end c)
                               :score ,(wime-candidate-score c)
                               :bonus ,(wime-neural--bonus job (wime-candidate-text c))
                               :dictionary ,(if (wime-candidate-dictionary c) t :false)))
                           (wime-neural--candidates job))))))

(defun wime-neural--bonus (job text)
  "Learned-reading bonus of TEXT for JOB's keys."
  (* wime-neural-memory-weight
     (wime--memory-coverage text (wime--parse (wime-neural--job-input job)))))

(defun wime-neural--candidates (job)
  "Readings sent with JOB: its wider pool once searched, else the display's."
  (or (wime-neural--job-pool job) (wime-neural--job-records job)))

(defun wime-neural--pool (job)
  "Complete and shorter readings JOB verifies, at `wime-neural-pool-width'."
  (if (and wime-neural-pool-width (> wime-neural-pool-width wime-search-width)
           (> (wime-neural--job-syllables job) 1))
      (with-current-buffer (wime-neural--job-buffer job)
        (let ((wime-search-width wime-neural-pool-width)
              (wime--search-cache wime-neural--pool-cache))
          (prog1 (wime--sentence-records (wime--parse (wime-neural--job-input job)))
            (setq wime-neural--pool-cache wime--search-cache))))
    (wime-neural--job-records job)))

(defun wime-neural--enqueue (job)
  "Send JOB or replace pending work while the worker is busy."
  (when (and (not wime-neural--failed) (wime-neural--valid-p job))
    (if wime-neural--busy
        (setq wime-neural--pending job)
      (unless (or (wime-neural--job-pool job) (eq (wime-neural--job-method job) 'decode))
        (setf (wime-neural--job-pool job) (wime-neural--pool job)))
      (condition-case nil
          (let ((process (wime-neural--start)))
            (setq wime-neural--busy job
                  wime-neural--timeout-timer
                  (run-with-timer (+ (if (process-get process 'wime-initialized)
                                         wime-neural-timeout wime-neural-startup-timeout)
                                     ;; Decoding from scratch, as after a new
                                     ;; composition's pause, costs per syllable.
                                     (if (eq (wime-neural--job-method job) 'decode)
                                         (* 0.1 (wime-neural--job-syllables job)) 0))
                                  nil #'wime-neural--fail "worker timed out"))
            (process-send-string process (concat (wime-neural--request job) "\n")))
        (error (wime-neural--fail "worker could not start or receive input"))))))

(defun wime-neural--publish (job candidate)
  "Promote a complete CANDIDATE validated against JOB's input."
  (when (and candidate (= (wime-candidate-end candidate) (wime-neural--job-syllables job)))
    (setq wime--candidate-scores (wime-neural--job-scored job)
          wime--candidate-records
          (cons candidate (seq-remove
                           (lambda (c) (equal (wime-candidate-text c) (wime-candidate-text candidate)))
                           (wime-neural--job-records job)))
          wime--candidates (seq-take (mapcar #'wime-candidate-text wime--candidate-records)
                                    wime-candidates-max))
    (unless wime--pins
      (setq wime--default-reading (car wime--candidates)))
    (wime--display)))

(defun wime-neural--verify-apply (job response)
  "Score JOB's draft, then search the first viable model prefix in RESPONSE."
  (let* ((index (alist-get 'verified_index response))
         (seed (and (integerp index) (>= index 0)
                    (wime-neural--job-seed-candidates job)
                    (= (wime-neural--job-round job) 0)
                    (< index (length (wime-neural--candidates job)))
                    (nth index (wime-neural--candidates job))))
         (draft (or seed (wime-neural--job-draft job)))
         (text (wime-candidate-text draft))
         (score (alist-get 'score response))
         (next nil))
    (when (and (numberp score)
               (or (not (assq 'verified_index response))
                   (and seed (= (wime-candidate-end seed) (wime-neural--job-syllables job)))))
      ;; Only the initial, validated seed batch may disable the configured
      ;; weight.  Keep it on the job so later drafts use comparable scores.
      (when (and seed (= (wime-neural--job-round job) 0)
                 (< (wime-neural--job-prior-confidence job) 1)
                 (numberp (alist-get 'effective_prior_weight response))
                 (= (alist-get 'effective_prior_weight response) 0))
        (setf (wime-neural--job-prior-weight job) 0.0))
      (let ((weight (wime-neural--job-prior-weight job))
            (n (float (wime-neural--job-syllables job)))
            ;; Worker indices refer to the readings sent: the wider pool.
            (records (wime-neural--candidates job)))
        (setq score (+ score (* weight (/ (wime-candidate-score draft) n))
                       (wime-neural--bonus job text)))
        ;; The same joint score ranks verified drafts and estimates confidence.
        (when seed
          (seq-doseq (pair (alist-get 'candidate_scores response))
            (let ((record (and (integerp (aref pair 0)) (nth (aref pair 0) records))))
              (when (and record (numberp (aref pair 1)))
                (push (cons (wime-candidate-text record)
                            (+ (aref pair 1) (* weight (/ (wime-candidate-score record) n))
                               (wime-neural--bonus job (wime-candidate-text record))))
                      (wime-neural--job-scored job))))))
        (push (cons text score) (wime-neural--job-scored job)))
      (cl-incf (wime-neural--job-round job))
      (push text (wime-neural--job-visited job))
      (when (or (null (wime-neural--job-best-score job))
                (> score (wime-neural--job-best-score job)))
        (setf (wime-neural--job-best job) draft
              (wime-neural--job-best-score job) score))
      (when (and (< (wime-neural--job-round job) (max 1 wime-neural-inference-budget))
                 (not wime-neural--pending))
        (catch 'found
          (seq-doseq (prefix (seq-take (alist-get 'prefixes response) 16))
            (when (and (stringp prefix) (> (length prefix) 0)
                       (<= (length prefix) (wime-neural--job-syllables job)))
              (let ((candidate (car (let ((wime-search-width
                                           (max wime-search-width (or wime-neural-pool-width 0))))
                                      (wime--sentence-records
                                       (wime--parse (wime-neural--job-input job)) prefix)))))
                (when (and candidate
                           (= (wime-candidate-end candidate) (wime-neural--job-syllables job))
                           (not (member (wime-candidate-text candidate) (wime-neural--job-visited job))))
                  (setq next candidate)
                  (throw 'found t))))))))
    (if next
        (let ((continuation (copy-wime-neural--job job)))
          (setf (wime-neural--job-id continuation) (cl-incf wime-neural--serial)
                (wime-neural--job-draft continuation) next)
          (setq wime-neural--pending continuation))
      ;; The last proposed draft has not been verified and is never displayed.
      (wime-neural--publish job (wime-neural--job-best job)))))

(defun wime-neural--apply (job response)
  "Apply RESPONSE only if JOB is current and its promoted reading covers input."
  (when (wime-neural--valid-p job)
    (with-current-buffer (wime-neural--job-buffer job)
      (let* ((index (alist-get 'promoted response))
             (records (wime-neural--job-records job))
             (candidate (and (integerp index) (> index 0) (< index (length records))
                             (nth index records))))
        (setq wime-neural-last-latency (alist-get 'elapsed_ms response))
        (pcase (wime-neural--job-method job)
          ('decode (wime-neural--decode-apply job response))
          ('verify (wime-neural--verify-apply job response))
          (_ (when candidate (wime-neural--publish job candidate))))))))

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

;;;###autoload
(define-minor-mode wime-neural-mode
  "Rerank wime sentence candidates locally after a short typing pause."
  :global t :group 'wime
  (if wime-neural-mode
      (progn
        (setq wime-neural--failed nil)
        (add-hook 'wime-candidates-updated-hook #'wime-neural--schedule)
        (add-hook 'wime-reset-hook #'wime-neural--reset)
        (add-hook 'kill-buffer-hook #'wime-neural--reset))
    (remove-hook 'wime-candidates-updated-hook #'wime-neural--schedule)
    (remove-hook 'wime-reset-hook #'wime-neural--reset)
    (remove-hook 'kill-buffer-hook #'wime-neural--reset)
    (dolist (buffer (buffer-list))
      (with-current-buffer buffer (wime-neural--reset)))
    (wime-neural--stop)))

(provide 'wime-neural)
;;; wime-neural.el ends here
