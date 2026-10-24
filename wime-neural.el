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

(defcustom wime-neural-delay 0.04
  "Seconds without a new input key before requesting neural reranking."
  :type 'number :group 'wime)

(defcustom wime-neural-timeout 2.0
  "Seconds allowed for a worker response, including initial model loading."
  :type 'number :group 'wime)

(defcustom wime-neural-search-method 'rerank
  "Use candidate reranking or conditional draft verification.
`verify' requires a wime-ms-pinyin-v1 model and a Microsoft double-pinyin
database.  Auxiliary codes continue to constrain the dictionary search."
  :type '(choice (const rerank) (const verify)) :group 'wime)

(defcustom wime-neural-inference-budget 4
  "Maximum verified drafts per composition in `verify' mode."
  :type 'integer :group 'wime)

(defcustom wime-neural-prior-weight 0.1
  "Weight of the traditional path score in conditional draft selection.
Both model and traditional scores are normalized by syllable count.  Zero
selects by the model alone.  Prefix proposals still come from the model;
this weight only selects among complete, verified drafts."
  :type 'number :group 'wime)

(cl-defstruct (wime-neural--job (:constructor wime-neural--make-job))
  id buffer input position tick context records syllables
  method draft best best-score (round 0) visited
  (prior-weight wime-neural-prior-weight))

(defvar wime-neural-mode)
(defvar wime-neural--process nil)
(defvar wime-neural--busy nil)
(defvar wime-neural--pending nil)
(defvar wime-neural--timeout-timer nil)
(defvar wime-neural--serial 0)
(defvar wime-neural--failed nil)
(defvar-local wime-neural--timer nil)
(defvar-local wime-neural-last-latency nil
  "Milliseconds spent in the most recently accepted model evaluation.")

(defun wime-neural--context ()
  "Return up to 64 characters before the insertion point."
  (buffer-substring-no-properties (max (point-min) (- (point) 64)) (point)))

(defun wime-neural--valid-p (job)
  "Whether JOB still belongs to the current unselected composition."
  (and wime-neural-mode (buffer-live-p (wime-neural--job-buffer job))
       (with-current-buffer (wime-neural--job-buffer job)
         (and (wime--active-p) (= wime--selected 0)
              (= (point) (wime-neural--job-position job))
              (= (buffer-chars-modified-tick) (wime-neural--job-tick job))
              (equal wime--input (wime-neural--job-input job))
              (eq wime--candidate-records (wime-neural--job-records job))))))

(defun wime-neural--reset ()
  "Cancel this buffer's scheduled work.  In-flight replies become stale."
  (when (timerp wime-neural--timer) (cancel-timer wime-neural--timer))
  (setq wime-neural--timer nil)
  (when (and wime-neural--pending
             (eq (current-buffer) (wime-neural--job-buffer wime-neural--pending)))
    (setq wime-neural--pending nil)))

(defun wime-neural--schedule ()
  "Schedule a bounded comparison after the traditional candidates appear."
  (wime-neural--reset)
  (when (and wime-neural-mode (not wime-neural--failed)
             wime-neural-model-file wime--candidate-records)
    (let* ((records wime--candidate-records)
           (n (length (wime--parse wime--input)))
           (full (seq-filter (lambda (c) (= n (wime-candidate-end c))) records)))
      (when (and full (eq (car full) (car records))
                 (if (eq wime-neural-search-method 'verify)
                     (condition-case nil
                         (equal (caar (sqlite-select (wime--database)
                                                    "SELECT value FROM meta WHERE key = 'scheme'"))
                                "mspy")
                       (error nil))
                   (and (> (length full) 1)
                        (not (wime-candidate-dictionary (car records))))))
        (let ((job (wime-neural--make-job
                    :id (cl-incf wime-neural--serial) :buffer (current-buffer)
                    :input wime--input :position (point)
                    :tick (buffer-chars-modified-tick) :context (wime-neural--context)
                    :records records :syllables n :method wime-neural-search-method
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

(defun wime-neural--start ()
  "Return a running local worker or signal a startup error."
  (unless (process-live-p wime-neural--process)
    (unless (and wime-neural-model-file (file-readable-p wime-neural-model-file))
      (error "Model file is not readable"))
    (let ((model (expand-file-name wime-neural-model-file))
          (program (or (executable-find wime-neural-program)
                       (expand-file-name wime-neural-program)))
          (default-directory temporary-file-directory))
      (setq wime-neural--process
            (make-process :name "wime-neural" :buffer nil :noquery t
                          :connection-type 'pipe :coding 'utf-8-unix
                          :stderr " *wime-neural-errors*"
                          :command (list program model)
                          :filter #'wime-neural--filter
                          :sentinel #'wime-neural--sentinel))))
  wime-neural--process)

(defun wime-neural--request (job)
  "Encode JOB with explicit coverage and dictionary provenance."
  (json-serialize
   `(:id ,(wime-neural--job-id job) :context ,(wime-neural--job-context job)
     :syllables ,(wime-neural--job-syllables job)
     :operation ,(if (eq (wime-neural--job-method job) 'verify) "verify" "rerank")
     :input ,(wime--full-code (wime--parse (wime-neural--job-input job)))
     :draft ,(if (wime-neural--job-draft job)
                 (wime-candidate-text (wime-neural--job-draft job)) "")
     :candidates ,(vconcat
                   (mapcar (lambda (c)
                             `(:text ,(wime-candidate-text c) :end ,(wime-candidate-end c)
                               :dictionary ,(if (wime-candidate-dictionary c) t :false)))
                           (wime-neural--job-records job))))))

(defun wime-neural--enqueue (job)
  "Send JOB or replace pending work while the worker is busy."
  (when (and (not wime-neural--failed) (wime-neural--valid-p job))
    (if wime-neural--busy
        (setq wime-neural--pending job)
      (condition-case nil
          (let ((process (wime-neural--start)))
            (setq wime-neural--busy job
                  wime-neural--timeout-timer
                  (run-with-timer wime-neural-timeout nil
                                  #'wime-neural--fail "worker timed out"))
            (process-send-string process (concat (wime-neural--request job) "\n")))
        (error (wime-neural--fail "worker could not start or receive input"))))))

(defun wime-neural--publish (job candidate)
  "Promote a complete CANDIDATE validated against JOB's input."
  (when (and candidate (= (wime-candidate-end candidate) (wime-neural--job-syllables job)))
    (setq wime--candidate-records
          (cons candidate (seq-remove
                           (lambda (c) (equal (wime-candidate-text c) (wime-candidate-text candidate)))
                           (wime-neural--job-records job)))
          wime--candidates (seq-take (mapcar #'wime-candidate-text wime--candidate-records)
                                    wime-candidates-max))
    (wime--display)))

(defun wime-neural--verify-apply (job response)
  "Score JOB's draft, then search the first viable model prefix in RESPONSE."
  (let* ((draft (wime-neural--job-draft job))
         (text (wime-candidate-text draft))
         (score (alist-get 'score response))
         (next nil))
    (when (numberp score)
      (setq score (+ score (* (wime-neural--job-prior-weight job)
                             (/ (wime-candidate-score draft)
                                (float (wime-neural--job-syllables job))))))
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
              (let ((candidate (car (wime--sentence-records
                                     (wime--parse (wime-neural--job-input job)) prefix))))
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
        (if (eq (wime-neural--job-method job) 'verify)
            (wime-neural--verify-apply job response)
          (when candidate (wime-neural--publish job candidate)))))))

(defun wime-neural--reply (response)
  "Complete the matching request with RESPONSE and dispatch latest pending work."
  (when (and wime-neural--busy
             (equal (alist-get 'id response) (wime-neural--job-id wime-neural--busy)))
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
