;;; record-japanese.el --- Type Japanese test sentences in AZIK through the real worker -*- lexical-binding: t; -*-
;; Environment: WIME_NEURAL_PROGRAM, WIME_NEURAL_INTERPRETER, WIME_NEURAL_MODEL,
;; WIME_RESCORER, WIME_JAPANESE_MODEL, WIME_JAPANESE_TABLE, WIME_CASES
;; (prepare-japanese.py's test.json), WIME_RECORDING, optional
;; WIME_TYPING_INTERVAL (seconds between keys), WIME_LIMIT and WIME_EXPLICIT
;; (hold Shift over the kana the cases' shift field marks).
(require 'wime-japanese)
(require 'json)
(set-language-environment "UTF-8")
(setq coding-system-for-read 'utf-8-unix coding-system-for-write 'utf-8-unix
      wime-neural-program (getenv "WIME_NEURAL_PROGRAM")
      wime-neural-interpreter (getenv "WIME_NEURAL_INTERPRETER")
      wime-neural-model-file (getenv "WIME_NEURAL_MODEL")
      wime-neural-rescorer (getenv "WIME_RESCORER")
      wime-japanese-model-file (getenv "WIME_JAPANESE_MODEL")
      wime-japanese-table-file (getenv "WIME_JAPANESE_TABLE")
      wime-neural-search-method 'verify wime-neural-delay 0 wime-neural-startup-timeout 120)

(defun record-japanese--shift (key)
  "KEY as typed with Shift held, or nil when Shift would type another rule."
  (cond ((and (>= key ?a) (<= key ?z)) (upcase key))
        ((car (rassq key wime-japanese-shifted-keys)))))

(defun record-japanese--keys (kana &optional shift)
  "Fewest AZIK keys spelling KANA, Shift held where SHIFT (a string of 0/1
per kana) says, or nil when no key sequence converts back to exactly that."
  (let* ((rules (car (wime-japanese--rules))) (n (length kana))
         (best (make-vector (1+ n) nil)))
    (aset best 0 "")
    (dotimes (i n)
      (when (aref best i)
        (maphash
         (lambda (keys rule)
           (let* ((out (cdr rule)) (j (+ i (length out)))
                  (flags (and shift (<= j n) (substring shift i j))))
             (when (and (string-empty-p (car rule)) (not (string-empty-p out)) (<= j n)
                        (string= out (substring kana i j))
                        (or (null (aref best j))
                            (< (+ (length (aref best i)) (length keys)) (length (aref best j)))))
               ;; The first kana follows the first key, the rest the last key.
               (let* ((first (and flags (eq (aref flags 0) ?1)))
                      (last (and flags (eq (aref flags (1- (length flags))) ?1)))
                      (typed (and (or (= (length flags) 1) (null flags)
                                      (string= (substring flags 1) (make-string (1- (length flags)) (aref flags (1- (length flags))))))
                                  (or (> (length keys) 1) (eq first last))
                                  (concat (cl-loop for key across keys for k from 0
                                                   collect (if (if (= k (1- (length keys))) (if (= (length keys) 1) first last) first)
                                                               (or (record-japanese--shift key) ?\0)
                                                             key))))))
                 (when (and typed (not (seq-contains-p typed ?\0)))
                   (aset best j (concat (aref best i) typed)))))))
         rules)))
    (let* ((keys (aref best n)) (converted (and keys (wime-japanese--convert keys))))
      (and keys (equal (substring-no-properties (car converted)) kana) (string-empty-p (cdr converted))
           (or (null shift)
               (string= shift (mapconcat (lambda (i) (if (get-text-property i 'wime-japanese-shift (car converted)) "1" "0"))
                                         (number-sequence 0 (1- n)) "")))
           keys))))

(defun record-japanese--wait (deadline)
  (while (and (< (float-time) deadline)
              (or (memq wime-neural--timer timer-list) wime-neural--busy wime-neural--pending))
    (accept-process-output wime-neural--process 0.001)))

(let* ((cases (with-temp-buffer
                (insert-file-contents (getenv "WIME_CASES"))
                (json-parse-buffer :object-type 'alist :array-type 'list)))
       (limit (string-to-number (or (getenv "WIME_LIMIT") "100000")))
       (interval (and (getenv "WIME_TYPING_INTERVAL") (string-to-number (getenv "WIME_TYPING_INTERVAL"))))
       (skipped 0) results)
  (unwind-protect
      (progn
        (wime-neural-mode 1)
        (dolist (case (seq-take cases limit))
          (let ((keys (record-japanese--keys (alist-get 'input case)
                                             (and (getenv "WIME_EXPLICIT") (alist-get 'shift case)))))
            (if (not keys)
                (cl-incf skipped)
              (with-temp-buffer
                (let (start)
                  (dolist (key (string-to-list (substring keys 0 -1)))
                    (wime-japanese-input-method key)
                    (when interval
                      (let ((until (+ (float-time) interval)))
                        (while (< (float-time) until)
                          (accept-process-output wime-neural--process 0.001)))))
                  (setq start (float-time))
                  (wime-japanese-input-method (aref keys (1- (length keys))))
                  (record-japanese--wait (+ start 60))
                  (when (or wime-neural--failed wime-neural--busy wime-neural--pending)
                    (error "Japanese worker did not complete"))
                  (push `(:id ,(alist-get 'id case) :keys ,keys :kana ,(alist-get 'input case)
                          :gold ,(alist-get 'gold case) :result ,(car wime-japanese--candidates)
                          :candidates ,(vconcat (seq-take wime-japanese--candidates 5))
                          :worker_ms ,(or wime-neural-last-latency :null)
                          :wall_ms ,(* 1000 (- (float-time) start)))
                        results)
                  (wime-japanese--reset))))))
        (with-temp-file (getenv "WIME_RECORDING")
          (insert (json-serialize `(:skipped ,skipped :results ,(vconcat (nreverse results)))))))
    (wime-neural-mode -1)))
