;;; record-commits.el --- Verify real commits retain their typed readings -*- lexical-binding: t; -*-
(require 'wime-neural)
(require 'json)
(set-language-environment "UTF-8")
(setq coding-system-for-read 'utf-8-unix coding-system-for-write 'utf-8-unix
      wime-database-file (getenv "WIME_DATABASE")
      wime-grammar-module (getenv "WIME_GRAMMAR_MODULE")
      wime-grammar-shared-dir (getenv "WIME_GRAMMAR")
      wime-neural-program (getenv "WIME_NEURAL_PROGRAM")
      wime-neural-model-file (getenv "WIME_NEURAL_MODEL")
      wime-neural-search-method 'verify wime-neural-delay 0
      wime-search-width 16 wime-neural-inference-budget 4
      wime-neural-seed-candidates t wime-neural-single-character t
      wime-neural-context-anchor 2)
(let ((cases (with-temp-buffer
               (insert-file-contents (getenv "WIME_CASES"))
               (json-parse-buffer :object-type 'alist :array-type 'list))) results)
  (unwind-protect
      (dolist (case cases)
        (with-temp-buffer
          (insert (alist-get 'context case))
          ;; Select a dictionary-valid prefix, as a user can explicitly do.
          (setq wime--input (alist-get 'committed_input case)
                wime--candidate-records
                (wime--sentence-records (wime--parse wime--input) (alist-get 'committed_text case)))
          (unless (seq-find (lambda (c) (equal (wime-candidate-text c) (alist-get 'committed_text case)))
                            wime--candidate-records)
            (error "Committed prefix is not a valid dictionary reading"))
          (wime--commit (alist-get 'committed_text case))
          (let ((known (wime--context-reading (wime--context))))
            (unless (equal (alist-get 'input known) (alist-get 'committed_input case))
              (error "Committed keys were lost or reinterpreted"))
            (wime-neural-mode 1)
            (mapc #'wime-input-method (string-to-list (alist-get 'input case)))
            (let ((deadline (+ (float-time) 20)))
              (while (and (< (float-time) deadline)
                          (or (memq wime-neural--timer timer-list) wime-neural--busy wime-neural--pending))
                (accept-process-output wime-neural--process .001)))
            (when (or wime-neural--failed wime-neural--busy wime-neural--pending
                      (not wime-neural-last-latency))
              (error "Model did not finish after a real commit"))
            (push `(:id ,(alist-get 'id case) :committed_reading ,known
                    :input ,(alist-get 'input case) :gold ,(vconcat (alist-get 'gold case))
                    :result ,(car wime--candidates)) results)
            (wime-neural-mode -1)
            (wime--reset))))
    (wime-neural-mode -1))
  (with-temp-file (getenv "WIME_RECORDING")
    (insert (json-serialize (vconcat (nreverse results))))))
