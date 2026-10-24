;;; record-conditional.el --- Exercise the actual asynchronous worker path -*- lexical-binding: t; -*-
(require 'wime-neural)
(require 'json)
(set-language-environment "UTF-8")
(setq coding-system-for-read 'utf-8-unix coding-system-for-write 'utf-8-unix
      wime-database-file (getenv "WIME_DATABASE")
      wime-grammar-module (getenv "WIME_GRAMMAR_MODULE")
      wime-grammar-shared-dir (getenv "WIME_GRAMMAR")
      wime-neural-program (getenv "WIME_NEURAL_PROGRAM")
      wime-neural-model-file (getenv "WIME_NEURAL_MODEL")
      wime-neural-search-method 'verify wime-neural-delay 0)
(when (getenv "WIME_SEARCH_WIDTH")
  (setq wime-search-width (string-to-number (getenv "WIME_SEARCH_WIDTH"))))
(when (getenv "WIME_INFERENCE_BUDGET")
  (setq wime-neural-inference-budget (string-to-number (getenv "WIME_INFERENCE_BUDGET"))))
(setq wime-neural-seed-candidates (equal (getenv "WIME_SEED_CANDIDATES") "1"))
(unless (file-readable-p (expand-file-name "wanxiang-lts-zh-hans.gram" wime-grammar-shared-dir))
  (error "Missing octagram artifact"))
(let ((cases (with-temp-buffer
               (insert-file-contents (getenv "WIME_CASES"))
               (json-parse-buffer :object-type 'alist :array-type 'list))) results)
  (unwind-protect
      (progn
        (wime-neural-mode 1)
        (dolist (case cases)
          (with-temp-buffer
            (insert (alist-get 'context case))
            (setq wime--input (alist-get 'input case))
            (let ((start (float-time)) (cold (not (process-live-p wime-neural--process)))
                  baseline initial-search-ms)
              (wime--refresh)
              (setq baseline (car wime--candidates)
                    initial-search-ms (* 1000 (- (float-time) start)))
              (while (and (< (- (float-time) start) 20)
                          (or (memq wime-neural--timer timer-list)
                              wime-neural--busy wime-neural--pending))
                ;; A 10 ms polling interval masks the latency of a warm GPU
                ;; request. Wait on the worker, with 1 ms for the initial timer.
                (accept-process-output wime-neural--process 0.001))
              (when (or wime-neural--failed wime-neural--busy wime-neural--pending)
                (error "Conditional worker did not complete"))
              (unless wime-neural-last-latency (error "Conditional model was not evaluated"))
              (push `(:id ,(alist-get 'id case) :input ,wime--input
                      :gold ,(vconcat (alist-get 'gold case))
                      :baseline ,baseline :result ,(car wime--candidates)
                      :cold_start ,(if cold t :false)
                      :initial_search_ms ,initial-search-ms
                      :last_inference_ms ,wime-neural-last-latency
                      :wall_ms ,(* 1000 (- (float-time) start))) results)
              (wime--reset))))
        (with-temp-file (getenv "WIME_RECORDING")
          (insert (json-serialize (vconcat (nreverse results))))))
    (wime-neural-mode -1)))
