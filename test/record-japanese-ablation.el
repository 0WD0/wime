;;; record-japanese-ablation.el --- Warm, instrumented real input path -*- lexical-binding: t; -*-

(setenv "WIME_RECORDING_HELPERS_ONLY" "1")
(load (expand-file-name "record-japanese.el" (file-name-directory (or load-file-name buffer-file-name))) nil t)
(setenv "WIME_RECORDING_HELPERS_ONLY" nil)

(defvar record-japanese-ablation--events nil)
(defvar record-japanese-ablation--sent (make-hash-table :test #'eql))

(defun record-japanese-ablation--request (encoded)
  (let* ((request (json-parse-string encoded :object-type 'alist))
         (id (alist-get 'id request)))
    (puthash id (float-time) record-japanese-ablation--sent)
    (push `(:event "sent" :id ,id :kana ,(alist-get 'kana request)
            :pending ,(or (alist-get 'pending request) [])) record-japanese-ablation--events))
  encoded)

(defun record-japanese-ablation--reply (response)
  (let* ((id (alist-get 'id response))
         (sent (gethash id record-japanese-ablation--sent)))
    (when sent
      (push `(:event "reply" :id ,id :wall_ms ,(* 1000 (- (float-time) sent))
              :worker_ms ,(alist-get 'elapsed_ms response)
              :applied ,(if (and wime-neural--busy (wime-neural--valid-p wime-neural--busy)) t :false))
            record-japanese-ablation--events))))

(defun record-japanese-ablation--key (function key)
  (let ((start (float-time)))
    (prog1 (funcall function key)
      (push `(:event "key" :handler_ms ,(* 1000 (- (float-time) start)))
            record-japanese-ablation--events))))

;; Load models and kernels through exactly the same protocol, outside measurements.
(wime-neural-mode 1)
(with-temp-buffer
  (mapc #'wime-japanese-input-method (string-to-list (record-japanese--keys "にほんご")))
  (record-japanese--wait (+ (float-time) 120))
  (when (or wime-neural--failed wime-neural--busy wime-neural--pending)
    (error "Ablation worker warmup failed"))
  (wime-japanese--reset))

(let ((warmup-id wime-neural--serial))
  (advice-add 'wime-neural--request :filter-return #'record-japanese-ablation--request)
  (advice-add 'wime-neural--reply :before #'record-japanese-ablation--reply)
  (advice-add 'wime-japanese-input-method :around #'record-japanese-ablation--key)
  (unwind-protect
      (record-japanese--run)
    (advice-remove 'wime-neural--request #'record-japanese-ablation--request)
    (advice-remove 'wime-neural--reply #'record-japanese-ablation--reply)
    (advice-remove 'wime-japanese-input-method #'record-japanese-ablation--key)
    (with-temp-file (getenv "WIME_ABLATION_EVENTS")
      (insert (json-serialize `(:warmup_last_id ,warmup-id
                               :events ,(vconcat (nreverse record-japanese-ablation--events))))))))
