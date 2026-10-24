;;; record-neural.el --- Record actual per-keystroke candidates -*- lexical-binding: t; -*-
(require 'json)
(require 'wime)
(set-language-environment "UTF-8")
(setq coding-system-for-read 'utf-8-unix
      coding-system-for-write 'utf-8-unix)

(setq wime-database-file (getenv "WIME_DATABASE"))
(when-let* ((grammar (getenv "WIME_GRAMMAR")))
  (setq wime-grammar-shared-dir grammar
        wime-grammar-module (getenv "WIME_GRAMMAR_MODULE")))
(when-let* ((width (getenv "WIME_WIDTH")))
  (setq wime-search-width (string-to-number width)))

(let ((cases (with-temp-buffer
               (insert-file-contents (getenv "WIME_CASES"))
               (json-parse-buffer :object-type 'alist :array-type 'list)))
      (serial 0))
  (with-temp-file (getenv "WIME_RECORDING")
    (dolist (case cases)
      (let ((context (alist-get 'context case)) (input (alist-get 'input case)))
        (cl-loop for end from 1 to (length input) do
                 (let* ((typed (substring input 0 end))
                        (start (float-time))
                        (strings (wime--candidates typed))
                        (elapsed (* 1000.0 (- (float-time) start)))
                        (n (length (wime--parse typed)))
                        (records (and (boundp 'wime--candidate-records)
                                      wime--candidate-records)))
                   (insert
                    (json-serialize
                     `(:id ,(cl-incf serial) :case ,(alist-get 'id case)
                       :final ,(if (= end (length input)) t :false)
                       :context ,context :syllables ,n
                       :input ,typed :search_ms ,elapsed
                       :leader ,(or (car strings) "")
                       :gold ,(vconcat (alist-get 'gold case))
                       :candidates ,(vconcat
                                     (mapcar
                                      (lambda (c)
                                        `(:text ,(wime-candidate-text c)
                                          :end ,(wime-candidate-end c)
                                          :dictionary ,(if (wime-candidate-dictionary c) t :false)))
                                      records)))) "\n")))))))
