;;; search-conditional.el --- Experiment adapter to the actual wime lattice -*- lexical-binding: t; -*-
(require 'wime)
(require 'json)
(set-language-environment "UTF-8")
(set-terminal-coding-system 'utf-8-unix)
(set-keyboard-coding-system 'utf-8-unix)
(setq coding-system-for-read 'utf-8-unix coding-system-for-write 'utf-8-unix
      wime-database-file (getenv "WIME_DATABASE")
      wime-grammar-module (getenv "WIME_GRAMMAR_MODULE")
      wime-grammar-shared-dir (getenv "WIME_GRAMMAR"))
(unless (file-readable-p (expand-file-name "wanxiang-lts-zh-hans.gram" wime-grammar-shared-dir))
  (error "Missing octagram model"))
(condition-case nil
    (while t
      (let* ((request (json-parse-string (read-from-minibuffer "") :object-type 'alist))
             (wime-search-width (alist-get 'width request))
             (prefix (alist-get 'prefix request))
             (start (float-time))
             (records (wime--sentence-records
                       (wime--parse (alist-get 'input request))
                       (unless (string-empty-p prefix) prefix))))
        (princ (decode-coding-string (json-serialize
                `(:ms ,(* 1000 (- (float-time) start))
                  :candidates ,(vconcat
                                (mapcar (lambda (c) `(:text ,(wime-candidate-text c)
                                                     :end ,(wime-candidate-end c))) records)))) 'utf-8-unix))
        (princ "\n")))
  (end-of-file nil))
