;;; onishi-generation-test.el --- Editable Onishi scheme generation -*- lexical-binding: t; -*-
(require 'ert)
(require 'wime-japanese)
(load (expand-file-name "../gen/onishi.el" (file-name-directory (or load-file-name buffer-file-name))) nil t)

(defun wime-onishi-test--source ()
  (wime-onishi--read (expand-file-name "onishi-rules.eld" wime-onishi--directory)))

(defmacro wime-onishi-test--with-generated (source &rest body)
  "Generate SOURCE in an isolated directory, then run BODY against it."
  (declare (indent 1))
  `(let* ((directory (make-temp-file "wime-onishi-" t))
          (file (expand-file-name "source.eld" directory))
          (wime-japanese--directory directory))
     (unwind-protect
         (progn
           (with-temp-file file (prin1 ,source (current-buffer)))
           (wime-onishi-generate file directory)
           ,@body)
       (delete-directory directory t))))

(ert-deftest wime-onishi-generated-files-are-current ()
  (wime-onishi-check))

(ert-deftest wime-onishi-changing-suffixes-moves-whole-families ()
  (let* ((source (wime-onishi-test--source))
         (scheme (plist-get source :onishi-azik)))
    (setf (nth 1 (assq 'ei (plist-get scheme :extensions))) "q"
          (nth 1 (assq 'an (plist-get scheme :extensions))) "d"
          (plist-get scheme :nasal-keys) '("x"))
    (wime-onishi-test--with-generated source
      (with-temp-buffer
        (let ((wime-neural-mode nil))
          (unwind-protect
              (progn
                (activate-input-method "wime-japanese-onishi-azik")
                ;; Physical hq -> kq; lnd -> sga.
                (mapc input-method-function (string-to-list "hqlnd"))
                (wime-japanese-commit-kana)
                (should (equal (buffer-string) "けいしゃ"))
                (should (equal (wime-japanese--convert "kdgyd") '("かんぎゃん" . "")))
                (should (equal (wime-japanese--convert "lyq") '("ぇい" . "")))
                (should (equal (wime-japanese--convert "x") '("ん" . "")))
                (dolist (old '("kv" "lyv" "kn" "gyn" "c"))
                  (should-not (gethash old (car (wime-japanese--rules))))))
            (deactivate-input-method)))))))

(ert-deftest wime-onishi-layouts-can-be-edited-independently ()
  (let* ((source (wime-onishi-test--source))
         (scheme (plist-get source :onishi-azik)))
    ;; Move logical k from physical h to physical n, exchanging k/g in AZIK only.
    (setf (plist-get scheme :layout-rows) '("qlu,.fwryp" "eiao-gtnsh" "jxcv;kdmzb" "/"))
    (wime-onishi-test--with-generated source
      (with-temp-buffer
        (unwind-protect
            (progn
              (activate-input-method "wime-japanese-onishi-azik")
              (should (= (wime-japanese--translate-key ?n) ?k))
              (should (= (wime-japanese--translate-key ?N) ?K))
              (should (= (wime-japanese--translate-key ?h) ?g))
              (activate-input-method "wime-japanese-onishi")
              (should (= (wime-japanese--translate-key ?n) ?g))
              (should (= (wime-japanese--translate-key ?h) ?k))
              ;; Regeneration is picked up when the same method is re-enabled.
              (setf (plist-get (plist-get source :onishi) :layout-rows)
                    (plist-get scheme :layout-rows))
              (with-temp-file file (prin1 source (current-buffer)))
              (wime-onishi-generate file directory)
              (deactivate-input-method)
              (activate-input-method "wime-japanese-onishi")
              (should (= (wime-japanese--translate-key ?h) ?g)))
          (deactivate-input-method))))))

(ert-deftest wime-onishi-stems-and-individual-exceptions-are-editable ()
  (let* ((source (wime-onishi-test--source))
         (scheme (plist-get source :onishi)))
    (setf (plist-get scheme :rows)
          (cons '("q" "k") (delete "k" (plist-get scheme :rows)))
          (plist-get scheme :remove) '("shi")
          (plist-get scheme :overrides) '(("qa" "くぁ")))
    (wime-onishi-test--with-generated source
      (with-temp-buffer
        (let ((wime-japanese--scheme 'onishi))
          (should (equal (wime-japanese--convert "qaqi") '("くぁき" . "")))
          (should-not (gethash "ka" (car (wime-japanese--rules))))
          (should-not (gethash "shi" (car (wime-japanese--rules)))))))))

(ert-deftest wime-onishi-rejects-conflicting-definitions-before-writing ()
  (let* ((source (wime-onishi-test--source))
         (scheme (plist-get source :onishi-azik))
         (directory (make-temp-file "wime-onishi-invalid-" t))
         (file (expand-file-name "source.eld" directory))
         (output (expand-file-name "romaji.eld" directory)))
    (unwind-protect
        (progn
          ;; ei cannot take ou's c binding silently.
          (setf (nth 1 (assq 'ei (plist-get scheme :extensions))) "c")
          (with-temp-file file (prin1 source (current-buffer)))
          (with-temp-file output (insert "keep existing output"))
          (should (string-match-p "Conflicting key" (error-message-string
                                                     (should-error (wime-onishi-generate file directory)))))
          (should (equal (with-temp-buffer (insert-file-contents output) (buffer-string))
                         "keep existing output")))
      (delete-directory directory t))))

(ert-deftest wime-onishi-rejects-ambiguous-layouts-and-unknown-rows ()
  (let* ((source (wime-onishi-test--source))
         (scheme (plist-get source :onishi)))
    (setf (plist-get scheme :layout-rows) '("qlu,.fwryp" "eiao-ktnsk" "zxcv;gdmjb" "/"))
    (should-error (wime-onishi--outputs source)))
  (let* ((source (wime-onishi-test--source))
         (scheme (plist-get source :onishi)))
    (push '("q" "typo") (plist-get scheme :rows))
    (should-error (wime-onishi--outputs source)))
  (dolist (key '("1" "A" "k:"))
    (should-error (wime-onishi--key key))))

;;; onishi-generation-test.el ends here
