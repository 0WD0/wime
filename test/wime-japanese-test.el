;;; wime-japanese-test.el --- AZIK conversion and the Japanese worker protocol -*- lexical-binding: t; -*-
(require 'ert)
(require 'wime-japanese)

(defmacro wime-japanese-test--with-rules (contents &rest body)
  "Run BODY with a temporary rule table containing CONTENTS."
  (declare (indent 1))
  `(let ((wime-japanese-azik-file (make-temp-file "wime-azik-" nil ".eld"))
         (wime-japanese--table nil))
     (unwind-protect
         (progn
           (with-temp-file wime-japanese-azik-file (insert ,contents))
           ,@body)
       (delete-file wime-japanese-azik-file))))

(ert-deftest wime-japanese-rule-data-preserves-comments-punctuation-and-pending ()
  (wime-japanese-test--with-rules
      ";; A hand-edited table.\n((\"ba\" \"ば\")\n (\"bb\" \"っ\" \"b\") ; retain b\n (\";\" \"っ\") (\"z \" \"\\u3000\"))\n; End.\n"
    (should (equal (wime-japanese--convert "bba;z ") '("っばっ　" . "")))
    (should (= (hash-table-count (car (wime-japanese--rules))) 4))))

(ert-deftest wime-japanese-rule-data-rejects-duplicate-keys ()
  (wime-japanese-test--with-rules "((\"ka\" \"か\") (\"ka\" \"が\"))"
    (should (string-match-p "Duplicate AZIK keys"
                            (error-message-string (should-error (wime-japanese--rules)))))
    (should-not wime-japanese--table)))

(ert-deftest wime-japanese-rule-data-rejects-malformed-tables ()
  (dolist (contents '("[\"ka\" \"か\"]" "((\"ka\"))" "((\"ka\" 1))"
                      "((\"ka\" \"か\" nil))" "((\"ka\" \"か\" \"\" \"extra\"))"
                      "((\"\" \"か\"))" "((\"ka\" \"\"))" "((\"ka\" . \"か\"))"
                      "((\"ka\" \"か\")) (" "((\"ka\" \"か\")) nil" "((\"ka\" \"か\")"))
    (wime-japanese-test--with-rules contents
      (should-error (wime-japanese--rules))
      (should-not wime-japanese--table))))

(ert-deftest wime-japanese-azik-converts-like-skk ()
  (dolist (case '(("kz" "かん" "") ("kl" "こん" "") ("nihlgo" "にほんご" "") ("ds" "です" "")
                  ("xa;ta" "しゃった" "") ("k" "" "k") ("ky" "" "ky") ("n" "" "n") ("nn" "ん" "")
                  ;; A finished rule may leave a key pending; a stuck sequence stays as typed.
                  ("bb" "っ" "b") ("jbm" "jb" "m") ("ka," "か、" "")))
    (should (equal (wime-japanese--convert (car case)) (cons (nth 1 case) (nth 2 case))))))

(ert-deftest wime-japanese-pending-keys-name-the-kana-they-can-start ()
  (should (equal (seq-take (wime-japanese--starts "k") 5) '("か" "き" "く" "け" "こ")))
  (should (equal (wime-japanese--starts "ky") '("き")))
  (should-not (wime-japanese--starts "")))

(ert-deftest wime-japanese-keys-that-only-continue-a-rule-are-accepted-mid-composition ()
  (with-temp-buffer
    (setq wime-japanese--keys "")
    (should (wime-japanese--key-p ?k))
    (should-not (wime-japanese--key-p ?l))
    (setq wime-japanese--keys "k")
    (should (wime-japanese--key-p ?l))
    (should-not (wime-japanese--key-p ?1))))

(ert-deftest wime-japanese-typing-offers-kana-and-commits ()
  (with-temp-buffer
    (let ((wime-neural-mode nil))
      (mapc #'wime-japanese-input-method (string-to-list "kz"))
      (should (equal wime-japanese--candidates '("かん" "カン")))
      (should (string-prefix-p "かん" (overlay-get wime-japanese--overlay 'after-string)))
      (wime-japanese-input-method ?2)
      (should (equal (buffer-string) "カン"))
      (should-not (wime-japanese--active-p))
      (mapc #'wime-japanese-input-method (string-to-list "kzk"))
      (wime-japanese-delete-backward)
      (wime-japanese-commit-kana)
      (should (equal (buffer-string) "カンかん")))))

(ert-deftest wime-japanese-request-carries-kana-and-pending-starts ()
  (with-temp-buffer
    (insert "前")
    (let ((wime-neural-mode t) (wime-neural--failed nil) (wime-neural-rescorer "/lm")
          (wime-japanese-model-file "/m") (wime-japanese-table-file "/t") scheduled)
      (cl-letf (((symbol-function 'run-with-timer)
                 (lambda (_delay _repeat _function job) (setq scheduled job) nil)))
        (mapc #'wime-japanese-input-method (string-to-list "nihlg")))
      (should (eq (wime-neural--job-method scheduled) 'japanese))
      (let ((request (json-parse-string (wime-neural--request scheduled) :object-type 'alist :array-type 'list)))
        (should (equal (alist-get 'operation request) "japanese"))
        (should (equal (alist-get 'kana request) "にほん"))
        (should (equal (alist-get 'context request) "前"))
        (should (member "ご" (alist-get 'pending request))))
      ;; Decoded readings lead; kana remain; a reply for older keys is ignored.
      (should (wime-neural--valid-p scheduled))
      (wime-neural--apply scheduled '((id . 1) (candidates . [["日本語" -1.0] ["" -2.0]])))
      (should (equal (car wime-japanese--candidates) "日本語"))
      (should (member "にほんg" wime-japanese--candidates))
      (cl-letf (((symbol-function 'run-with-timer) #'ignore))
        (wime-japanese-input-method ?o))
      (should-not (wime-neural--valid-p scheduled))
      (wime-japanese--reset))))

(ert-deftest wime-japanese-worker-loads-the-japanese-models ()
  (let ((wime-neural-program "/bin/true") (wime-neural-interpreter nil) (wime-neural-rescorer "/lm")
        (wime-neural-search-method 'verify) (wime-japanese-model-file "/ja.safetensors")
        (wime-japanese-table-file "/ja.json"))
    (should (equal (wime-neural--command "/m.safetensors")
                   '("/bin/true" "--rescorer" "/lm" "--rescorer-weight" "1.5"
                     "--japanese-model" "/ja.safetensors" "--japanese-table" "/ja.json" "/m.safetensors")))
    (let ((wime-japanese-zenz-directory "/zenz"))
      (should (equal (last (wime-neural--command "/m.safetensors") 3)
                     '("--japanese-zenz" "/zenz" "/m.safetensors"))))))

(defun wime-japanese-test--shift (kana)
  (mapconcat (lambda (i) (if (get-text-property i 'wime-japanese-shift kana) "1" "0"))
             (number-sequence 0 (1- (length kana))) ""))

(ert-deftest wime-japanese-held-shift-marks-the-kana-to-convert ()
  (dolist (case '(("KAku" "かく" "10") ("TAKq" "たかい" "110") ("Yqtnda" "やいたんだ" "10000")
                  ;; Shift turns ; into :, which still types っ.
                  ("GA:KOU" "がっこう" "1111") ("I:SYOni" "いっしょに" "11110")))
    (let ((converted (wime-japanese--convert (car case))))
      (should (equal (substring-no-properties (car converted)) (nth 1 case)))
      (should (equal (wime-japanese-test--shift (car converted)) (nth 2 case)))))
  (should (wime-japanese--explicit-p "KAku"))
  (should-not (wime-japanese--explicit-p "kaku?"))
  (with-temp-buffer
    (setq wime-japanese--keys "")
    (should (wime-japanese--key-p ?K))
    (should (wime-japanese--key-p ?:))))

(ert-deftest wime-japanese-shift-spans-reach-the-worker-and-the-preedit ()
  (with-temp-buffer
    (let ((wime-neural-mode t) (wime-neural--failed nil) (wime-neural-rescorer "/lm")
          (wime-japanese-model-file "/m") (wime-japanese-table-file "/t") scheduled)
      (cl-letf (((symbol-function 'run-with-timer)
                 (lambda (_delay _repeat _function job) (setq scheduled job) nil)))
        (mapc #'wime-japanese-input-method (string-to-list "TAKqY")))
      (let ((request (json-parse-string (wime-neural--request scheduled) :object-type 'alist :array-type 'list)))
        (should (equal (alist-get 'kana request) "たかい"))
        (should (equal (alist-get 'shift request) "110"))
        (should (eq (alist-get 'explicit request) t))
        (should (eq (alist-get 'pending_shift request) t)))
      ;; Shifted kana as katakana leads the fallbacks; the preedit underlines them.
      (should (equal (car wime-japanese--candidates) "タカいY"))
      (let ((preedit (overlay-get wime-japanese--overlay 'after-string)))
        (should (memq 'wime-japanese-shift-face (ensure-list (get-text-property 0 'face preedit))))
        (should-not (memq 'wime-japanese-shift-face (ensure-list (get-text-property 2 'face preedit)))))
      (wime-japanese--reset))))
