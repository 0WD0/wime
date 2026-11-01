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

(ert-deftest wime-japanese-azik-converts-kana ()
  (dolist (case '(("kz" "かん" "") ("kl" "こん" "") ("nihlgo" "にほんご" "") ("ds" "です" "")
                  ("xa;ta" "しゃった" "") ("k" "" "k") ("ky" "" "ky") ("n" "" "n") ("nn" "ん" "")
                  ("ga;kp" "がっこう" "") ("xt" "しゅつ" "") ("xxa" "ぁ" "")
                  ;; Only ; types sokuon; unmatched keys stay visible as typed.
                  ("bb" "b" "b") ("jbm" "jb" "m") ("ka," "か、" "")))
    (should (equal (wime-japanese--convert (car case)) (cons (nth 1 case) (nth 2 case))))))

(defmacro wime-japanese-test--with-scheme (name &rest body)
  "Run BODY in a buffer using the input method NAME."
  (declare (indent 1))
  `(with-temp-buffer
     (unwind-protect
         (progn (activate-input-method ,name) ,@body)
       (deactivate-input-method))))

(ert-deftest wime-japanese-base-types-ordinary-romaji ()
  (dolist (case '(("nihongo" "にほんご") ("shichitsu" "しちつ") ("ga;kou" "がっこう")
                  ("nihon" "にほん") ("GA:KOU" "ガッコウ")))
    (wime-japanese-test--with-scheme "wime-japanese-base"
      (dolist (key (string-to-list (car case)))
        (should-not (funcall input-method-function key)))
      (wime-japanese-commit-kana)
      (should (equal (buffer-string) (cadr case))))))

(ert-deftest wime-japanese-onishi-types-ordinary-romaji ()
  ;; Physical QWERTY keys, read as nihongo, shichitsu, ga;kou, and nihon.
  (dolist (case '(("ks;fknf" "にほんご") ("l;sc;sjle" "しちつ")
                  ("ndbhfe" "がっこう") ("ks;fk" "にほん")
                  ("zd.d" "ざじゃ") ("rtg" "、。ー")))
    (wime-japanese-test--with-scheme "wime-japanese-onishi"
      (dolist (key (string-to-list (car case)))
        (should-not (funcall input-method-function key)))
      (wime-japanese-commit-kana)
      (should (equal (buffer-string) (cadr case))))))

(ert-deftest wime-japanese-onishi-azik-types-article-rules ()
  ;; Logical: nihtgo, ga;kc, kn/ks/kr/kh/kt, kc/k,/kv/kl, and the J/Z swap.
  (dolist (case '(("ks;jnf" "にほんご") ("ndbhc" "がっこう")
                  ("hkhlhih;hj" "かんきんくんけんこん")
                  ("hchrhvhw" "こうかいけいくう") ("c" "ん")
                  ("zd.d" "じゃざ") (",lml" "みんぢん")
                  ("lnd" "しゃ") ("wd" "ぁ")
                  ("nor/nrvr" "ぎゃいびゃいゔぁい")))
    (wime-japanese-test--with-scheme "wime-japanese-onishi-azik"
      (dolist (key (string-to-list (car case)))
        (should-not (funcall input-method-function key)))
      (wime-japanese-commit-kana)
      (should (equal (buffer-string) (cadr case))))))

(ert-deftest wime-japanese-onishi-shift-follows-the-physical-key ()
  (dolist (case '(("wime-japanese-onishi" "NDBHFE" "GA:KOU")
                  ("wime-japanese-onishi-azik" "NDBHC" "GA:KC")))
    (wime-japanese-test--with-scheme (car case)
      (mapc input-method-function (string-to-list (nth 1 case)))
      (should (equal wime-japanese--keys (nth 2 case)))
      (let ((kana (car (wime-japanese--convert wime-japanese--keys))))
        (should (equal (substring-no-properties kana) "がっこう"))
        (should (equal (wime-japanese-test--shift kana) "1111")))
      (should (= (wime-japanese--translate-key ?:) ?H))
      (should (= (wime-japanese--translate-key ??) ?B))
      (wime-japanese-cancel))))

(ert-deftest wime-japanese-scheme-switching-is-buffer-local ()
  (wime-japanese-test--with-scheme "wime-japanese-onishi"
    (should (equal (wime-japanese--convert "sh") '("" . "sh")))
    (wime-japanese-test--with-scheme "wime-japanese-onishi-azik"
      (should (equal (wime-japanese--convert "sh") '("せん" . "")))
      (mapc input-method-function (string-to-list "hd"))
      ;; Digits still select candidates; source-table digit rules are omitted.
      (funcall input-method-function ?2)
      (should (equal (buffer-string) "カ")))
    (should (equal (wime-japanese--convert "sh") '("" . "sh")))
    (activate-input-method "wime-japanese-onishi-azik")
    (should (equal (wime-japanese--convert "sh") '("せん" . "")))
    (activate-input-method "wime-japanese")
    (should (equal (wime-japanese--convert "sh") '("すう" . "")))
    (activate-input-method "wime-japanese-base")
    (should (equal (wime-japanese--convert "sh") '("" . "sh")))
    (should (equal (wime-japanese--convert "desu") '("です" . "")))
    (should (= (wime-japanese--translate-key ?h) ?h))))

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
    (mapc #'wime-japanese-input-method (string-to-list "kz"))
    (should (equal wime-japanese--candidates '("かん" "カン")))
    (should (string-prefix-p "かん" (overlay-get wime-japanese--overlay 'after-string)))
    (wime-japanese-input-method ?2)
    (should (equal (buffer-string) "カン"))
    (should-not (wime-japanese--active-p))
    (mapc #'wime-japanese-input-method (string-to-list "kzk"))
    (wime-japanese-delete-backward)
    (wime-japanese-commit-kana)
    (should (equal (buffer-string) "カンかん"))))

(ert-deftest wime-japanese-request-carries-kana-and-pending-starts ()
  (with-temp-buffer
    (insert "前")
    (let ((wime-neural--failed nil) (wime-neural-rescorer "/lm")
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
  ;; Bind what the Wime Lab startup recipe would otherwise set in wdoom.
  (let ((wime-neural-command '("/bin/true")) (wime-neural-rescorer "/lm")
        (wime-neural-rescorer-weight 1.5) (wime-neural-beam 32) (wime-database-file "/w.db")
        (wime-japanese-model-file "/ja.safetensors")
        (wime-japanese-table-file "/ja.json") (wime-japanese-zenz-directory nil))
    (should (equal (wime-neural--command "/m.safetensors")
                   '("/bin/true" "--rescorer" "/lm" "--rescorer-weight" "1.5"
                     "--database" "/w.db" "--beam" "32" "--japanese-model" "/ja.safetensors" "--japanese-table" "/ja.json" "/m.safetensors")))
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
    (let ((wime-neural--failed nil) (wime-neural-rescorer "/lm")
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
