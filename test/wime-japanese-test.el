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
  (dolist (case '(("wime-japanese-onishi" "NDBHFE") ("wime-japanese-onishi-azik" "NDBHC")))
    (wime-japanese-test--with-scheme (car case)
      (mapc input-method-function (string-to-list (nth 1 case)))
      (let ((kana (wime-japanese--kana (wime-japanese--run))))
        (should (equal (substring-no-properties kana) "がっこう"))
        (should (equal (wime-japanese--shift kana) "1111")))
      (should (= (wime-japanese--translate-key ?:) ?H))
      (should (= (wime-japanese--translate-key ??) ?B))
      (wime-japanese-cancel))))

(ert-deftest wime-japanese-scheme-switching-is-buffer-local ()
  (wime-japanese-test--with-scheme "wime-japanese-onishi"
    (should (equal (wime-japanese--convert "sh") '("" . "sh")))
    (wime-japanese-test--with-scheme "wime-japanese-onishi-azik"
      (should (equal (wime-japanese--convert "sh") '("せん" . "")))
      (mapc input-method-function (string-to-list "hd"))
      ;; Digits join the sentence; source-table digit rules are omitted.
      (funcall input-method-function ?2)
      (should (equal (buffer-string) "か2"))
      (wime-japanese-cancel))
    (should (equal (wime-japanese--convert "sh") '("" . "sh")))
    (activate-input-method "wime-japanese-onishi-azik")
    (should (equal (wime-japanese--convert "sh") '("せん" . "")))
    (activate-input-method "wime-japanese")
    (should (equal (wime-japanese--convert "sh") '("すう" . "")))
    (activate-input-method "wime-japanese-base")
    (should (equal (wime-japanese--convert "sh") '("" . "sh")))
    (should (equal (wime-japanese--convert "desu") '("です" . "")))
    (should (= (wime-japanese--translate-key ?h) ?h))))

(ert-deftest wime-japanese-keys-that-only-continue-a-rule-are-accepted-mid-composition ()
  (with-temp-buffer
    (should (wime-japanese--key-p ?k))
    (should-not (wime-japanese--key-p ?l))
    (setq wime-japanese--pending "k")
    (should (wime-japanese--key-p ?l))
    (should (wime-japanese--key-p ?1))))

(defmacro wime-japanese-test--decoding (&rest body)
  "Run BODY in a buffer after 前, with decoding configured and SCHEDULED
bound to the latest job instead of a timer."
  (declare (indent 0))
  `(with-temp-buffer
     (insert "前")
     (let ((wime-neural--failed nil) (wime-neural-rescorer "/lm")
           (wime-japanese-model-file "/m") (wime-japanese-table-file "/t") scheduled)
       (cl-letf (((symbol-function 'run-with-timer)
                  (lambda (_delay _repeat _function job) (setq scheduled job) nil)))
         ,@body))))

(defun wime-japanese-test--type (keys)
  (dolist (key (string-to-list keys))
    (should-not (wime-japanese-input-method key))))

(defun wime-japanese-test--request (job)
  (json-parse-string (wime-neural--request job) :object-type 'alist :array-type 'list))

(defun wime-japanese-test--segments ()
  "The run at point as (TEXT . KANA) pairs."
  (mapcar (lambda (g) (cons (buffer-substring-no-properties (car g) (cadr g))
                            (substring-no-properties (wime-japanese--segment-kana (nth 2 g)))))
          (wime-japanese--run)))

(ert-deftest wime-japanese-kana-enter-the-buffer-and-a-candidate-confirms ()
  (with-temp-buffer
    (wime-japanese-test--type "kz")
    (should (equal (buffer-string) "かん"))
    (should (equal wime-japanese--candidates '("かん" "カン")))
    (let ((last-command-event ?\C-2))
      (wime-japanese-choose))
    (should (equal (buffer-string) "カン"))
    (should-not (wime-japanese--uncommitted-p))
    (should-not (wime-japanese--converting-p))
    ;; A pending key is deleted first, then the kana before point.
    (wime-japanese-test--type "kzk")
    (wime-japanese-delete-backward)
    (should (string-empty-p wime-japanese--pending))
    (wime-japanese-delete-backward)
    (should (equal (buffer-string) "カンか"))
    (wime-japanese-commit-kana)
    (should (equal (buffer-string) "カンか"))
    (should-not (wime-japanese--uncommitted-p))))

(ert-deftest wime-japanese-decoding-rewrites-the-window-and-keeps-each-segments-kana ()
  (wime-japanese-test--decoding
    (wime-japanese-test--type "nihlg")
    (should (equal (buffer-string) "前にほん"))
    (should (equal wime-japanese--pending "g"))
    (let ((request (wime-japanese-test--request scheduled)))
      (should (equal (alist-get 'operation request) "japanese"))
      (should (equal (alist-get 'kana request) "にほん"))
      (should (equal (alist-get 'context request) "前"))
      (should-not (alist-get 'pending request)))
    (should (wime-neural--valid-p scheduled))
    (wime-neural--apply scheduled '((id . 1) (candidates . [["日本" -1.0 [[3 "日本"]] []]
                                                             ["にほん" -2.0 [[2 "にほ"] [1 "ん"]] []]
                                                             ["偽" -3.0 [[2 "偽"]] []]])))
    (should (equal (buffer-string) "前日本"))
    (should (= (point) (point-max)))
    (should (equal (wime-japanese-test--segments) '(("日本" . "にほん"))))
    ;; Unaligned readings are dropped; the kana remain choices.
    (should (equal wime-japanese--candidates '("日本" "にほん" "ニホン")))
    ;; Later kana rewrite earlier segments in the same decode.
    (let ((previous scheduled))
      (wime-japanese-test--type "o")
      (should-not (wime-neural--valid-p previous)))
    (should (equal (buffer-string) "前日本ご"))
    (should (equal (alist-get 'kana (wime-japanese-test--request scheduled)) "にほんご"))
    (wime-neural--apply scheduled '((id . 2) (candidates . [["日本語" -1.0 [[4 "日本語"]] []]])))
    (should (equal (wime-japanese-test--segments) '(("日本語" . "にほんご"))))
    ;; Deleting takes the last kana of a written segment.
    (wime-japanese-delete-backward)
    (should (equal (wime-japanese-test--segments) '(("にほん" . "にほん"))))
    (wime-japanese--reset)))

(defmacro wime-japanese-test--spans (&rest body)
  "Run BODY after decoding にほんごを with the span view: 日本語を leads
and 日本後を, 二本語を follow, with weights that leave 語 and 日 uncertain."
  (declare (indent 0))
  `(wime-japanese-test--decoding
     (let ((wime-candidate-style 'spans) (wime-confidence-threshold 0.9)
           (wime-japanese-confidence-temperature 1.0))
       (wime-japanese-test--type "nihlgowo")
       (wime-neural--apply scheduled
                           '((id . 1) (candidates . [["日本語を" -1.0 [[4 "日本語"] [1 "を"]] [[1 1] [3 2] [4 3]]]
                                                     ["日本後を" -1.5 [[3 "日本"] [1 "後"] [1 "を"]] [[1 1] [3 2] [4 3]]]
                                                     ["二本語を" -2.0 [[1 "二"] [2 "本"] [1 "語"] [1 "を"]] [[1 1] [3 2] [4 3]]]])))
       ,@body)))

(ert-deftest wime-japanese-span-view-numbers-alternatives-under-uncertain-spans ()
  (wime-japanese-test--spans
    (should (equal (buffer-string) "前日本語を"))
    (should (equal (wime-japanese--uncertain-spans) '((0 1 ("二")) (3 4 ("後")))))
    (should (equal wime-japanese--span-choices '((0 1 "二") (3 4 "後"))))
    ;; The alternatives sit under their spans; the whole readings are not listed.
    (should (seq-some (lambda (o) (string-match-p "1\\..*二\\|2\\..*後" (or (overlay-get o 'after-string) "")))
                      wime-japanese--columns))
    (should-not (string-match-p "日本後を" (overlay-get wime-japanese--overlay 'after-string)))
    ;; TAB lists whole readings instead, and a digit then confirms one.
    (wime-japanese-toggle-candidate-list)
    (should-not wime-japanese--columns)
    (should (string-match-p "2\\.日本後を" (overlay-get wime-japanese--overlay 'after-string)))
    (wime-japanese-toggle-candidate-list)
    (should wime-japanese--columns)
    (wime-japanese--reset)))

(ert-deftest wime-japanese-a-chosen-span-is-pinned-and-the-rest-decodes-again ()
  (wime-japanese-test--spans
    (let ((last-command-event ?\C-2))
      (wime-japanese-choose))
    ;; 語 is split off the 日本語 segment and pinned as 後.
    (should (equal (buffer-string) "前日本後を"))
    (should (equal (wime-japanese-test--segments) '(("日本" . "にほん") ("後" . "ご") ("を" . "を"))))
    (should (= (point) (point-max)))
    (let ((request (wime-japanese-test--request scheduled)))
      (should (equal (alist-get 'pins request) '((3 4 "後")))))
    (wime-neural--apply scheduled '((id . 2) (candidates . [["二本後を" -1.0 [[1 "二"] [2 "本"] [1 "後"] [1 "を"]] []]])))
    (should (equal (buffer-string) "前二本後を"))
    ;; The pin travels with its segment through later decodes.
    (should (equal (wime-japanese--pins (wime-japanese--window)) '((3 4 "後"))))
    (wime-japanese--reset)))

(ert-deftest wime-japanese-f7-and-f6-pin-the-segment-before-point-as-kana ()
  (wime-japanese-test--decoding
    (wime-japanese-test--type "terebi")
    (wime-neural--apply scheduled '((id . 1) (candidates . [["てれび" -1.0 [[3 "てれび"]] []]])))
    (wime-japanese-katakana)
    (should (equal (buffer-string) "前テレビ"))
    (should (equal (wime-japanese--pins (wime-japanese--window)) '((0 3 "テレビ"))))
    (should (equal (alist-get 'pins (wime-japanese-test--request scheduled)) '((0 3 "テレビ"))))
    (wime-japanese-hiragana)
    (should (equal (buffer-string) "前てれび"))
    (wime-japanese--reset)))

(ert-deftest wime-japanese-replies-for-other-text-are-ignored ()
  (wime-japanese-test--decoding
    (wime-japanese-test--type "nihl")
    (let ((job scheduled))
      (wime-japanese-test--type "go")
      (should-not (wime-neural--valid-p job))
      (wime-neural--apply job '((id . 1) (candidates . [["日本" -1.0 [[3 "日本"]] []]])))
      (should (equal (buffer-string) "前にほんご")))
    (wime-japanese--reset)))

(ert-deftest wime-japanese-suspended-text-keeps-its-kana-and-typing-resumes-it ()
  (wime-japanese-test--decoding
    (wime-japanese-test--type "nihl")
    (wime-neural--apply scheduled '((id . 1) (candidates . [["日本" -1.0 [[3 "日本"]] []]])))
    (wime-japanese-pause)
    (should-not (wime-japanese--converting-p))
    (should (wime-japanese--uncommitted-p))
    (should (equal (wime-japanese-test--segments) '(("日本" . "にほん"))))
    (wime-japanese-test--type "go")
    (should (equal (alist-get 'kana (wime-japanese-test--request scheduled)) "にほんご"))
    ;; SPC confirms every segment and keeps the text; then it is a space again.
    (should-not (wime-japanese-input-method ?\s))
    (should (equal (buffer-string) "前日本ご"))
    (should-not (wime-japanese--uncommitted-p))
    (should (equal (wime-japanese-input-method ?\s) '(?\s)))))

(ert-deftest wime-japanese-window-bounds-the-kana-decoded-together ()
  (wime-japanese-test--decoding
    (let ((wime-japanese-window 4))
      (wime-japanese-test--type "kakikukeko")
      (let ((request (wime-japanese-test--request scheduled)))
        (should (equal (alist-get 'kana request) "きくけこ"))
        (should (equal (alist-get 'context request) "前か"))))
    (wime-japanese--reset)))

(ert-deftest wime-japanese-an-ordinary-edit-makes-a-segment-ordinary-text ()
  (wime-japanese-test--decoding
    (activate-input-method "wime-japanese")
    (wime-japanese-test--type "nihl")
    (wime-neural--apply scheduled '((id . 1) (candidates . [["日本" -1.0 [[3 "日本"]] []]])))
    (wime-japanese-pause)
    (delete-char -1)
    (should (equal (buffer-string) "前日"))
    (should-not (wime-japanese--uncommitted-p))
    (deactivate-input-method)))

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

(ert-deftest wime-japanese-held-shift-marks-the-kana-to-convert ()
  (dolist (case '(("KAku" "かく" "10") ("TAKq" "たかい" "110") ("Yqtnda" "やいたんだ" "10000")
                  ;; Shift turns ; into :, which still types っ.
                  ("GA:KOU" "がっこう" "1111") ("I:SYOni" "いっしょに" "11110")))
    (let ((converted (wime-japanese--convert (car case))))
      (should (equal (substring-no-properties (car converted)) (nth 1 case)))
      (should (equal (wime-japanese--shift (car converted)) (nth 2 case)))))
  (should (wime-japanese--explicit-p (car (wime-japanese--convert "KAku"))))
  (should-not (wime-japanese--explicit-p (car (wime-japanese--convert "kaku?"))))
  (with-temp-buffer
    (should (wime-japanese--key-p ?K))
    (should (wime-japanese--key-p ?:))))

(ert-deftest wime-japanese-shift-marks-reach-the-worker-and-the-buffer ()
  (wime-japanese-test--decoding
    (wime-japanese-test--type "TAKqY")
    (let ((request (wime-japanese-test--request scheduled)))
      (should (equal (alist-get 'kana request) "たかい"))
      (should (equal (alist-get 'shift request) "110"))
      (should (eq (alist-get 'explicit request) t)))
    ;; Shifted kana as katakana leads the fallbacks; the text underlines them.
    (should (equal (car wime-japanese--candidates) "タカい"))
    (cl-flet ((shift-p (position)
                (seq-some (lambda (o) (eq (overlay-get o 'face) 'wime-japanese-shift-face))
                          (overlays-at position))))
      (should (shift-p 2))
      (should (shift-p 3))
      (should-not (shift-p 4)))
    (wime-japanese-commit-kana)
    (should (equal (buffer-string) "前タカいY"))))
