;;; Source definitions for both Onishi schemes.  Read as data, never evaluated.
;; Generate: emacs -Q --batch -l emacs-packages/wime/gen/onishi.el -f wime-onishi-generate
;; https://o24.works/layout/
;; https://zenn.dev/maki10645/articles/68f7c10a018691 (2025-01-09 article)
;;
;; Edit layout rows to move physical keys.  Edit :rows to bind syllable stems:
;; "k" uses the k row; ("hg" "hy") binds the hy row to hg.
;; Edit :extensions to move a whole family of AZIK suffixes, not hundreds of rules.
;; :extra-rules adds individual rules; :remove and :overrides are explicit exceptions.
;; Duplicate keys are errors unless changed through :overrides.

(
 :physical-rows ("qwertyuiop" "asdfghjkl;" "zxcvbnm,./" "-")
 :vowels ("a" "i" "u" "e" "o")
 :sokuon-key ";"                         ; っ in both schemes

 ;; Canonical syllables in a/i/u/e/o order.  nil means no syllable.
 ;; These also supply the ordinary-romaji comments in the generated tables.
 :syllables
 (
  ("" "あ" "い" "う" "え" "お")
  ("b" "ば" "び" "ぶ" "べ" "ぼ")
  ("by" "びゃ" "びぃ" "びゅ" "びぇ" "びょ")
  ("ch" "ちゃ" "ち" "ちゅ" "ちぇ" "ちょ")
  ("d" "だ" "ぢ" "づ" "で" "ど")
  ("dh" "でゃ" "でぃ" "でゅ" "でぇ" "でょ")
  ("dw" "どぁ" "どぃ" "どぅ" "どぇ" "どぉ")
  ("dw-azik" "だぁ" "でぃ" "どぅ" "でぇ" "どぉ")
  ("dy" "ぢゃ" "ぢぃ" "ぢゅ" "ぢぇ" "ぢょ")
  ("f" "ふぁ" "ふぃ" "ふ" "ふぇ" "ふぉ")
  ("fy" "ふゃ" "ふぃ" "ふゅ" "ふぇ" "ふょ")
  ("g" "が" "ぎ" "ぐ" "げ" "ご")
  ("gw" "ぐぁ" "ぐぃ" "ぐ" "ぐぇ" "ぐぉ")
  ("gy" "ぎゃ" "ぎぃ" "ぎゅ" "ぎぇ" "ぎょ")
  ("h" "は" "ひ" "ふ" "へ" "ほ")
  ("hy" "ひゃ" "ひぃ" "ひゅ" "ひぇ" "ひょ")
  ("j" "じゃ" "じ" "じゅ" "じぇ" "じょ")
  ("jy" "じゃ" "じぃ" "じゅ" "じぇ" "じょ")
  ("k" "か" "き" "く" "け" "こ")
  ("kw" "くぁ" "くぃ" "く" "くぇ" "くぉ")
  ("ky" "きゃ" "きぃ" "きゅ" "きぇ" "きょ")
  ("m" "ま" "み" "む" "め" "も")
  ("my" "みゃ" "みぃ" "みゅ" "みぇ" "みょ")
  ("n" "な" "に" "ぬ" "ね" "の")
  ("ny" "にゃ" "にぃ" "にゅ" "にぇ" "にょ")
  ("p" "ぱ" "ぴ" "ぷ" "ぺ" "ぽ")
  ("py" "ぴゃ" "ぴぃ" "ぴゅ" "ぴぇ" "ぴょ")
  ("r" "ら" "り" "る" "れ" "ろ")
  ("ry" "りゃ" "りぃ" "りゅ" "りぇ" "りょ")
  ("s" "さ" "し" "す" "せ" "そ")
  ("sh" "しゃ" "し" "しゅ" "しぇ" "しょ")
  ("sy" "しゃ" "しぃ" "しゅ" "しぇ" "しょ")
  ("t" "た" "ち" "つ" "て" "と")
  ("th" "てゃ" "てぃ" "てゅ" "てぇ" "てょ")
  ("ts" "つぁ" "つぃ" "つ" "つぇ" "つぉ")
  ("tw" "とぁ" "とぃ" "とぅ" "とぇ" "とぉ")
  ("tw-azik" "たぁ" "てぃ" "とぅ" "てぇ" "とぉ")
  ("ty" "ちゃ" "ちぃ" "ちゅ" "ちぇ" "ちょ")
  ("v" "ゔぁ" "ゔぃ" "ゔ" "ゔぇ" "ゔぉ")
  ("vy" "ゔゃ" "ゔぃ" "ゔゅ" "ゔぇ" "ゔょ")
  ("w" "わ" "うぃ" "う" "うぇ" "を")
  ("wh" "うぁ" "うぃ" "う" "うぇ" "うぉ")
  ("wh-azik" "わ" "うぃ" "う" "うぇ" "うぉ")
  ("x" "ぁ" "ぃ" "ぅ" "ぇ" "ぉ")
  ("xk" "ヵ" nil nil "ヶ" nil)
  ("xw" "ゎ" "ゐ" nil "ゑ" nil)
  ("xy" "ゃ" nil "ゅ" nil "ょ")
  ("y" "や" "い" "ゆ" "いぇ" "よ")
  ("z" "ざ" "じ" "ず" "ぜ" "ぞ")
  ("zy" "じゃ" "じぃ" "じゅ" "じぇ" "じょ")
 )

 ;; Shared punctuation; digits select candidates and Shift marks kana.
 :punctuation
 (("!" "！") ("," "、") ("-" "ー") ("." "。") ("?" "？") ("[" "「") ("]" "」"))

 :onishi
 (
  ;; Each row corresponds to the same physical row above.
  :layout-rows ("qlu,.fwryp" "eiao-ktnsh" "zxcv;gdmjb" "/")
  :rows
  ("" "b" "by" "ch" ("cy" "ty") "d" "dh"
   "dw" "dy" "f" "fy" "g" "gw" "gy"
   "h" "hy" "j" "jy" "k" "kw" "ky"
   ("l" "x") ("lk" "xk") ("ly" "xy") "m" "my" "n" "ny"
   "p" "py" "r" "ry" "s" "sh" "sy"
   "t" "th" "ts" "tw" "ty" "v" "vy"
   "w" "wh" "x" "xk" "xw" "xy" "y"
   "z" "zy")
  :nasal-keys ("n" "nn" "n'") ; standalone ん
  :extra-rules
  (("lwa" "ゎ"))
  :remove ()
  :overrides ()
 )

 :onishi-azik
 (
  ;; Each row corresponds to the same physical row above.
  :layout-rows ("qlu,.fwryp" "eiao-ktnsh" "jxcv;gdmzb" "/")
  ;; The article swaps J/Z; these are independent, editable layout rows.
  :rows
  ("" "b" ("bg" "by") "by" "d" "dh" ("dw" "dw-azik")
   "dy" "f" "fy" "g" "gw" "gy" "h"
   ("hg" "hy") ("hw" "f") "hy" "j" ("jy" "j") "k" "kw"
   "ky" ("l" "x") ("ly" "xy") "m" "my" "n" "ny"
   "p" ("pg" "py") "py" "r" "ry" "s" ("sg" "sh")
   "sy" "t" "th" ("tw" "tw-azik") "ty" "v" "w"
   ("wh" "wh-azik") "y" "z" ("zg" "zy") "zy")
  :nasal-keys ("c") ; standalone ん

  ;; (ID KEY VOWEL APPEND): consonant + KEY emits its VOWEL kana + APPEND.
  ;; The vowel row itself (empty stem) is never extended.
  ;; Example: change (ei "v" "e" "い") to (ei "q" "e" "い") to move all ei chords.
  :extensions
  ((an "n" "a" "ん") (in "s" "i" "ん") (un "r" "u" "ん")
   (en "h" "e" "ん") (on "t" "o" "ん")
   (ou "c" "o" "う") (ai "," "a" "い") (ei "v" "e" "い") (uu "l" "u" "う"))

  ;; Article exception: ly has no e syllable, but ly + the ei key emits ぇい.
  ;; Refer to the extension ID so changing v also moves this rule.
  :extension-extras ((ei "ly" "ぇい"))
  :extra-rules
  (("#" "#") ("$" "$") ("%" "%") ("&" "&")
   ("'" "'") ("(" "(") (")" ")") ("*" "*")
   ("+" "+") ("/" "/") ("=" "=") ("@" "@")
   ("\\" "\\") ("^" "^") ("`" "`") ("v." "゜")
   ("z(" "（") ("z)" "）") ("z-" "～") ("z/" "・")
   ("z;" "゛") ("z[" "『") ("z]" "』") ("z~" "〜")
   ("|" "|") ("~" "~"))
  :remove ()
  :overrides ()
 )
)
