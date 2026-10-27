;;; Source definitions for all three Japanese schemes.  Read as data, never evaluated.
;; Generate: emacs -Q --batch -l emacs-packages/wime/gen/japanese.el -f wime-japanese-generate
;; https://o24.works/layout/
;; https://zenn.dev/maki10645/articles/68f7c10a018691 (2025-01-09 article)
;;
;; Edit layout rows to move physical keys.  Edit :rows to bind syllable stems:
;; "k" uses the k row; ("hg" "hy") binds the hy row to hg.
;; A row can restrict :vowels and :extensions; () disables that family for the row.
;; Edit :extensions to move a whole family of AZIK suffixes, not hundreds of rules.
;; :extension-extras and :extension-overrides use recipe IDs, so they move too.
;; :extra-rules adds individual rules; :remove and :overrides are explicit exceptions.
;; Duplicate keys are errors unless changed through :overrides.

(
 :physical-rows ("qwertyuiop" "asdfghjkl;" "zxcvbnm,./" "-")
 :vowels ("a" "i" "u" "e" "o")
 :sokuon-key ";"                         ; っ in all three schemes

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
  ("y-azik" "や" "ゐ" "ゆ" "いぇ" "よ")
  ("z" "ざ" "じ" "ず" "ぜ" "ぞ")
  ("zy" "じゃ" "じぃ" "じゅ" "じぇ" "じょ")
 )

 ;; Shared punctuation; digits select candidates and Shift marks kana.
 :punctuation
 (("!" "！") ("," "、") ("-" "ー") ("." "。") ("?" "？") ("[" "「") ("]" "」"))

 :azik
 (
  ;; Original QWERTY AZIK, now maintained locally; no ddskk import.
  :layout-rows ("qwertyuiop" "asdfghjkl;" "zxcvbnm,./" "-")
  :rows
  (
   ""
   "b"
   ("by" "by" :extensions (an an-alt un en on ou ai ei uu))
   ("c" "ch")
   ("cy" "ty" :extensions ())
   "d"
   ("dc" "dh" :vowels ("i" "u") :extensions (in uu))
   ("dy" "dy" :extensions ())
   "f"
   ("fy" "fy" :extensions ())
   "g"
   ("gy" "gy" :extensions (an an-alt un en on ou ai ei uu))
   "h"
   ("hg" "hy" :vowels ("a" "u" "e" "o") :extensions (an an-alt un en on ou ai ei uu))
   ("hy" "hy" :extensions (an-alt en on ou ai ei uu))
   "j"
   ("jy" "jy" :extensions ())
   "k"
   ("kg" "ky" :vowels ("a" "u" "e" "o") :extensions (an an-alt en on ou ai ei uu))
   ("ky" "ky" :extensions (an an-alt un en on ou ai ei uu))
   ("m" "m" :extensions (an-alt in un en on ou ai ei uu))
   ("mg" "my" :vowels ("a" "u" "e" "o") :extensions (an an-alt un en on ou ai ei uu))
   ("my" "my" :extensions (an an-alt un en on ou ai ei uu))
   ("n" "n" :extensions (an-alt in un en on ou ai ei uu))
   ("ng" "ny" :vowels ("a" "u" "e" "o") :extensions (an an-alt un en on ou ai ei uu))
   ("ny" "ny" :extensions (an an-alt un en on ou ai ei uu))
   "p"
   ("pg" "py" :vowels ("a" "u" "e" "o") :extensions (an an-alt un en on ou ai ei uu))
   ("py" "py" :extensions (an an-alt un en on ou ai ei uu))
   "r"
   ("ry" "ry" :extensions (an an-alt un en on ou ai ei uu))
   "s"
   ("sy" "sy" :extensions (an-alt un en on ou ai ei uu))
   "t"
   ("tg" "th" :vowels ("i" "u") :extensions (in uu))
   ("ts" "ts" :extensions ())
   ("ty" "ty" :extensions (an an-alt un en on ou ai ei uu))
   ("v" "v" :extensions (an an-alt in en on ou ai ei))
   ("vy" "vy" :vowels ("a" "u" "e" "o") :extensions ())
   ("w" "w" :extensions (an an-alt in en on ou ai))
   ("wh" "wh" :extensions ())
   ("ws" "wh" :vowels ("o") :extensions ())
   ("x" "sh")
   ("xk" "k" :vowels ("a" "e") :extensions ())
   ("xw" "xw" :extensions ())
   ("xx" "x" :extensions ())
   ("xy" "xy" :extensions ())
   ("y" "y-azik" :extensions (an an-alt un on ou ai uu))
   "z"
   ("zy" "zy" :extensions (an an-alt un en on ou ai ei uu))
  )
  :nasal-keys ("n" "nn" "n'" "q")
  ;; (ID KEY VOWEL APPEND).  an/an-alt are independent bindings of the same sound.
  :extensions
  ((an "n" "a" "ん") (an-alt "z" "a" "ん")
   (in "k" "i" "ん") (un "j" "u" "ん") (en "d" "e" "ん") (on "l" "o" "ん")
   (ou "p" "o" "う") (ai "q" "a" "い") (ei "w" "e" "い") (uu "h" "u" "う"))
  ;; Long-vowel exceptions follow the recipe binding when its key changes.
  :extension-extras ((ou "dc" "どぅー") (ou "tg" "とぅー"))
  :extension-overrides
  ((uu "dc" "でゅー")
   (ou "f" "ふぉー")
   (uu "tg" "てゅー")
   (ou "v" "ゔぉー")
   (on "w" "うぉん")
   (ou "w" "うぉー"))
  ;; Word shortcuts, alternate bindings and symbols are explicit local choices.
  ;; mn and ryk take priority over the nasal combinations excluded above.
  :extra-rules
  (
   ("'" "ー")             ; -
   ("br" "ばら")          ; bara
   ("bt" "びと")          ; bito
   ("bx" "べい")          ; bei
   ("cc" "ちゃ")          ; cha
   ("cf" "ちぇ")          ; che
   ("cv" "ちゃい")        ; chai
   ("cx" "ちぇい")        ; chei
   ("df" "で")            ; de
   ("dg" "だが")          ; daga
   ("dm" "でも")          ; demo
   ("dr" "である")        ; dearu
   ("ds" "です")          ; desu
   ("dt" "だち")          ; dachi
   ("dv" "でん")          ; den
   ("dy" "でぃ")          ; dhi
   ("fm" "ふむ")          ; fumu
   ("fr" "ふる")          ; furu
   ("fs" "ふぁい")        ; fai
   ("gr" "がら")          ; gara
   ("gt" "ごと")          ; goto
   ("hf" "ふ")            ; fu
   ("ht" "ひと")          ; hito
   ("jf" "じゅ")          ; ju
   ("jv" "じゅう")        ; juu
   ("kf" "き")            ; ki
   ("km" "かも")          ; kamo
   ("kr" "から")          ; kara
   ("kt" "こと")          ; koto
   ("kv" "きん")          ; kin
   ("mf" "む")            ; mu
   ("mn" "もの")          ; mono
   ("mr" "まる")          ; maru
   ("ms" "ます")          ; masu
   ("mt" "また")          ; mata
   ("mv" "むん")          ; mun
   ("nb" "ねば")          ; neba
   ("nf" "ぬ")            ; nu
   ("nr" "なる")          ; naru
   ("nt" "にち")          ; nichi
   ("nv" "ぬん")          ; nun
   ("pf" "ぽん")          ; pon
   ("pv" "ぽう")          ; pou
   ("rr" "られ")          ; rare
   ("ryk" "りょく")       ; ryoku
   ("sf" "さい")          ; sai
   ("sm" "しも")          ; shimo
   ("sr" "する")          ; suru
   ("ss" "せい")          ; sei
   ("st" "した")          ; shita
   ("sv" "さい")          ; sai
   ("tb" "たび")          ; tabi
   ("tm" "ため")          ; tame
   ("tr" "たら")          ; tara
   ("tt" "たち")          ; tachi
   ("wf" "わい")          ; wai
   ("wr" "われ")          ; ware
   ("wt" "わた")          ; wata
   ("x'" "'")
   ("x;" ";")
   ("xc" "しゃ")          ; sha
   ("xf" "しぇい")        ; shei
   ("xt" "しゅつ")        ; shutsu
   ("xv" "しゃい")        ; shai
   ("xxh" "←")
   ("xxj" "↓")
   ("xxk" "↑")
   ("xxl" "→")
   ("y^" "↑")
   ("yf" "ゆ")            ; yu
   ("yr" "よる")          ; yoru
   ("yv" "ゆう")          ; yuu
   ("z(" "（")
   ("z)" "）")
   ("z*" "※")
   ("z," "‥")
   ("z-" "〜")
   ("z." "…")
   ("z/" "・")
   ("z;" "゛")
   ("z@" "◎")
   ("z[" "『")
   ("z]" "』")
   ("zc" "ざ")            ; za
   ("zf" "ぜ")            ; ze
   ("zr" "ざる")          ; zaru
   ("zv" "ざい")          ; zai
   ("zx" "ぜい")          ; zei
  )
  ;; Keep the historical table exactly, including reserved/shifted aliases.
  ;; Runtime Shift normalization is unchanged; these are not new case-sensitive bindings.
  :literal-rules
  (
   (":" "：")
   ("kA" "ヵ")            ; xka (small katakana)
   ("kE" "ヶ")            ; xke (small katakana)
   ("wA" "ゎ")            ; xwa
   ("y<" "←")
   ("y>" "→")
   ("z " "\u3000")
   ("z0" "○")
   ("z:" "゜")
   ("zL" "⇒")
   ("z{" "【")
   ("z}" "】")
  )
  :remove ()
  :overrides ()
 )

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
