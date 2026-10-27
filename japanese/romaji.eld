;;; Ordinary romaji for wime's original Onishi layout.
;; Layout: https://o24.works/layout/
;; Keys here are logical romaji; wime remaps physical QWERTY keys separately.
;; Each row is (KEYS KANA).  Edit this file directly.
;; Sokuon (っ) uses ; exclusively; nn or n' explicitly enters ん.
;; Comments use ordinary romaji, with n for ん and - for ー.

(
 ;; Punctuation
 ("!" "！")
 ("," "、")
 ("-" "ー")             ; -
 ("." "。")
 (";" "っ")
 ("?" "？")
 ("[" "「")
 ("]" "」")

 ;; a
 ("a" "あ")             ; a

 ;; b
 ("ba" "ば")            ; ba
 ("be" "べ")            ; be
 ("bi" "び")            ; bi
 ("bo" "ぼ")            ; bo
 ("bu" "ぶ")            ; bu
 ("bya" "びゃ")         ; bya
 ("bye" "びぇ")         ; bye
 ("byi" "びぃ")         ; byi
 ("byo" "びょ")         ; byo
 ("byu" "びゅ")         ; byu

 ;; c
 ("cha" "ちゃ")         ; cha
 ("che" "ちぇ")         ; che
 ("chi" "ち")           ; chi
 ("cho" "ちょ")         ; cho
 ("chu" "ちゅ")         ; chu
 ("cya" "ちゃ")         ; cha
 ("cye" "ちぇ")         ; che
 ("cyi" "ちぃ")         ; cyi
 ("cyo" "ちょ")         ; cho
 ("cyu" "ちゅ")         ; chu

 ;; d
 ("da" "だ")            ; da
 ("de" "で")            ; de
 ("dha" "でゃ")         ; dexya
 ("dhe" "でぇ")         ; dexe
 ("dhi" "でぃ")         ; dhi
 ("dho" "でょ")         ; dexyo
 ("dhu" "でゅ")         ; dhu
 ("di" "ぢ")            ; di
 ("do" "ど")            ; do
 ("du" "づ")            ; du
 ("dwa" "どぁ")         ; doxa
 ("dwe" "どぇ")         ; doxe
 ("dwi" "どぃ")         ; doxi
 ("dwo" "どぉ")         ; doxo
 ("dwu" "どぅ")         ; doxu
 ("dya" "ぢゃ")         ; dya
 ("dye" "ぢぇ")         ; dye
 ("dyi" "ぢぃ")         ; dyi
 ("dyo" "ぢょ")         ; dyo
 ("dyu" "ぢゅ")         ; dyu

 ;; e
 ("e" "え")             ; e

 ;; f
 ("fa" "ふぁ")          ; fa
 ("fe" "ふぇ")          ; fe
 ("fi" "ふぃ")          ; fi
 ("fo" "ふぉ")          ; fo
 ("fu" "ふ")            ; fu
 ("fya" "ふゃ")         ; fya
 ("fye" "ふぇ")         ; fe
 ("fyi" "ふぃ")         ; fi
 ("fyo" "ふょ")         ; fyo
 ("fyu" "ふゅ")         ; fyu

 ;; g
 ("ga" "が")            ; ga
 ("ge" "げ")            ; ge
 ("gi" "ぎ")            ; gi
 ("go" "ご")            ; go
 ("gu" "ぐ")            ; gu
 ("gwa" "ぐぁ")         ; guxa
 ("gwe" "ぐぇ")         ; guxe
 ("gwi" "ぐぃ")         ; guxi
 ("gwo" "ぐぉ")         ; guxo
 ("gwu" "ぐ")           ; gu
 ("gya" "ぎゃ")         ; gya
 ("gye" "ぎぇ")         ; gye
 ("gyi" "ぎぃ")         ; gyi
 ("gyo" "ぎょ")         ; gyo
 ("gyu" "ぎゅ")         ; gyu

 ;; h
 ("ha" "は")            ; ha
 ("he" "へ")            ; he
 ("hi" "ひ")            ; hi
 ("ho" "ほ")            ; ho
 ("hu" "ふ")            ; fu
 ("hya" "ひゃ")         ; hya
 ("hye" "ひぇ")         ; hye
 ("hyi" "ひぃ")         ; hyi
 ("hyo" "ひょ")         ; hyo
 ("hyu" "ひゅ")         ; hyu

 ;; i
 ("i" "い")             ; i

 ;; j
 ("ja" "じゃ")          ; ja
 ("je" "じぇ")          ; je
 ("ji" "じ")            ; ji
 ("jo" "じょ")          ; jo
 ("ju" "じゅ")          ; ju
 ("jya" "じゃ")         ; ja
 ("jye" "じぇ")         ; je
 ("jyi" "じぃ")         ; jyi
 ("jyo" "じょ")         ; jo
 ("jyu" "じゅ")         ; ju

 ;; k
 ("ka" "か")            ; ka
 ("ke" "け")            ; ke
 ("ki" "き")            ; ki
 ("ko" "こ")            ; ko
 ("ku" "く")            ; ku
 ("kwa" "くぁ")         ; kuxa
 ("kwe" "くぇ")         ; kuxe
 ("kwi" "くぃ")         ; kuxi
 ("kwo" "くぉ")         ; kuxo
 ("kwu" "く")           ; ku
 ("kya" "きゃ")         ; kya
 ("kye" "きぇ")         ; kye
 ("kyi" "きぃ")         ; kyi
 ("kyo" "きょ")         ; kyo
 ("kyu" "きゅ")         ; kyu

 ;; l
 ("la" "ぁ")            ; xa
 ("le" "ぇ")            ; xe
 ("li" "ぃ")            ; xi
 ("lka" "ヵ")           ; xka (small katakana)
 ("lke" "ヶ")           ; xke (small katakana)
 ("lo" "ぉ")            ; xo
 ("lu" "ぅ")            ; xu
 ("lwa" "ゎ")           ; xwa
 ("lya" "ゃ")           ; xya
 ("lyo" "ょ")           ; xyo
 ("lyu" "ゅ")           ; xyu

 ;; m
 ("ma" "ま")            ; ma
 ("me" "め")            ; me
 ("mi" "み")            ; mi
 ("mo" "も")            ; mo
 ("mu" "む")            ; mu
 ("mya" "みゃ")         ; mya
 ("mye" "みぇ")         ; mye
 ("myi" "みぃ")         ; myi
 ("myo" "みょ")         ; myo
 ("myu" "みゅ")         ; myu

 ;; n
 ("n" "ん")             ; n
 ("n'" "ん")            ; n
 ("na" "な")            ; na
 ("ne" "ね")            ; ne
 ("ni" "に")            ; ni
 ("nn" "ん")            ; n
 ("no" "の")            ; no
 ("nu" "ぬ")            ; nu
 ("nya" "にゃ")         ; nya
 ("nye" "にぇ")         ; nye
 ("nyi" "にぃ")         ; nyi
 ("nyo" "にょ")         ; nyo
 ("nyu" "にゅ")         ; nyu

 ;; o
 ("o" "お")             ; o

 ;; p
 ("pa" "ぱ")            ; pa
 ("pe" "ぺ")            ; pe
 ("pi" "ぴ")            ; pi
 ("po" "ぽ")            ; po
 ("pu" "ぷ")            ; pu
 ("pya" "ぴゃ")         ; pya
 ("pye" "ぴぇ")         ; pye
 ("pyi" "ぴぃ")         ; pyi
 ("pyo" "ぴょ")         ; pyo
 ("pyu" "ぴゅ")         ; pyu

 ;; r
 ("ra" "ら")            ; ra
 ("re" "れ")            ; re
 ("ri" "り")            ; ri
 ("ro" "ろ")            ; ro
 ("ru" "る")            ; ru
 ("rya" "りゃ")         ; rya
 ("rye" "りぇ")         ; rye
 ("ryi" "りぃ")         ; ryi
 ("ryo" "りょ")         ; ryo
 ("ryu" "りゅ")         ; ryu

 ;; s
 ("sa" "さ")            ; sa
 ("se" "せ")            ; se
 ("sha" "しゃ")         ; sha
 ("she" "しぇ")         ; she
 ("shi" "し")           ; shi
 ("sho" "しょ")         ; sho
 ("shu" "しゅ")         ; shu
 ("si" "し")            ; shi
 ("so" "そ")            ; so
 ("su" "す")            ; su
 ("sya" "しゃ")         ; sha
 ("sye" "しぇ")         ; she
 ("syi" "しぃ")         ; syi
 ("syo" "しょ")         ; sho
 ("syu" "しゅ")         ; shu

 ;; t
 ("ta" "た")            ; ta
 ("te" "て")            ; te
 ("tha" "てゃ")         ; texya
 ("the" "てぇ")         ; texe
 ("thi" "てぃ")         ; thi
 ("tho" "てょ")         ; texyo
 ("thu" "てゅ")         ; thu
 ("ti" "ち")            ; chi
 ("to" "と")            ; to
 ("tsa" "つぁ")         ; tsuxa
 ("tse" "つぇ")         ; tsuxe
 ("tsi" "つぃ")         ; tsuxi
 ("tso" "つぉ")         ; tsuxo
 ("tsu" "つ")           ; tsu
 ("tu" "つ")            ; tsu
 ("twa" "とぁ")         ; toxa
 ("twe" "とぇ")         ; toxe
 ("twi" "とぃ")         ; toxi
 ("two" "とぉ")         ; toxo
 ("twu" "とぅ")         ; toxu
 ("tya" "ちゃ")         ; cha
 ("tye" "ちぇ")         ; che
 ("tyi" "ちぃ")         ; cyi
 ("tyo" "ちょ")         ; cho
 ("tyu" "ちゅ")         ; chu

 ;; u
 ("u" "う")             ; u

 ;; v
 ("va" "ゔぁ")          ; va
 ("ve" "ゔぇ")          ; ve
 ("vi" "ゔぃ")          ; vi
 ("vo" "ゔぉ")          ; vo
 ("vu" "ゔ")            ; vu
 ("vya" "ゔゃ")         ; vuxya
 ("vye" "ゔぇ")         ; ve
 ("vyi" "ゔぃ")         ; vi
 ("vyo" "ゔょ")         ; vuxyo
 ("vyu" "ゔゅ")         ; vuxyu

 ;; w
 ("wa" "わ")            ; wa
 ("we" "うぇ")          ; we
 ("wha" "うぁ")         ; uxa
 ("whe" "うぇ")         ; we
 ("whi" "うぃ")         ; wi
 ("who" "うぉ")         ; who
 ("whu" "う")           ; u
 ("wi" "うぃ")          ; wi
 ("wo" "を")            ; wo
 ("wu" "う")            ; u

 ;; x
 ("xa" "ぁ")            ; xa
 ("xe" "ぇ")            ; xe
 ("xi" "ぃ")            ; xi
 ("xka" "ヵ")           ; xka (small katakana)
 ("xke" "ヶ")           ; xke (small katakana)
 ("xo" "ぉ")            ; xo
 ("xu" "ぅ")            ; xu
 ("xwa" "ゎ")           ; xwa
 ("xwe" "ゑ")           ; xwe
 ("xwi" "ゐ")           ; xwi
 ("xya" "ゃ")           ; xya
 ("xyo" "ょ")           ; xyo
 ("xyu" "ゅ")           ; xyu

 ;; y
 ("ya" "や")            ; ya
 ("ye" "いぇ")          ; ye
 ("yi" "い")            ; i
 ("yo" "よ")            ; yo
 ("yu" "ゆ")            ; yu

 ;; z
 ("za" "ざ")            ; za
 ("ze" "ぜ")            ; ze
 ("zi" "じ")            ; ji
 ("zo" "ぞ")            ; zo
 ("zu" "ず")            ; zu
 ("zya" "じゃ")         ; ja
 ("zye" "じぇ")         ; je
 ("zyi" "じぃ")         ; jyi
 ("zyo" "じょ")         ; jo
 ("zyu" "じゅ")         ; ju
)
