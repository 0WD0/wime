;;; fixture.el --- Disposable word database for runtime tests -*- lexical-binding: t; -*-

;; SPDX-License-Identifier: GPL-3.0-or-later

(require 'sqlite)

(defun wime-test--fixture ()
  "Create a small word database, removed when this Emacs exits."
  (let* ((file (make-temp-file "wime-test-" nil ".db"))
         (db (sqlite-open file)))
    (unwind-protect
        (progn
          (sqlite-execute db "CREATE TABLE entry (code TEXT, word TEXT, aux TEXT, weight REAL, tones TEXT)")
          (sqlite-execute db "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
          (dolist (row '(("scheme" "mspy")
                         ("initial_keys" "abcdefghijklmnopqrstuvwxyz")
                         ("final_keys" "abcdefghijklmnopqrstuvwxyz;")))
            (sqlite-execute db "INSERT INTO meta VALUES (?, ?)" row))
          ;; Deliberately small, handwritten entries: these tests exercise the
          ;; input engine rather than the frequency choices of a nightly corpus.
          (dolist (row '(("ni" "你" "r" 100 "3") ("ni" "匿" "k" 10 "4")
                         ("hk" "好" "n" 100 "3") ("hk" "号" "k" 90 "4")
                         ("b;" "并" "k" 100 "4") ("y;" "应" "g" 100 "1")
                         ("wo" "我" "w" 100 "3") ("mf" "们" "r" 100 "5")
                         ("nihk" "你好" "r/n" 1000 "33")
                         ("womf" "我们" "w/r" 1000 "35")
                         ("m;" "明" "o" 100 "2") ("tm" "天" "a" 100 "1")
                         ("m;tm" "明天" "o/a" 1000 "21")
                         ("xmzl" "现在" "w/u" 1000 "44")
                         ("yr" "云" "e" 100 "2") ("tk" "陶" "e" 100 "2")
                         ("qd" "强" "g" 100 "2") ("qd" "墙" "t" 90 "2")
                         ("qd" "枪" "m" 80 "1") ("qd" "抢" "f" 70 "3")
                         ("qd" "腔" "o" 60 "1") ("qd" "羌" "u" 50 "1")
                         ("qd" "呛" "k" 40 "4") ("qd" "蔷" "c" 30 "2")
                         ("qd" "锵" "j" 20 "1") ("qd" "炝" "h" 10 "4")
                         ("jv" "决" "b" 100 "2") ("jt" "决" "b" 100 "2")
                         ("nv" "虐" "h" 100 "4") ("ny" "女" "v" 100 "3")
                         ("gv" "贵" "v" 100 "4") ("jq" "就" "w" 100 "4")
                         ("bmyi" "便宜" "r/b" 100 "42")
                         ("pmyi" "便宜" "r/b" 100 "25")
                         ("bmyi" "编译" "s/y" 90 "14")
                         ("yk" "要" "x" 100 "4") ("yk" "咬" "k" 90 "3")
                         ("woyk" "我要" "w/x" 1000 "34")
                         ("yi" "一" "a" 100 "1") ("ge" "个" "r" 100 "4")
                         ("yige" "一个" "a/r" 1000 "24")
                         ("vo" "桌" "b" 100 "1") ("zi" "子" "v" 100 "3")
                         ("vozi" "桌子" "b/v" 1000 "15")))
            (sqlite-execute db "INSERT INTO entry VALUES (?, ?, ?, ?, ?)" row)))
      (sqlite-close db))
    (add-hook 'kill-emacs-hook (lambda () (delete-file file)))
    file))

;;; fixture.el ends here
