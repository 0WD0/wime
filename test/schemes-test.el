;;; schemes-test.el --- Checks against built layout databases -*- lexical-binding: t; -*-
;; SPDX-License-Identifier: GPL-3.0-or-later

(require 'ert)
(require 'wime-test)

(ert-deftest wime-common-schemes-use-their-own-zero-initials-finals-and-aliases ()
  (dolist (case '(("wanxiang-ziranma.db" "nihk" "udpn" "an" "ly")
                  ("wanxiang-flypy.db" "nihc" "ulpb" "an" "lk")
                  ("wanxiang-sogou.db" "nihk" "udpn" "oj" "l;")))
    (wime-test--with-database (car case)
      (cl-loop for code in (cdr case) for word in '("你好" "双拼" "安" "零")
               do (should (member word (wime--candidates code))))
      (should (member "略" (wime--candidates "lt")))
      (should (member "决" (wime--candidates "jt")))
      (should (equal (caar (sqlite-select (wime--database)
                                         "SELECT value FROM meta WHERE key='syllable_encoding'"))
                     "wime-syllable-features-v1")))))

;;; schemes-test.el ends here
