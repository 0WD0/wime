;;; run.el --- Standalone runtime tests -*- lexical-binding: t; -*-

;; SPDX-License-Identifier: GPL-3.0-or-later

(require 'ert)

(unless (sqlite-available-p)
  (error "Wime tests require Emacs built with SQLite support"))

(let* ((directory (file-name-directory (or load-file-name buffer-file-name)))
       (package-directory (expand-file-name ".." directory)))
  (add-to-list 'load-path package-directory)
  (setq wime-user-database nil
        wime-neural-model-file nil
        wime-grammar-module nil
        native-comp-enable-subr-trampolines nil)
  (dolist (file '("wime-test.el" "wime-edit-test.el" "wime-edit-neural-test.el"
                  "wime-neural-test.el" "wime-japanese-test.el" "wime-memory-test.el"
                  "japanese-generation-test.el"))
    (load (expand-file-name file directory) nil t)))

(ert-run-tests-batch-and-exit)
