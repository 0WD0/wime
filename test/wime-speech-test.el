;;; wime-speech-test.el --- Dictation display and insertion tests -*- lexical-binding: t; -*-
(require 'ert)
(require 'wime-speech)

(ert-deftest wime-speech-final-text-lands-where-dictation-started ()
  (with-temp-buffer
    (insert "上文：")
    (save-excursion (insert "\n后文"))
    (let* ((start (point))
           (wime-speech--session (wime-speech--make-session
                                  :id 7 :marker (point-marker) :overlay (make-overlay start start nil t nil))))
      (wime-speech--reply '((id . 7) (recording . t)))
      (wime-speech--reply '((id . 7) (partial . "空域组织")))
      (should (equal (substring-no-properties
                      (overlay-get (wime-speech--session-overlay wime-speech--session) 'after-string))
                     "●空域组织"))
      (should (equal (buffer-string) "上文：\n后文"))
      (wime-speech--reply '((id . 6) (final . "旧的")))
      (goto-char (point-max))
      (wime-speech--reply '((id . 7) (final . "空域组织的目的")))
      (should (equal (buffer-string) "上文：空域组织的目的\n后文"))
      (should (= (point) (point-max)))
      (should-not wime-speech--session))))

(ert-deftest wime-speech-point-follows-text-dictated-at-point ()
  (with-temp-buffer
    (insert "上文：")
    (let ((wime-speech--session (wime-speech--make-session
                                 :id 1 :marker (point-marker) :overlay (make-overlay (point) (point) nil t nil))))
      (wime-speech--reply '((id . 1) (final . "你好。")))
      (should (equal (buffer-substring (point-min) (point)) "上文：你好。")))))
