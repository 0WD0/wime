;;; wime-datasets.el --- Inspect versioned Wime corpora -*- lexical-binding: t; -*-

;;; Commentary:
;; M-x wime-datasets opens the registry without starting the input method.
;; Queries run asynchronously; only one page lives in the listing buffer.
;; Human reviews are stored separately from immutable training releases.

;;; Code:
(require 'cl-lib)
(require 'json)
(require 'subr-x)
(require 'tabulated-list)
(declare-function evil-set-initial-state "evil-core" (mode state))
(declare-function evil-define-key* "evil-core" (state keymap key def &rest bindings))

(defgroup wime-datasets nil "Inspect and review Wime training data." :group 'applications)
(defcustom wime-datasets-python
  (expand-file-name "wime/environments/datasets/bin/python" (or (getenv "XDG_DATA_HOME") "~/.local/share"))
  "Python environment with the dataset requirements installed." :type 'file)
(defcustom wime-datasets-program
  (let ((source (expand-file-name "neural/dataset_browser.py" (file-name-directory (or load-file-name buffer-file-name)))))
    (if (file-readable-p source) source
      (expand-file-name "wime/datasets/controller/dataset_browser.py" (or (getenv "XDG_DATA_HOME") "~/.local/share"))))
  "Path to the dataset query controller." :type 'file)
(defcustom wime-datasets-page-size 100 "Maximum samples displayed at once." :type 'natnum)
(defvar wime-datasets--catalog nil)
(defvar-local wime-datasets--process nil)
(defvar-local wime-datasets--dataset nil)
(defvar-local wime-datasets--split "dev")
(defvar-local wime-datasets--source "")
(defvar-local wime-datasets--query "")
(defvar-local wime-datasets--tone nil)
(defvar-local wime-datasets--after 0)
(defvar-local wime-datasets--history nil)
(defvar-local wime-datasets--page nil)
(defvar-local wime-datasets--row nil)
(defvar-local wime-datasets--snapshot nil)
(defvar-local wime-datasets--undo nil)

(defun wime-datasets--cancel ()
  (let ((process wime-datasets--process))
    (setq wime-datasets--process nil)
    (when (process-live-p process) (delete-process process))))

(defun wime-datasets--request (request callback)
  "Run REQUEST asynchronously and invoke CALLBACK in the originating buffer."
  (unless (file-executable-p wime-datasets-python)
    (user-error "请先安装 neural/requirements-datasets.txt 对应的数据环境"))
  (unless (file-readable-p wime-datasets-program)
    (user-error "请设置 wime-datasets-program 指向 dataset_browser.py"))
  (wime-datasets--cancel)
  (setq header-line-format "正在查询数据…（切换查询会取消本次请求）")
  (let ((owner (current-buffer))
        (output (generate-new-buffer " *wime-datasets-output*"))
        (errors (generate-new-buffer " *wime-datasets-errors*")))
    (setq wime-datasets--process
          (make-process
           :name "wime-datasets" :buffer output :stderr errors :noquery t
           :connection-type 'pipe :coding 'utf-8-unix
           :command (list wime-datasets-python (file-truename wime-datasets-program))
           :sentinel
           (lambda (process _event)
             (when (memq (process-status process) '(exit signal))
               (unwind-protect
                   (when (buffer-live-p owner)
                     (with-current-buffer owner
                       (when (eq process wime-datasets--process)
                         (setq wime-datasets--process nil)
                         (condition-case err
                             (let ((reply (with-current-buffer output
                                            (json-parse-string (buffer-string) :object-type 'plist
                                                               :array-type 'list :null-object nil :false-object nil))))
                               (when (plist-get reply :error) (error "%s" (plist-get reply :error)))
                               (unless (zerop (process-exit-status process)) (error "数据查询失败"))
                               (funcall callback (plist-get reply :result)))
                           (error
                            (setq header-line-format (error-message-string err))
                            (message "Wime 数据集：%s" (error-message-string err)))))))
                 (kill-buffer output)
                 (kill-buffer errors))))))
    (process-send-string wime-datasets--process (json-serialize request :null-object nil :false-object :false))
    (process-send-eof wime-datasets--process)))

(defun wime-datasets--text (value)
  (replace-regexp-in-string "[\n\r\t]+" " " (or value "")))

(defun wime-datasets--status (review)
  (or (cdr (assoc (plist-get review :status)
                  '(("approved" . "已确认") ("problem" . "有问题") ("skipped" . "跳过")))) "待审查"))

(defun wime-datasets--status-cell (review)
  "Display REVIEW with a symbol, theme color and its full meaning."
  (let* ((status (plist-get review :status))
         (symbol (pcase status ("approved" "✓") ("problem" "!") ("skipped" "–") (_ "·")))
         (face (pcase status ("approved" 'success) ("problem" 'error) (_ 'shadow)))
         (note (plist-get review :note)))
    (propertize symbol 'face face
                'help-echo (concat (wime-datasets--status review)
                                   (if (string-empty-p (or note "")) "" (concat "：" note))))))

(defun wime-datasets--context-cell (context width)
  "Keep the end of CONTEXT adjacent to the target, within WIDTH columns."
  (let* ((text (wime-datasets--text context)) (columns (string-width text)))
    (if (string-empty-p text)
        (propertize "∅" 'face 'shadow 'help-echo "没有已确认的左文")
      (propertize (if (> columns width)
                      (concat "…" (truncate-string-to-width text columns (+ 1 (- columns width))))
                    text)
                  'help-echo context))))

(defun wime-datasets--sample-columns ()
  "Allocate the visible page to context and target; return context width."
  (let* ((available (max 52 (- (window-body-width (get-buffer-window (current-buffer) t)) 16)))
         (context-width (/ (* available 3) 5)))
    (setq tabulated-list-format
          (vector (list "模型左文" context-width nil)
                  (list "目标文本" (- available context-width) nil)
                  '("审" 2 nil) '("行" 8 nil)))
    (tabulated-list-init-header)
    context-width))

(defvar wime-datasets-mode-map
  (let ((map (make-sparse-keymap)))
    (set-keymap-parent map tabulated-list-mode-map)
    (dolist (binding '(("RET" . wime-datasets-visit) ("g" . wime-datasets-refresh)
                       ("]" . wime-datasets-next) ("[" . wime-datasets-previous)
                       ("/" . wime-datasets-search) ("s" . wime-datasets-source)
                       ("t" . wime-datasets-tone) ("S" . wime-datasets-split)
                       ("r" . wime-datasets-random) ("b" . wime-datasets)
                       ("a" . wime-datasets-approve) ("x" . wime-datasets-problem)
                       ("k" . wime-datasets-skip) ("u" . wime-datasets-undo)
                       ("i" . wime-datasets-info)))
      (define-key map (kbd (car binding)) (cdr binding)))
    map))

(define-derived-mode wime-datasets-mode tabulated-list-mode "Wime 数据集"
  "Browse corpus versions or a bounded page of samples."
  (setq-local tabulated-list-padding 1)
  (setq-local tabulated-list-use-header-line nil)
  (setq-local revert-buffer-function (lambda (&rest _) (wime-datasets-refresh)))
  (add-hook 'kill-buffer-hook #'wime-datasets--cancel nil t))

(defun wime-datasets--show-catalog (entries)
  (setq wime-datasets--catalog entries)
  (setq tabulated-list-format [("数据集" 43 t) ("类型" 13 t) ("状态" 12 t) ("训练样本" 12 t) ("标注方案" 30 t)])
  (setq tabulated-list-entries
        (mapcar (lambda (entry)
                  (let* ((train (plist-get (plist-get entry :files) :train)) (count (plist-get train :rows)))
                    (list (plist-get entry :id)
                          (vector (plist-get entry :id) (plist-get entry :kind) (plist-get entry :status)
                                  (if count (number-to-string count) "—") (plist-get entry :annotation))))) entries))
  (tabulated-list-init-header)
  (setq header-line-format "RET 浏览样本 · i 来源/质量说明 · g 刷新")
  (tabulated-list-print t))

;;;###autoload
(defun wime-datasets ()
  "Open the dataset registry independently of the input method."
  (interactive)
  (pop-to-buffer (get-buffer-create "*Wime 数据集*"))
  (wime-datasets-mode)
  (wime-datasets--request '(:op "catalog") #'wime-datasets--show-catalog))

(defun wime-datasets--entry ()
  (let ((id (or wime-datasets--dataset (tabulated-list-get-id))))
    (or (cl-find id wime-datasets--catalog :key (lambda (entry) (plist-get entry :id)) :test #'equal)
        (user-error "请选择一个数据集"))))

(defun wime-datasets-open (dataset &optional split)
  "Browse DATASET's SPLIT, defaulting to dev when available."
  (let* ((entry (cl-find dataset wime-datasets--catalog :key (lambda (e) (plist-get e :id)) :test #'equal))
         (splits (plist-get entry :browsable)))
    (unless splits (user-error "这个条目没有可分页浏览的样本文件，请按 i 查看信息"))
    (pop-to-buffer (get-buffer-create (format "*Wime 样本 %s*" dataset)))
    (wime-datasets-mode)
    (setq wime-datasets--dataset dataset
          wime-datasets--split (or split (and (member "dev" splits) "dev") (car splits)))
    (wime-datasets-refresh)))

(defun wime-datasets--show-page (page)
  (setq wime-datasets--page page wime-datasets--after (plist-get page :after)
        wime-datasets--snapshot (plist-get page :snapshot))
  (let ((context-width (wime-datasets--sample-columns)))
    (setq tabulated-list-entries
          (mapcar (lambda (row)
                    (list (plist-get row :_row)
                          (vector (wime-datasets--context-cell (plist-get row :context) context-width)
                                  (propertize (wime-datasets--text (car (plist-get row :gold)))
                                              'help-echo (car (plist-get row :gold)))
                                  (wime-datasets--status-cell (plist-get row :review))
                                  (number-to-string (plist-get row :_row))))) (plist-get page :rows))))
  (setq header-line-format
        (format "%s / %s · 匹配 %s 条 · 来源 %s · 声调 %s · 查询 %s · [] 翻页 / 搜索 s 来源 t 声调 r 随机 RET 详情 a 确认 x 问题"
                wime-datasets--dataset wime-datasets--split (plist-get page :total)
                (if (string-empty-p wime-datasets--source) "全部" wime-datasets--source)
                (or wime-datasets--tone "全部") wime-datasets--query))
  (tabulated-list-print t))

(defun wime-datasets-refresh (&optional random)
  (interactive)
  (if (not wime-datasets--dataset)
      (wime-datasets--request '(:op "catalog") #'wime-datasets--show-catalog)
    (wime-datasets--request
     (list :op "page" :dataset wime-datasets--dataset :split wime-datasets--split
           :after wime-datasets--after :limit wime-datasets-page-size :source wime-datasets--source
           :query wime-datasets--query :tone wime-datasets--tone :random (if random t :false))
     #'wime-datasets--show-page)))

(defun wime-datasets-next ()
  (interactive)
  (unless (plist-get wime-datasets--page :has_next) (user-error "已经是最后一页"))
  (push wime-datasets--after wime-datasets--history)
  (setq wime-datasets--after (plist-get wime-datasets--page :next))
  (wime-datasets-refresh))

(defun wime-datasets-previous ()
  (interactive)
  (unless wime-datasets--history (user-error "没有上一页"))
  (setq wime-datasets--after (pop wime-datasets--history))
  (wime-datasets-refresh))

(defun wime-datasets--reset-page ()
  (setq wime-datasets--after 0 wime-datasets--history nil)
  (wime-datasets-refresh))

(defun wime-datasets-search (query)
  (interactive (list (read-string "搜索目标文本或模型左文：" wime-datasets--query)))
  (setq wime-datasets--query query) (wime-datasets--reset-page))
(defun wime-datasets-source (source)
  (interactive (list (completing-read "来源（空为全部）：" (mapcar #'car (plist-get wime-datasets--page :sources)) nil t)))
  (setq wime-datasets--source source) (wime-datasets--reset-page))
(defun wime-datasets-tone (tone)
  (interactive (list (completing-read "声调（5 为轻声；空为全部）：" '("1" "2" "3" "4" "5") nil t)))
  (setq wime-datasets--tone (unless (string-empty-p tone) (string-to-number tone)))
  (wime-datasets--reset-page))
(defun wime-datasets-split (split)
  (interactive (list (completing-read "数据划分：" (plist-get (wime-datasets--entry) :browsable) nil t)))
  (setq wime-datasets--split split wime-datasets--source "" wime-datasets--tone nil)
  (wime-datasets--reset-page))
(defun wime-datasets-random ()
  (interactive)
  (push wime-datasets--after wime-datasets--history)
  (wime-datasets-refresh t))

(defun wime-datasets-info ()
  (interactive)
  (let ((entry (wime-datasets--entry)) (page wime-datasets--page))
    (with-help-window "*Wime 数据集说明*"
      (princ (format "%s\n\n%s\n\n" (plist-get entry :id) (plist-get entry :description)))
      (princ (format "上游数据：%s\n标注方案：%s\n\n质量信息：%S\n\n已知问题：\n"
                     (plist-get entry :parents) (plist-get entry :annotation) (plist-get entry :quality)))
      (dolist (issue (plist-get entry :known_issues)) (princ (concat "• " issue "\n")))
      (when page
        (princ "\n审查符号：· 待审查　✓ 已确认　! 有问题　– 跳过\n来源和逐字读音见 RET 详情；悬停可查看完整左文及审查说明。\n")
        (princ (format "\n当前划分文档数：%s\n来源与样本数：%S\n" (or (plist-get page :documents) "未记录")
                       (plist-get page :sources)))))))

(defun wime-datasets--show-detail (detail)
  (let* ((row (plist-get detail :row)) (text (car (plist-get row :gold)))
         (pinyin (plist-get row :pinyin)) (tones (plist-get row :tones))
         (inhibit-read-only t))
    (erase-buffer)
    (setq wime-datasets--row (plist-get row :_row))
    (insert (format "%s / %s / 第 %s 条\n\n目标：%s\n实际按键：%s\n"
                    wime-datasets--dataset wime-datasets--split wime-datasets--row text (plist-get row :input)))
    (insert (format "\n模型看到的左文：\n%s\n\n标注器看到的上下文：\n%s\n"
                    (if (string-empty-p (or (plist-get row :context) "")) "（空上下文）" (plist-get row :context))
                    (or (plist-get detail :annotation_text) "（历史数据未保存可定位的标注窗口）")))
    (when-let* ((sentence (plist-get detail :sentence_text)) (start (plist-get detail :sentence_target)))
      (insert (format "\n所在句子（目标前 %s 字、后 %s 字%s；上屏边界尚未人工确认）：\n"
                      (plist-get detail :sentence_prefix) (plist-get detail :sentence_suffix)
                      (if (plist-get detail :sentence_clipped) "，长句已截取" "")))
      (insert (substring sentence 0 start)
              (propertize (substring sentence start (+ start (length text))) 'face 'highlight)
              (substring sentence (+ start (length text))) "\n"))
    (insert "\n逐字标注（5 为轻声）：\n")
    (when (stringp pinyin) (setq pinyin (split-string pinyin)))
    (dotimes (i (length text))
      (let ((note (cl-find i (plist-get row :reading_notes) :key (lambda (n) (plist-get n :index)))))
        (insert (format "%s  %s%s  %s%s\n" (substring text i (1+ i)) (or (nth i pinyin) "?")
                        (or (nth i tones) "（声调未标注）")
                        (if (plist-get note :before_tone) (format "变调前 %s；" (plist-get note :before_tone)) "")
                        (mapconcat #'identity (plist-get note :rules) "；")))))
    (insert (format "\n来源：%s\n原始记录定位：%s\n文档 ID：%s\n清洗操作：%S\n"
                    (or (plist-get row :source) "未提供")
                    (or (plist-get detail :source_document) "未提供") (or (plist-get row :document) "未提供")
                    (plist-get detail :cleaning)))
    (when (plist-get detail :document_excerpt)
      (insert (format "\n清洗后文档片段（全文 %s 字）：\n%s\n"
                      (plist-get detail :document_characters) (plist-get detail :document_excerpt))))
    (insert (format "\n审查：%s\n%s\n" (wime-datasets--status (plist-get row :review))
                    (plist-get (plist-get row :review) :note)))
    (setq header-line-format "a 确认 · x 记录问题 · k 跳过 · u 撤销上次审查 · q 关闭；反馈独立保存，不改训练语料")
    (goto-char (point-min))))

(defun wime-datasets-detail (number)
  (let ((dataset wime-datasets--dataset) (split wime-datasets--split) (snapshot wime-datasets--snapshot))
    (pop-to-buffer (get-buffer-create (format "*Wime 样本详情 %s*" dataset)))
    (unless (derived-mode-p 'wime-datasets-detail-mode) (wime-datasets-detail-mode))
    (setq wime-datasets--dataset dataset wime-datasets--split split wime-datasets--snapshot snapshot
          wime-datasets--row number)
    (wime-datasets--request (list :op "detail" :dataset dataset :split split :row number)
                            #'wime-datasets--show-detail)))

(defun wime-datasets-visit ()
  (interactive)
  (if wime-datasets--dataset
      (wime-datasets-detail (or (tabulated-list-get-id) (user-error "请选择样本")))
    (wime-datasets-open (or (tabulated-list-get-id) (user-error "请选择数据集")))))

(defvar wime-datasets-detail-mode-map
  (let ((map (make-sparse-keymap)))
    (set-keymap-parent map special-mode-map)
    (define-key map (kbd "a") #'wime-datasets-approve)
    (define-key map (kbd "x") #'wime-datasets-problem)
    (define-key map (kbd "k") #'wime-datasets-skip)
    (define-key map (kbd "u") #'wime-datasets-undo)
    map))
(define-derived-mode wime-datasets-detail-mode special-mode "Wime 样本详情"
  "Inspect the actual model context, annotation context and human feedback."
  (add-hook 'kill-buffer-hook #'wime-datasets--cancel nil t))

(defun wime-datasets--review (status note)
  (let ((row (or wime-datasets--row (tabulated-list-get-id) (user-error "请选择样本"))))
    (wime-datasets--request
     (list :op "review" :dataset wime-datasets--dataset :split wime-datasets--split
           :snapshot wime-datasets--snapshot :row row :status status :note note)
     (lambda (reply)
       (push (plist-get reply :event) wime-datasets--undo)
       (if (derived-mode-p 'wime-datasets-detail-mode) (wime-datasets-detail row) (wime-datasets-refresh))))))
(defun wime-datasets-approve () (interactive) (wime-datasets--review "approved" ""))
(defun wime-datasets-problem (note)
  (interactive (list (read-string "问题（读音、清洗、上屏边界或上下文）：")))
  (when (string-empty-p note) (user-error "请描述问题"))
  (wime-datasets--review "problem" note))
(defun wime-datasets-skip () (interactive) (wime-datasets--review "skipped" ""))
(defun wime-datasets-undo ()
  (interactive)
  (unless wime-datasets--undo (user-error "本缓冲区没有待撤销的审查"))
  (let ((event (car wime-datasets--undo)))
    (wime-datasets--request
     (list :op "undo" :event event)
     (lambda (_reply)
       (pop wime-datasets--undo)
       (if (derived-mode-p 'wime-datasets-detail-mode) (wime-datasets-detail wime-datasets--row)
         (wime-datasets-refresh))))))

(with-eval-after-load 'evil
  (dolist (mode '(wime-datasets-mode wime-datasets-detail-mode))
    (evil-set-initial-state mode 'motion))
  ;; Keep j/k for navigation in Evil; K records an explicitly skipped sample.
  (evil-define-key* '(normal motion) wime-datasets-mode-map
    (kbd "RET") #'wime-datasets-visit
    (kbd "]") #'wime-datasets-next (kbd "[") #'wime-datasets-previous
    (kbd "/") #'wime-datasets-search (kbd "s") #'wime-datasets-source
    (kbd "t") #'wime-datasets-tone (kbd "S") #'wime-datasets-split
    (kbd "r") #'wime-datasets-random (kbd "b") #'wime-datasets
    (kbd "a") #'wime-datasets-approve (kbd "x") #'wime-datasets-problem
    (kbd "K") #'wime-datasets-skip (kbd "u") #'wime-datasets-undo
    (kbd "i") #'wime-datasets-info (kbd "g r") #'wime-datasets-refresh
    (kbd "q") #'quit-window)
  (evil-define-key* '(normal motion) wime-datasets-detail-mode-map
    (kbd "a") #'wime-datasets-approve (kbd "x") #'wime-datasets-problem
    (kbd "K") #'wime-datasets-skip (kbd "u") #'wime-datasets-undo
    (kbd "q") #'quit-window))

(provide 'wime-datasets)
;;; wime-datasets.el ends here
