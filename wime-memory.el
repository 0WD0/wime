;;; wime-memory.el --- Manage Wime learned readings and history -*- lexical-binding: t; -*-

;;; Commentary:
;; Manage the live input method's user database.  Every mutation checks the
;; displayed row, runs in a transaction, and returns an inverse-operation
;; record.  History and learned preferences have independent lifetimes.

;;; Code:

(require 'wime)
(require 'tabulated-list)
(require 'thingatpt)

(declare-function evil-set-initial-state "evil-core" (mode state))
(declare-function evil-define-key* "evil-core" (state keymap key def &rest bindings))

(cl-defstruct (wime-memory-change (:constructor wime-memory--make-change))
  database kind before after)

(defun wime-memory--database ()
  "Return the configured database, or explain how to enable learning."
  (or (wime--user-db)
      (user-error "尚未启用学习，请先设置 wime-user-database")))

(defun wime-memory--file ()
  (and wime-user-database (expand-file-name wime-user-database)))

(defun wime-memory-rows (kind &optional query limit within)
  "Return KIND's newest rows matching literal QUERY, up to LIMIT.
KIND is `memory' or `history'.  Memory rows are (CODE TEXT AUX COUNT UPDATED);
history rows are (ID TIME INPUT KEYS TEXT DEFAULT CONTEXT).
With WITHIN, also find learned words contained in QUERY."
  (let ((db (wime-memory--database)) (query (or query ""))
        (limit (or limit 200)))
    (unless (and (integerp limit) (> limit 0)) (error "Invalid row limit"))
    (pcase kind
      ('memory
       (sqlite-select db
                      (concat "SELECT code, text, aux, count, updated FROM memory "
                              "WHERE instr(code, ?) > 0 OR instr(text, ?) > 0 "
                              "OR (? AND instr(?, text) > 0) "
                              "ORDER BY updated DESC, code, text LIMIT ?")
                      (list query query (if within 1 0) query limit)))
      ('history
       (sqlite-select db
                      (concat "SELECT rowid, time, input, keys, text, default_reading, context FROM commits "
                              "WHERE instr(input, ?) > 0 OR instr(keys, ?) > 0 OR instr(text, ?) > 0 "
                              "OR instr(default_reading, ?) > 0 OR instr(context, ?) > 0 "
                              "ORDER BY time DESC, rowid DESC LIMIT ?")
                      (list query query query query query limit)))
      (_ (error "Unknown record kind: %S" kind)))))

(defun wime-memory--id (kind row)
  (if (eq kind 'memory) (seq-take row 2) (car row)))

(defun wime-memory--row (db kind row)
  "Read the current record with ROW's identity."
  (car (pcase kind
         ('memory
          (sqlite-select db "SELECT code, text, aux, count, updated FROM memory WHERE code = ? AND text = ?"
                         (seq-take row 2)))
         ('history
          (sqlite-select db "SELECT rowid, time, input, keys, text, default_reading, context FROM commits WHERE rowid = ?"
                         (list (car row))))
         (_ (error "Unknown record kind: %S" kind)))))

(defun wime-memory--refresh-input ()
  "Refresh live Chinese compositions after changing their preferences."
  (dolist (buffer (buffer-list))
    (with-current-buffer buffer
      (when (and (equal current-input-method "wime") wime-mode (wime--active-p))
        (setq wime--search-cache nil)
        (condition-case err
            (wime--refresh)
          (error (message "学习记录已保存；%s 的候选刷新失败：%s"
                          (buffer-name) (error-message-string err))))))))

(defun wime-memory--change (kind before after)
  "Atomically replace exact BEFORE rows with AFTER rows and return a change.
Reject stale records and occupied destinations, including during undo."
  (let ((db (wime-memory--database)) committed)
    (sqlite-execute db "BEGIN IMMEDIATE")
    (unwind-protect
        (progn
          (dolist (row before)
            (unless (equal row (wime-memory--row db kind row))
              (user-error "记录已发生变化，请刷新后重试")))
          (dolist (row before)
            (pcase kind
              ('memory (sqlite-execute db "DELETE FROM memory WHERE code = ? AND text = ?"
                                       (seq-take row 2)))
              ('history (sqlite-execute db "DELETE FROM commits WHERE rowid = ?" (list (car row))))))
          (dolist (row after)
            (when (wime-memory--row db kind row)
              (user-error "目标记录已存在，操作未覆盖现有数据"))
            (pcase kind
              ('memory (sqlite-execute db "INSERT INTO memory (code, text, aux, count, updated) VALUES (?, ?, ?, ?, ?)" row))
              ('history (sqlite-execute db "INSERT INTO commits (rowid, time, input, keys, text, default_reading, context) VALUES (?, ?, ?, ?, ?, ?, ?)" row))))
          (sqlite-execute db "COMMIT")
          (setq committed t))
      (unless committed (sqlite-execute db "ROLLBACK")))
    (when (eq kind 'memory)
      (wime--reload-memory db)
      (wime-memory--refresh-input))
    (wime-memory--make-change :database (wime-memory--file) :kind kind
                              :before (copy-tree before) :after (copy-tree after))))

(defun wime-memory-delete-records (kind rows)
  "Delete exact ROWS of KIND, returning a reversible change."
  (unless rows (user-error "没有选中的记录"))
  (wime-memory--change kind (delete-dups (copy-tree rows)) nil))

(defun wime-memory-revert-change (change)
  "Undo CHANGE without overwriting records learned or edited since it."
  (unless (equal (wime-memory-change-database change) (wime-memory--file))
    (user-error "学习数据库已切换，请回到原数据库后撤销"))
  (wime-memory--change (wime-memory-change-kind change)
                       (wime-memory-change-after change) (wime-memory-change-before change)))

(defun wime-memory--validated-aux (code text)
  "Validate Chinese TEXT's complete double-pinyin CODE; derive auxiliary keys."
  (let ((units (wime--parse code)) (db (wime--database)))
    (unless (and (<= 2 (length text) 16) (= (length units) (length text))
                 (string-match-p "\\`[㐀-鿿]+\\'" text)
                 (equal code (wime--full-code units))
                 (cl-every (lambda (unit) (and (= (length (car unit)) 2)
                                               (string-empty-p (cdr unit)))) units))
      (user-error "请输入 2–16 个汉字及其完整双拼编码（不含辅助码）"))
    (unless db (user-error "当前词库不可用，无法校验读音"))
    (let ((word (car (sqlite-select db "SELECT aux FROM entry WHERE code = ? AND word = ? ORDER BY weight DESC LIMIT 1"
                                    (list code text)))))
      (if word (car word)
        (mapconcat
         (lambda (i)
           (let ((entry (car (sqlite-select db "SELECT aux FROM entry WHERE code = ? AND word = ? ORDER BY weight DESC LIMIT 1"
                                           (list (car (nth i units)) (substring text i (1+ i)))))))
             (unless entry (user-error "词库中没有 %s 对应 %s 的读音" (car (nth i units)) (substring text i (1+ i))))
             (car entry)))
         (number-sequence 0 (1- (length text))) "/")))))

(defun wime-memory-update-record (row code text)
  "Replace learned ROW with validated CODE/TEXT and newly derived auxiliary keys.
A changed reading starts a new learning count.  Undo restores the old count."
  (when (equal (list code text) (seq-take row 2))
    (user-error "词语和编码没有变化"))
  (wime-memory--change 'memory (list row)
                       (list (list code text (wime-memory--validated-aux code text) 1 (float-time)))))

;;; Management buffer

(defcustom wime-memory-page-size 200
  "Number of records loaded initially and by `wime-memory-more'."
  :type 'natnum :group 'wime)

(defvar-local wime-memory--kind 'memory)
(defvar-local wime-memory--query "")
(defvar-local wime-memory--within nil)
(defvar-local wime-memory--limit 200)
(defvar-local wime-memory--rows nil)
(defvar-local wime-memory--database-file nil)
(defvar-local wime-memory--dictionary-file nil)
(defvar-local wime-memory--undo nil)

(defvar wime-memory-mode-map
  (let ((map (make-sparse-keymap)))
    (set-keymap-parent map tabulated-list-mode-map)
    (dolist (binding '(("RET" . wime-memory-details) ("e" . wime-memory-edit)
                       ("m" . wime-memory-mark) ("u" . wime-memory-unmark)
                       ("d" . wime-memory-delete) ("U" . wime-memory-undo)
                       ("C-/" . wime-memory-undo) ("/" . wime-memory-filter)
                       ("g" . wime-memory-refresh) ("+" . wime-memory-more) ("h" . wime-history)
                       ("l" . wime-memory) ("?" . wime-memory-help)))
      (define-key map (kbd (car binding)) (cdr binding)))
    map))

(define-derived-mode wime-memory-mode tabulated-list-mode "Wime 学习"
  "Manage learned readings and commit history.
Use `wime-memory-help' for commands and their effects."
  (setq-local tabulated-list-padding 2)
  (setq-local revert-buffer-function (lambda (&rest _) (wime-memory-refresh))))

;; Keep Evil optional and support either package load order.  Preserve its
;; movement, search and column-navigation keys; use separate view commands.
(with-eval-after-load 'evil
  (evil-set-initial-state 'wime-memory-mode 'normal)
  (evil-define-key* '(normal motion) wime-memory-mode-map
    (kbd "RET") #'wime-memory-details
    "e" #'wime-memory-edit
    "m" #'wime-memory-mark
    "M" #'wime-memory-unmark
    "d" #'wime-memory-delete
    "u" #'wime-memory-undo
    "U" #'wime-memory-undo
    (kbd "C-/") #'wime-memory-undo
    "s" #'wime-memory-filter
    "+" #'wime-memory-more
    "H" #'wime-history
    "L" #'wime-memory
    "gr" #'wime-memory-refresh
    "g?" #'wime-memory-help
    "q" #'quit-window))

(defun wime-memory--check-database (&optional dictionary)
  (unless (equal wime-memory--database-file (wime-memory--file))
    (user-error "学习数据库已切换，请重新打开 M-x wime-memory"))
  (when (and dictionary (not (equal wime-memory--dictionary-file wime-database-file)))
    (user-error "输入方案已切换，请重新打开列表后修改读音")))

(defun wime-memory--time (time)
  (if (numberp time) (format-time-string "%Y-%m-%d %H:%M:%S" (seconds-to-time time)) "—"))

(defun wime-memory--cell (text)
  "Render stored text on one list row, without interpreted properties."
  (replace-regexp-in-string "[[:cntrl:]]" " " (substring-no-properties (or text ""))))

(defun wime-memory--marked-ids ()
  (save-excursion
    (goto-char (point-min))
    (let (ids)
      (while (not (eobp))
        (when (eq (char-after) ?*) (push (tabulated-list-get-id) ids))
        (forward-line 1))
      (nreverse ids))))

(defun wime-memory-refresh ()
  "Reload records, retaining visible selection marks."
  (interactive)
  (wime-memory--check-database)
  (let* ((marked (wime-memory--marked-ids))
         (rows (wime-memory-rows wime-memory--kind wime-memory--query
                                 (1+ wime-memory--limit) wime-memory--within))
         (more (> (length rows) wime-memory--limit)))
    (setq wime-memory--rows
          (mapcar (lambda (row) (cons (wime-memory--id wime-memory--kind row) row))
                  (seq-take rows wime-memory--limit))
          tabulated-list-entries
          (mapcar
           (lambda (entry)
             (let ((row (cdr entry)))
               (list (car entry)
                     (if (eq wime-memory--kind 'memory)
                         (vector (wime-memory--cell (nth 1 row)) (wime-memory--cell (nth 0 row))
                                 (number-to-string (or (nth 3 row) 0))
                                 (wime-memory--time (nth 4 row)) (wime-memory--cell (nth 2 row)))
                       (vector (wime-memory--time (nth 1 row)) (wime-memory--cell (nth 4 row))
                               (wime-memory--cell (nth 5 row)) (wime-memory--cell (nth 2 row))
                               (wime-memory--cell (nth 6 row)))))))
           wime-memory--rows)
          mode-line-process (format " [%d%s%s]" (length tabulated-list-entries)
                                    (if more "+" "")
                                    (if (string-empty-p wime-memory--query) "" " 筛选")))
    (tabulated-list-print t)
    (save-excursion
      (goto-char (point-min))
      (while (not (eobp))
        (when (member (tabulated-list-get-id) marked) (tabulated-list-put-tag "*"))
        (forward-line 1)))
    (message (substitute-command-keys "%s：%d 条%s · \\[wime-memory-filter] 筛选 · \\[wime-memory-details] 详情 · \\[wime-memory-mark]/\\[wime-memory-unmark] 标记 · \\[wime-memory-delete] 删除 · \\[wime-memory-undo] 撤销 · \\[wime-memory-help] 帮助")
             (if (eq wime-memory--kind 'memory) "学习偏好" "上屏历史")
             (length tabulated-list-entries) (if more "（+ 加载更多）" ""))))

(defun wime-memory--open (kind query &optional within)
  (wime-memory--database)
  (let ((file (wime-memory--file))
        (buffer (get-buffer-create "*Wime 学习*")))
    (with-current-buffer buffer
      (unless (and (derived-mode-p 'wime-memory-mode)
                   (equal wime-memory--database-file file))
        (wime-memory-mode)
        (setq wime-memory--database-file file))
      (setq wime-memory--kind kind wime-memory--query (or query "")
            wime-memory--within within wime-memory--dictionary-file wime-database-file
            wime-memory--limit (max 1 wime-memory-page-size)
            list-buffers-directory file tabulated-list-sort-key nil
            mode-name (if (eq kind 'memory) "Wime 学习偏好" "Wime 上屏历史")
            tabulated-list-format
            (if (eq kind 'memory)
                [("词语" 24 t) ("双拼编码" 22 t)
                 ("学习次数" 8 (lambda (a b) (< (string-to-number (aref (cadr a) 2))
                                                (string-to-number (aref (cadr b) 2)))))
                 ("最近学习" 19 t) ("辅助码" 0 t)]
              [("上屏时间" 19 t) ("最终文本" 26 t) ("最初候选" 26 t)
               ("原始按键" 22 t) ("前文" 0 t)]))
      ;; Marks belong to one query and view; changing either clears them.
      (let ((inhibit-read-only t)) (erase-buffer))
      (tabulated-list-init-header)
      (wime-memory-refresh))
    (pop-to-buffer buffer)))

;;;###autoload
(defun wime-memory (&optional query)
  "Manage learned Chinese readings, optionally filtered by QUERY."
  (interactive)
  (wime-memory--open 'memory query))

;;;###autoload
(defun wime-history (&optional query)
  "Browse commit history, optionally filtered by QUERY.
Deleting history leaves learned preferences intact."
  (interactive)
  (wime-memory--open 'history query))

;;;###autoload
(defun wime-memory-at-point ()
  "Find learned words in the region, current candidate, or word at point."
  (interactive)
  (let ((text (cond ((use-region-p) (buffer-substring-no-properties (region-beginning) (region-end)))
                    ((and (equal current-input-method "wime") (wime--active-p)) (car wime--candidates))
                    (t (thing-at-point 'word t)))))
    (unless (and text (not (string-empty-p text)))
      (user-error "请选中要查找的词语，或把光标放在词语上"))
    (wime-memory--open 'memory text t)))

(defun wime-memory-filter (query)
  "Filter the current view by literal QUERY; empty QUERY shows all rows."
  (interactive (list (read-string "筛选文字或编码（空白显示全部）：" wime-memory--query)))
  (wime-memory--check-database)
  (wime-memory--open wime-memory--kind query))

(defun wime-memory-more ()
  "Load the next page without losing selection marks."
  (interactive)
  (cl-incf wime-memory--limit (max 1 wime-memory-page-size))
  (wime-memory-refresh))

(defun wime-memory--selected-row ()
  (or (cdr (assoc (tabulated-list-get-id) wime-memory--rows))
      (user-error "当前行没有记录")))

(defun wime-memory-mark ()
  "Mark this row for deletion and move down."
  (interactive)
  (wime-memory--selected-row)
  (tabulated-list-put-tag "*" t))

(defun wime-memory-unmark ()
  "Unmark this row and move down."
  (interactive)
  (tabulated-list-put-tag " " t))

(defun wime-memory-delete ()
  "Delete marked rows, or this row when none are marked.
Use `wime-memory-undo' to restore the deleted records."
  (interactive)
  (wime-memory--check-database)
  (let* ((marked (wime-memory--marked-ids))
         (rows (if marked (mapcar (lambda (id) (cdr (assoc id wime-memory--rows))) marked)
                 (list (wime-memory--selected-row)))))
    (push (wime-memory-delete-records wime-memory--kind rows) wime-memory--undo)
    (wime-memory-refresh)
    (message (substitute-command-keys "已删除 %d 条%s；\\[wime-memory-undo] 撤销%s") (length rows)
             (if (eq wime-memory--kind 'memory) "学习偏好" "上屏历史")
             (if (eq wime-memory--kind 'history) "（学习偏好保留）" "（上屏历史保留）"))))

(defun wime-memory-edit ()
  "Edit the current learned word and its reading, validating against the dictionary."
  (interactive)
  (wime-memory--check-database t)
  (unless (eq wime-memory--kind 'memory) (user-error "上屏历史保留原始输入；请在学习偏好视图修改词语"))
  (let* ((row (wime-memory--selected-row))
         (text (read-string "词语：" (nth 1 row)))
         (code (read-string "完整双拼编码：" (nth 0 row))))
    (wime-memory--check-database t)
    (push (wime-memory-update-record row code text) wime-memory--undo)
    (wime-memory-refresh)
    (message (substitute-command-keys "已修改学习偏好并重新校验辅助码；\\[wime-memory-undo] 撤销"))))

(defun wime-memory-undo ()
  "Undo the last successful management action in this buffer."
  (interactive)
  (wime-memory--check-database)
  (unless wime-memory--undo (user-error "没有可撤销的管理操作"))
  (wime-memory-revert-change (car wime-memory--undo))
  (pop wime-memory--undo)
  (wime-memory-refresh)
  (message "已撤销上一次管理操作"))

(defun wime-memory-details ()
  "Show the complete record, including unabridged input and context."
  (interactive)
  (let ((row (wime-memory--selected-row)) (kind wime-memory--kind)
        (file wime-memory--database-file))
    (with-help-window "*Wime 记录详情*"
      (princ (if (eq kind 'memory)
                 (format "学习偏好\n\n词语：%s\n双拼编码：%s\n辅助码：%s\n学习次数：%s\n最近学习：%s\n\n删除偏好会让输入法忘记它，上屏历史仍保留。\n"
                         (nth 1 row) (nth 0 row) (nth 2 row) (nth 3 row) (wime-memory--time (nth 4 row)))
               (format "上屏历史 · %s\n\n原始按键：%s\n完整编码：%s\n\n最初候选：\n%s\n\n最终文本：\n%s\n\n前文：\n%s\n\n删除这条历史不会删除已经学到的偏好。\n"
                       (wime-memory--time (nth 1 row)) (nth 2 row) (nth 3 row)
                       (nth 5 row) (nth 4 row) (or (nth 6 row) ""))))
      (princ (format "\n数据文件：%s\n" file)))))

(defun wime-memory-help ()
  "Show the management commands using the current key bindings."
  (interactive)
  ;; Resolve keys in the manager before entering the help buffer.
  (let ((text
         (substitute-command-keys
          "Wime 学习管理\n\n \\[wime-memory]  学习偏好       \\[wime-history]  上屏历史\n \\[wime-memory-filter]  筛选文字或编码   \\[wime-memory-refresh]  刷新   \\[wime-memory-more]  加载更多\n \\[wime-memory-details]  完整详情   \\[wime-memory-edit]  修改学习词语及编码\n \\[wime-memory-mark]  标记   \\[wime-memory-unmark]  取消标记\n \\[wime-memory-delete]  删除标记记录；没有标记时删除当前行\n \\[wime-memory-undo]  撤销最近一次修改或删除\n \\[quit-window]  返回\n\nEvil 普通状态保留 h/j/k/l、gg/G 和 /、?、n/N 导航与搜索。\n修改与删除可在这个管理 buffer 存活期间撤销。\n学习偏好影响候选；上屏历史保存当时的输入，两者分别管理。\n当前管理的是中文学习记录；修改读音会按当前词库校验。\n\nM-x wime-memory-at-point 可从正文或当前候选查找相关学习词语。\n")))
    (with-help-window "*Wime 学习帮助*"
      (princ text))))

(provide 'wime-memory)
;;; wime-memory.el ends here
