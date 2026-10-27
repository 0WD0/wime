;;; wime-memory.el --- Manage Wime learned readings and history -*- lexical-binding: t; -*-

;;; Commentary:
;; Manage the live input method's user database.  Every mutation checks the
;; displayed row, runs in a transaction, and returns an inverse-operation
;; record.  History and learned preferences have independent lifetimes.

;;; Code:

(require 'wime)

(cl-defstruct (wime-memory-change (:constructor wime-memory--make-change))
  database kind before after)

(defun wime-memory--database ()
  "Return the configured database, or explain how to enable learning."
  (or (wime--user-db)
      (user-error "尚未启用学习，请先设置 wime-user-database")))

(defun wime-memory--file ()
  (and wime-user-database (expand-file-name wime-user-database)))

(defun wime-memory-rows (kind &optional query limit)
  "Return KIND's newest rows matching literal QUERY, up to LIMIT.
KIND is `memory' or `history'.  Memory rows are (CODE TEXT AUX COUNT UPDATED);
history rows are (ID TIME INPUT KEYS TEXT DEFAULT CONTEXT)."
  (let ((db (wime-memory--database)) (query (or query ""))
        (limit (or limit 200)))
    (unless (and (integerp limit) (> limit 0)) (error "Invalid row limit"))
    (pcase kind
      ('memory
       (sqlite-select db
                      (concat "SELECT code, text, aux, count, updated FROM memory "
                              "WHERE instr(code, ?) > 0 OR instr(text, ?) > 0 "
                              "ORDER BY updated DESC, code, text LIMIT ?")
                      (list query query limit)))
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

(provide 'wime-memory)
;;; wime-memory.el ends here
