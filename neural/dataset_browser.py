#!/usr/bin/env python3
"""Bounded dataset queries and version-scoped human reviews for Emacs."""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

import duckdb

from dataset_registry import Registry, fingerprint
from corpus_tones import compact_tones
from replay import parquet_row

DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "wime"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "wime/datasets"


class Browser:
    def __init__(self, registry=None, cache=CACHE, reviews=None):
        self.registry = registry or Registry()
        self.cache = Path(cache)
        self.db = duckdb.connect()
        self.db.execute("SET threads=4")
        self.db.execute("SET memory_limit='1GB'")
        path = Path(reviews or os.environ.get("WIME_DATASET_REVIEWS", DATA / "reviews/datasets.sqlite3"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.reviews = sqlite3.connect(path)
        self.reviews.execute("PRAGMA journal_mode=WAL")
        self.reviews.execute("CREATE TABLE IF NOT EXISTS reviews (dataset TEXT, snapshot TEXT, split TEXT, "
                             "row_number INTEGER, status TEXT, note TEXT, updated REAL, "
                             "PRIMARY KEY(dataset,snapshot,split,row_number))")
        self.reviews.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, dataset TEXT, "
                             "snapshot TEXT, split TEXT, row_number INTEGER, before TEXT, after TEXT)")

    def catalog(self):
        return [dict(id=name, **entry, browsable=[key for key, file in entry["files"].items()
                     if file.get("format") in ("jsonl", "json-array", "parquet") and key in ("train", "dev", "test", "cases")])
                for name, entry in self.registry.datasets.items()]

    def snapshot(self, dataset):
        entry = self.registry.entry(dataset)
        # Editing a description or quality assessment must not orphan reviews.
        content = entry.get("storage") or entry["files"]
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def view(self, dataset, split):
        entry = self.registry.entry(dataset)
        file = entry["files"][split]
        path = self.registry.path(dataset, split)
        if file.get("format") == "parquet":
            self.registry.verify(dataset, [split])
            files = [str(file) for file in sorted(path.glob("*.parquet"))]
        elif file.get("format") in ("jsonl", "json-array"):
            self.cache.mkdir(parents=True, exist_ok=True)
            identity = file.get("sha256") or hashlib.sha256(
                (self.snapshot(dataset) + "/" + split).encode()).hexdigest()
            converted = self.cache / (identity + ".parquet")
            if not converted.exists():
                self.registry.verify(dataset, [split])
                temporary = converted.with_suffix(f".{os.getpid()}.partial")
                try:
                    self.db.execute("COPY (SELECT row_number() OVER () AS _row, * FROM read_json(?, "
                                    "sample_size=-1)) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                                    [str(path), str(temporary)])
                    temporary.replace(converted)
                finally:
                    temporary.unlink(missing_ok=True)
            files = [str(converted)]
        else:
            raise ValueError("这个条目不是可浏览的样本集")
        self.db.read_parquet(files, union_by_name=True).create_view("raw_samples", replace=True)
        columns = {row[0] for row in self.db.execute("DESCRIBE raw_samples").fetchall()}
        self.db.execute("CREATE OR REPLACE VIEW samples AS SELECT *" +
                        (", ''::VARCHAR AS source" if "source" not in columns else "") + " FROM raw_samples")
        return columns | {"source"}

    def selected(self, dataset, split, number):
        records = self.db.execute("SELECT to_json(s) FROM samples s WHERE _row=?", [number]).fetchall()
        if not records:
            raise ValueError("样本不存在，请刷新")
        result = json.loads(records[0][0])
        ordinal = result["_row"]
        result = parquet_row(result)
        if "reading_evidence" in result:
            result = compact_tones(result)
        result["_row"] = ordinal
        return result

    def page(self, request):
        dataset, split = request["dataset"], request["split"]
        columns = self.view(dataset, split)
        clauses, args = [], []
        if request.get("source"):
            clauses.append("source=?"); args.append(request["source"])
        if request.get("query"):
            clauses.append("(contains(context,?) OR contains(array_to_string(gold,''),?))")
            args.extend([request["query"]] * 2)
        if request.get("tone"):
            if "tones" not in columns:
                raise ValueError("这个历史数据集没有声调标注")
            clauses.append("list_contains(tones,?)"); args.append(int(request["tone"]))
        where = " AND ".join(clauses) or "true"
        total = self.db.execute("SELECT count(*) FROM samples WHERE " + where, args).fetchall()[0][0]
        limit = min(max(int(request.get("limit", 100)), 1), 500)
        cursor = int(request.get("after", 0))
        if request.get("random") and total:
            # Select a random filtered ordinal, then show a page starting there.
            cursor = self.db.execute("SELECT _row FROM samples WHERE " + where + " ORDER BY random() LIMIT 1", args).fetchall()[0][0] - 1
        data = self.db.execute("SELECT to_json(s) FROM samples s WHERE " + where +
                               " AND _row>? ORDER BY _row LIMIT ?", args + [cursor, limit + 1]).fetchall()
        result = []
        snapshot = self.snapshot(dataset)
        for (value,) in data[:limit]:
            row = json.loads(value)
            row["review"] = self.review_value(dataset, snapshot, split, row["_row"])
            result.append(row)
        sources = self.db.execute("SELECT source,count(*) FROM samples GROUP BY source ORDER BY source").fetchall()
        documents = (self.db.execute("SELECT count(distinct document) FROM samples").fetchall()[0][0]
                     if "document" in columns else None)
        return dict(rows=result, total=total, after=cursor, has_next=len(data) > limit,
                    next=result[-1]["_row"] if result else cursor, sources=sources, documents=documents,
                    snapshot=snapshot)

    def detail(self, request):
        dataset, split = request["dataset"], request["split"]
        self.view(dataset, split)
        row = self.selected(dataset, split, int(request["row"]))
        row["review"] = self.review_value(dataset, self.snapshot(dataset), split, row["_row"])
        entry = self.registry.entry(dataset)
        result = dict(row=row, dataset=dataset, split=split, quality=entry.get("quality", {}),
                      known_issues=entry.get("known_issues", []))
        manifest_file = entry["files"].get("manifest")
        if not manifest_file or not row.get("document"):
            return result
        manifest = json.loads(self.registry.path(dataset, "manifest").read_text())
        cleaned = manifest.get("provenance", {}).get("cleaned_corpus")
        local_documents = entry["files"].get("documents")
        if not cleaned and not local_documents:
            return result
        if local_documents:
            self.registry.verify(dataset, ["documents"])
            values = self.db.execute("SELECT to_json(d) FROM read_json(?, sample_size=-1) d WHERE id=?",
                                     [str(self.registry.path(dataset, "documents")), row["document"]]).fetchall()
            document = json.loads(values[0][0]) if values else None
            record = (document["text"], document["content"]) if document else None
        else:
            if fingerprint(Path(cleaned["path"])) != cleaned["sha256"]:
                raise ValueError("清洗语料版本与样本来源不一致")
            root = Path(cleaned["path"]).parent
            with sqlite3.connect((root / "documents.sqlite3").resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
                record = db.execute("SELECT text,content FROM documents WHERE id=?", [row["document"]]).fetchone()
        if record is None or hashlib.sha256(record[0].encode()).hexdigest() != record[1]:
            raise ValueError("来源文档缺失或内容指纹不一致")
        text = record[0]
        start, end = row["reading_window"]
        offset = row["offset"]
        if text[offset:offset + len(row["gold"][0])] != row["gold"][0]:
            raise ValueError("样本与清洗文档不一致")
        result.update(annotation_text=text[start:end], annotation_target=offset-start,
                      document_excerpt=text[max(0, offset-256):offset+len(row["gold"][0])+256],
                      document_characters=len(text))
        sentence_start = max(text.rfind(mark, 0, offset) for mark in "。！？!?\n") + 1
        boundary = re.search(r"[。！？!?\n]", text[offset+len(row["gold"][0]):])
        sentence_end = offset + len(row["gold"][0]) + boundary.end() if boundary else len(text)
        view_start = max(sentence_start, offset - 512)
        view_end = min(sentence_end, offset + len(row["gold"][0]) + 1024)
        result.update(sentence_text=text[view_start:view_end], sentence_target=offset-view_start,
                      sentence_prefix=offset-sentence_start,
                      sentence_suffix=sentence_end-offset-len(row["gold"][0]),
                      sentence_clipped=view_start != sentence_start or view_end != sentence_end)
        if local_documents:
            result.update(source_document=document.get("source_document") or document["id"],
                          cleaning=document.get("cleaning") or [], source_origin=document.get("origin"))
            return result
        # Build a small provenance projection; the large text remains in its existing index.
        cache = self.cache / "documents" / cleaned["sha256"]
        cache.mkdir(parents=True, exist_ok=True)
        projected = cache / (row["source"] + ".parquet")
        if not projected.exists():
            original = root / (row["source"] + ".jsonl")
            temporary = projected.with_suffix(f".{os.getpid()}.partial")
            try:
                self.db.execute("COPY (SELECT document, source_document, cleaning FROM read_json(?, sample_size=-1)) "
                                "TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(original), str(temporary)])
                temporary.replace(projected)
            finally:
                temporary.unlink(missing_ok=True)
        provenance = self.db.execute("SELECT source_document,cleaning FROM read_parquet(?) WHERE document=?",
                                     [str(projected), row["document"]]).fetchall()
        if provenance:
            result.update(source_document=provenance[0][0], cleaning=provenance[0][1])
        return result

    def review_value(self, dataset, snapshot, split, number):
        row = self.reviews.execute("SELECT status,note FROM reviews WHERE dataset=? AND snapshot=? AND split=? AND row_number=?",
                                  (dataset, snapshot, split, number)).fetchone()
        return dict(status=row[0], note=row[1]) if row else dict(status="pending", note="")

    def review(self, request):
        dataset, split, number = request["dataset"], request["split"], int(request["row"])
        snapshot = self.snapshot(dataset)
        if snapshot != request["snapshot"]:
            raise ValueError("数据版本已经变化，请刷新后审查")
        self.view(dataset, split)
        sample = self.selected(dataset, split, number)
        after = dict(status=request["status"], note=request.get("note", ""))
        if after["status"] not in ("pending", "approved", "problem", "skipped"):
            raise ValueError("未知审查状态")
        key = (dataset, snapshot, split, number)
        with self.reviews:
            self.reviews.execute("BEGIN IMMEDIATE")
            before = self.review_value(*key)
            self.reviews.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?)", (*key, after["status"], after["note"], time.time()))
            event = self.reviews.execute("INSERT INTO history(dataset,snapshot,split,row_number,before,after) VALUES (?,?,?,?,?,?)",
                                         (*key, json.dumps(before), json.dumps(after))).lastrowid
        return dict(event=event, row=sample["_row"], review=after)

    def undo(self, event):
        record = self.reviews.execute("SELECT dataset,snapshot,split,row_number,before,after FROM history WHERE id=?", [event]).fetchone()
        if record is None:
            raise ValueError("审查记录不存在")
        key, before, after = record[:4], json.loads(record[4]), json.loads(record[5])
        with self.reviews:
            self.reviews.execute("BEGIN IMMEDIATE")
            if self.review_value(*key) != after:
                raise ValueError("这条审查已被修改，请刷新后处理")
            self.reviews.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?)", (*key, before["status"], before["note"], time.time()))
            self.reviews.execute("INSERT INTO history(dataset,snapshot,split,row_number,before,after) VALUES (?,?,?,?,?,?)",
                                 (*key, json.dumps(after), json.dumps(before)))
        return before

    def dispatch(self, request):
        op = request["op"]
        if op == "catalog": return self.catalog()
        if op == "page": return self.page(request)
        if op == "detail": return self.detail(request)
        if op == "review": return self.review(request)
        if op == "undo": return self.undo(int(request["event"]))
        raise ValueError(f"未知操作: {op}")


def main():
    try:
        result = Browser().dispatch(json.load(sys.stdin))
        print(json.dumps(dict(result=result), ensure_ascii=False))
    except Exception as error:
        print(json.dumps(dict(error=str(error)), ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
