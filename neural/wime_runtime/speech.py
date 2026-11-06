#!/usr/bin/env -S python3 -B
"""Persistent dictation worker: pw-record audio streamed through Qwen3-ASR.

JSON lines on stdin: {"op": "start", "id": n, "context": text} starts
recording, {"op": "stop", "id": n} ends it and {"op": "cancel", "id": n}
drops it.  Stdout answers {"ready": true} once the model is warm, then
{"id": n, "recording": true}, {"id": n, "partial": text} after each chunk
and {"id": n, "final": text, "elapsed_ms": ms} after stop.

Partial text follows qwen_asr's streaming policy as speech-qwen-streaming-v1
measured it: each 2 s chunk re-encodes all audio so far; from the third chunk
on the previous output minus its last five tokens is forced as a prefix and
greedy decoding adds at most 32 tokens.  Partials see no context: given the
text before point, a chunk of silence makes the model copy that text (129 of
149 pilot takes after 2 s of leading noise).  The final text decodes all audio
once with the context and no forced prefix, which copied nothing and was the
most accurate there.  A slow chunk skips to the newest boundary instead of
queueing.
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

RATE = 16000
UNFIXED_CHUNKS, UNFIXED_TOKENS, CHUNK_TOKENS, FINAL_TOKENS = 2, 5, 32, 128


class Recognizer:
    def __init__(self, model, device):
        self.processor = AutoProcessor.from_pretrained(model)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()
        self.device = device

    def prefix(self, raw, chunk):
        """The upstream rollback: nothing for the first chunks, else drop the last tokens without splitting a character."""
        if chunk < UNFIXED_CHUNKS:
            return ""
        ids = self.tokenizer.encode(raw)
        k = UNFIXED_TOKENS
        while True:
            end = max(0, len(ids) - k)
            text = self.tokenizer.decode(ids[:end]) if end else ""
            if "�" not in text or end == 0:
                return "" if "�" in text else text
            k += 1

    @torch.inference_mode()
    def __call__(self, audio, context, forced, limit):
        inputs = self.processor.apply_transcription_request(audio, language="zh", prompt=context or None)
        inputs = inputs.to(self.device, torch.bfloat16)
        if forced:
            ids = torch.tensor([self.tokenizer.encode(forced, add_special_tokens=False)], device=self.device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], 1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(ids)], 1)
        sequence = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False, num_beams=1)
        return forced + self.tokenizer.decode(sequence[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class Recording:
    """pw-record's raw PCM, appended by a reader thread."""

    def __init__(self, target):
        self.process = subprocess.Popen(
            ["pw-record", *(["--target", target] if target else []), "--rate", str(RATE), "--channels", "1",
             "--format", "s16", "--raw", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.data, self.lock = bytearray(), threading.Lock()
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        while block := self.process.stdout.read1(4096):
            with self.lock:
                self.data += block

    def samples(self):
        with self.lock:
            data = bytes(self.data[:len(self.data) // 2 * 2])
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768

    def close(self):
        self.process.terminate()
        self.process.wait()
        self.reader.join()


def send(**message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


def requests(stream, inbox):
    for line in stream:
        inbox.put(json.loads(line))
    inbox.put(None)


def finish(recognize, session):
    """All audio decoded once with the context, free of the partials' prefix."""
    start = time.perf_counter()
    audio = session["recording"].samples()
    text = recognize(audio, session["context"], "", FINAL_TOKENS) if len(audio) else ""
    send(id=session["id"], final=text, elapsed_ms=1000 * (time.perf_counter() - start))


def serve(recognize, inbox, chunk, target):
    size = round(chunk * RATE)
    session = None
    while True:
        try:
            request = inbox.get(timeout=0.02) if session else inbox.get()
        except queue.Empty:
            audio = session["recording"].samples()
            boundary = len(audio) // size * size
            if boundary > session["done"]:
                forced = recognize.prefix(session["raw"], session["chunk"])
                session.update(raw=recognize(audio[:boundary], "", forced, CHUNK_TOKENS),
                               chunk=session["chunk"] + 1, done=boundary)
                send(id=session["id"], partial=session["raw"])
            continue
        if session and (request is None or request["op"] == "start" or request["id"] == session["id"]):
            session["recording"].close()
            if request and request["op"] == "stop":
                finish(recognize, session)
            session = None
        if request is None:
            return
        if request["op"] == "start":
            session = dict(id=request["id"], context=request.get("context", ""), recording=Recording(target),
                           raw="", chunk=0, done=0)
            send(id=session["id"], recording=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Qwen3-ASR-1.7B-hf directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=float, default=2.0, help="seconds of audio per streaming update")
    parser.add_argument("--target", help="PipeWire source for pw-record; the default source otherwise")
    args = parser.parse_args()
    recognize = Recognizer(args.model, args.device)
    # Pay CUDA initialization before the first dictation.
    recognize(np.zeros(RATE, dtype=np.float32), "", "", 4)
    send(ready=True)
    inbox = queue.Queue()
    threading.Thread(target=requests, args=(sys.stdin, inbox), daemon=True).start()
    serve(recognize, inbox, args.chunk, args.target)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"wime speech worker failed ({type(error).__name__})", file=sys.stderr)
        sys.exit(1)
