#!/usr/bin/env -S python3 -B
"""Share one JSON-lines worker among every Emacs through a Unix socket.

    python -m wime_runtime.daemon [--idle SECONDS] SOCKET -- COMMAND...

If SOCKET already answers, this exits at once.  Otherwise it binds SOCKET,
forks into the background and exits, so the caller can connect as soon as it
returns.  The background process runs COMMAND (a worker reading requests on
stdin and answering on stdout) and relays every client's lines to it: request
ids are renumbered on the way in and restored on the way out, so each client
keeps its own id space; lines without an id go to every client.  A client's
{"daemon": "stop"} ends the daemon.  It also ends when COMMAND exits, closing
every client, or after SECONDS without any client.  COMMAND's stderr and the
daemon's own diagnostics go to SOCKET.log.

Only the standard library is imported, so starting a second time is cheap.
"""
import argparse
import fcntl
import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import time


def listening(path):
    try:
        with socket.socket(socket.AF_UNIX) as probe:
            probe.connect(path)
        return True
    except OSError:
        return False


def bind(path):
    """(listening socket, its inode) at PATH, or None when another daemon owns it."""
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if listening(path):
            return None
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX)
        server.bind(path)
        os.chmod(path, 0o600)
        server.listen(16)
        return server, os.stat(path).st_ino


class Relay:
    def __init__(self, server, inode, path, command, idle):
        self.server, self.inode, self.path, self.idle = server, inode, path, idle
        self.worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=sys.stderr, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(server, selectors.EVENT_READ, "server")
        self.selector.register(self.worker.stdout, selectors.EVENT_READ, "worker")
        self.clients = {}  # socket -> unfinished line
        self.routes = {}  # worker id -> (client socket, client id)
        self.serial = 0
        self.worker_partial = b""
        self.alone_since = time.monotonic()

    def run(self):
        while True:
            timeout = None if self.clients else max(0.0, self.alone_since + self.idle - time.monotonic())
            events = self.selector.select(timeout)
            if not events and not self.clients:
                return f"idle for {self.idle:g} s"
            for key, _ in events:
                if key.data == "server":
                    client, _ = self.server.accept()
                    self.clients[client] = b""
                    self.selector.register(client, selectors.EVENT_READ, "client")
                elif key.data == "worker":
                    chunk = os.read(self.worker.stdout.fileno(), 1 << 16)
                    if not chunk:
                        return f"worker exited (code {self.worker.wait()})"
                    self.from_worker(chunk)
                elif (reason := self.from_client(key.fileobj)):
                    return reason

    def from_client(self, client):
        try:
            chunk = client.recv(1 << 16)
        except OSError:
            chunk = b""
        if not chunk:
            self.drop(client)
            return None
        *lines, self.clients[client] = (self.clients[client] + chunk).split(b"\n")
        for line in lines:
            try:
                request = json.loads(line)
            except ValueError:
                print("daemon: dropped a client that sent invalid JSON", file=sys.stderr, flush=True)
                self.drop(client)
                return None
            if isinstance(request, dict) and request.get("daemon") == "stop":
                return "stopped by a client"
            if isinstance(request, dict) and "id" in request:
                self.serial += 1
                self.routes[self.serial] = (client, request["id"])
                request["id"] = self.serial
                line = json.dumps(request, ensure_ascii=False).encode()
            try:
                self.worker.stdin.write(line + b"\n")
            except BrokenPipeError:
                return f"worker exited (code {self.worker.wait()})"
        return None

    def from_worker(self, chunk):
        *lines, self.worker_partial = (self.worker_partial + chunk).split(b"\n")
        for line in lines:
            reply = json.loads(line)
            route = self.routes.get(reply.get("id")) if isinstance(reply, dict) else None
            if route is None:
                for client in list(self.clients):
                    self.send(client, line)
                continue
            client, reply["id"] = route
            if client in self.clients:
                self.send(client, json.dumps(reply, ensure_ascii=False).encode())

    def send(self, client, line):
        try:
            client.sendall(line + b"\n")
        except OSError:
            self.drop(client)

    def drop(self, client):
        if client in self.clients:
            self.selector.unregister(client)
            del self.clients[client]
            client.close()
            # A streaming worker may answer one id more than once, so routes
            # live as long as their client.
            self.routes = {k: v for k, v in self.routes.items() if v[0] is not client}
            if not self.clients:
                self.alone_since = time.monotonic()

    def close(self):
        for client in list(self.clients):
            client.close()
        if self.worker.poll() is None:
            self.worker.terminate()
            try:
                self.worker.wait(10)
            except subprocess.TimeoutExpired:
                self.worker.kill()
        try:
            if os.stat(self.path).st_ino == self.inode:
                os.unlink(self.path)
        except OSError:
            pass
        self.server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--idle", type=float, default=600.0, help="seconds without clients before exiting")
    parser.add_argument("socket")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing worker command")
    bound = bind(args.socket)
    if bound is None:
        return
    if os.fork():
        os._exit(0)
    os.setsid()
    log = os.open(args.socket + ".log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0), os.dup2(log, 1), os.dup2(log, 2)
    os.close(null), os.close(log)
    sys.stdout = sys.stderr = os.fdopen(2, "w", buffering=1)
    print(f"daemon {os.getpid()} started {time.strftime('%F %T')}: {command}", flush=True)
    relay = Relay(*bound, args.socket, command, args.idle)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    reason = "terminated"
    try:
        reason = relay.run()
    finally:
        relay.close()
        print(f"daemon {os.getpid()} ended {time.strftime('%F %T')}: {reason}", flush=True)


if __name__ == "__main__":
    main()
