#!/bin/bash
# Commit and tag v0.2.0 (the instrumented model load).
set -u
cd /root/projects/laya-module || exit 1

TOKEN=$(grep -oE 'ghp_[A-Za-z0-9]{20,}' \
  "/sdcard/Documents/Operit_Memory/50-Config/GitHub Tokens — 2 Persistent Credentials (2026-09-25).md" | head -1)
if [ -z "$TOKEN" ]; then echo "no token found"; exit 1; fi

git add -A
git -c user.email=saeed@device.local -c user.name=Saeed commit -q -F - <<'MSG'
v0.2.0 — diagnose and harden the model load

The first build died silently inside ONNX Runtime session creation: the app's own log
stopped right after the config line, there was no Java exception, and the system logs are
unreadable from this device (dropbox denied, the crash buffer only shows our own uid, no
tombstone visible). So the load is now instrumented and defensive:

  * every ONNX Runtime call is bracketed by a flushed log line with a memory reading, so
    the log names the exact call if the process dies again;
  * a 12.9 MB probe model (the whisper-tiny encoder) is loaded first, which separates
    "ONNX Runtime cannot run in this process" from "the 1.29 GB graph is too big";
  * a heartbeat thread logs elapsed time and RSS every 4 s during the big load;
  * session options are chosen for memory, not speed: basic optimisation, memory pattern
    off, cpu arena off, threads 2/1, prepacking off;
  * the model file is opened and its first bytes read, because canRead() can answer one
    thing and the emulated storage layer another;
  * the load is refused with a clear message when free device memory is below 2.2 GB;
  * loading is manual (a button) rather than automatic at startup, so a death during the
    load cannot make the app unusable and the load can be retried.

JVM parity tests still green: 118/118 tokenizer goldens, 70/70 prompt sequences,
14/14 decode rows.
MSG

git tag -a v0.2.0 -m "v0.2.0 — instrumented model load"
git remote set-url origin "https://$TOKEN@github.com/Saeedeeas123/laya-module.git"
git push -q origin main && echo "pushed main"
git push -q origin v0.2.0 && echo "pushed tag v0.2.0"
git remote set-url origin https://github.com/Saeedeeas123/laya-module.git
if grep -q "$TOKEN" .git/config; then echo "WARNING token in config"; else echo "token scrubbed"; fi
git log --oneline -2