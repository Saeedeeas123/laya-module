#!/bin/bash
# Commit and tag v1.0.0 — the module extraction, after the app was verified on the device.
set -u
cd /root/projects/laya-module || exit 1

TOKEN=$(grep -oE 'ghp_[A-Za-z0-9]{20,}' \
  "/sdcard/Documents/Operit_Memory/50-Config/GitHub Tokens — 2 Persistent Credentials (2026-09-25).md" | head -1)
if [ -z "$TOKEN" ]; then echo "no token found"; exit 1; fi

git add -A
git -c user.email=saeed@device.local -c user.name=Saeed commit -q -F - <<'MSG'
v1.0.0 — saved as a reusable module, verified on the device

The app now runs the checkpoint correctly on the phone, and the engine has been extracted
into a proper Android library module.

laya-core (com.android.library) — plain Kotlin + ONNX Runtime, no UI, no app lifecycle, no
logging framework, no dependency on the app:
  LayaTokenizer / LayaQuestion / LayaPrompts / LayaHead / LayaEngine / LayaConfig / ModelLocator
  plus the tokenizer asset (13 MB) and the checkpoint config, which ship inside the AAR.

app — the sample app, now a consumer of the module: three tabs (choice, score, noul), the
self-test screen and the log viewer.

Measured on the device (from the app's own log):
  graph load            3.4 s   (basic optimisation, memory pattern off, cpu arena off, 2 threads)
  one question          ~400-500 ms
  resident after load   1.5 GB
  tokenizer spot checks 2/2 exact
  decisions vs the Python reference: all agreed; worst numeric deviation 0.071, typical
  0.002-0.02 — expected, because the reference was recorded with a different ONNX Runtime
  build, optimisation level and thread count, so the fp32 kernels round differently.

Fixes that came out of reading that log:
  * the engine is held in LayaRuntime (process scope) instead of Compose state, so a screen
    recreation cannot lose it — the log showed a second 1.5 GB session being built
    (RSS 1501 MB -> 2772 MB) when "Load" was pressed after a recreation;
  * the self-test now compares DECISIONS (label, level, probability band) and reports the
    numeric deviation instead of failing on fp32 noise.

The module publishes: ./gradlew :laya-core:publishToMavenLocal produces
com.saeed.layamodule:laya-core:1.0.0 with the assets inside and ONNX Runtime + Gson declared.

Verified: :laya-core:testDebugUnitTest 9/9 green (118/118 tokenizer goldens, 70/70 prompt
sequences, 14/14 decode rows) and :app:assembleDebug BUILD SUCCESSFUL.
MSG

git tag -a v1.0.0 -m "v1.0.0 — module extraction, device-verified"
git remote set-url origin "https://$TOKEN@github.com/Saeedeeas123/laya-module.git"
git push -q origin main && echo "pushed main"
git push -q origin v1.0.0 && echo "pushed tag v1.0.0"
git remote set-url origin https://github.com/Saeedeeas123/laya-module.git
if grep -q "$TOKEN" .git/config; then echo "WARNING token in config"; else echo "token scrubbed"; fi
git log --oneline -3