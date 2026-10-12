# Laya Module

An Android app that is also a **reusable module**: it exposes the Laya decision engine as three
typed primitives — **choice**, **score**, **noul** — and gives you a workspace to ask any question
of any text and read the answer.

Laya is a *non-autoregressive System 1 decision engine*, not an LLM. It does not write anything.
It answers typed questions about a piece of text in **one forward pass**:

| primitive | what comes back | what it is for |
|---|---|---|
| `choice` | one label plus the whole distribution over the labels | routing, categorising, intent |
| `score` | the expected level on an ordered scale, plus the distribution | urgency, sentiment, severity |
| `noul` | a calibrated P(true) that *you* threshold | spam, fraud, churn, safety gates |

No generation means nothing to parse and nothing to hallucinate; one pass means the whole answer
arrives at once.

## The app

Three tabs, one per primitive — no more, because these are the engine's three functions. Each tab
is a workspace:

* a **state** to judge (any text: a message, an email, a document; optionally wrapped in a JSON
  field name),
* one or more **questions** of that type — instructions, options, level lists, yes/no meanings —
  with "add another question" so several questions about the same state are answered in one pass,
* a **run** button, and then per question: the answer, the full probability distribution, the
  calibrated `answer_confidence`, the entropy `confidence`, the auxiliary action head, and the
  cost (milliseconds, input tokens, output tokens = 0).

Two more screens, from the top bar:

* **Self-test** — replays answers recorded from the reference Python implementation on this same
  checkpoint. If every line says OK, this module computes what the reference computes: tokenizer,
  prompt builder, head decoder and ONNX graph together.
* **Log** — the live log, plus previous runs. Hard requirement: this app's own log is the only
  diagnostic channel it has, so every line is written and flushed as it happens, one timestamped
  file per run (10 kept), to `getExternalFilesDir/logs/` **and** to `Download/LayaModule/logs/`
  (through MediaStore, so no permission is needed).

## The module

Everything decision-related is plain Kotlin in `com.saeed.layamodule.laya` — no Android types, no
UI, so it is unit-tested on the JVM and can be lifted into another app as-is:

```
LayaTokenizer   the checkpoint's tokenizer, packed into one 13 MB asset, loaded by binary search
LayaQuestion    a typed question: parsing, validation, option rendering
LayaPrompts     builds the exact token sequence the checkpoint reads, and the option markers
LayaHead        turns raw logits into a typed answer (choice / score / noul + confidences)
LayaEngine      ONNX Runtime session, batching, one forward pass per state
LayaConfig      the checkpoint's budgets and temperatures
ModelLocator    finds the 1.29 GB graph, and explains every path it tried
```

```kotlin
val engine = LayaEngine.open(
    tokenizerStream = assets.open("laya/tokenizer.bin"),
    configStream = assets.open("laya/config.json"),
    modelPath = "/sdcard/Android/media/com.saeed.layamodule/laya/laya.fp32.onnx",
)
val question = LayaQuestion.fromJson("department", JsonParser.parseString("""
    {"type":"choice","instructions":"Which department should handle this?",
     "criteria":{"billing":"invoices, payments, refunds","technical":"bugs, outages",
                 "other":"everything else"}}""").asJsonObject)

val result = engine.systemOne("""{"message":"We were billed twice for March."}""", listOf(question))
println(result.answer("department")?.choice)           // billing
println(result.answer("department")?.answerConfidence) // how sure it is, 0..1
```

## Running it

Install `/sdcard/Download/layamodule-debug.apk`, then open **Laya Module**. The graph must be at

```
/sdcard/Android/media/com.saeed.layamodule/laya/laya.fp32.onnx
```

which needs no permission and no import step — copying the file there is enough. If it is missing,
the app lists every path it tried and what was wrong with each. Loading takes a while (~70-90 s for
1.29 GB) and the header shows progress; after that a decision takes a few seconds.

## How this was verified

Nothing here is "it compiles, ship it". Every layer is checked against the reference implementation
(`convaiinnovations/laya-multilingual`, through the official Python SDK):

| layer | oracle | result |
|---|---|---|
| tokenizer | 118 texts tokenized by the real HuggingFace tokenizer | **118/118 exact** (JVM test) |
| asset format | the packed asset re-read by a second implementation, in Python | **118/118 reproduced** |
| prompt builder | 70 sequences from `laya.common.build_sequence` | **70/70 exact** (JVM test) |
| head decoder | 14 rows from `laya.onnx_agent._decode_answers` | **14/14 within 5e-4** (JVM test) |
| whole path on the phone | 15 real answers recorded from the ONNX agent in Python | app's **Self-test** screen |

```bash
./gradlew :app:testDebugUnitTest --no-daemon   # the three JVM layers
python3 tools/show_failures.py                 # readable failure detail
```

Four tokenizer inputs are recorded as **known divergences** rather than hidden: they contain the
SentencePiece marker character U+2581 literally. The real pre-tokenizer splits the *original* text
on spaces and only then replaces them, so a literal marker and a space are not the same thing to it,
while this implementation normalises first. Nothing a user can type hits it (U+2581 is a
SentencePiece artefact) and every space-based input matches exactly.

## Rebuilding the assets

`app/src/main/assets/laya/` and `app/src/test/resources/laya/` are generated, not hand-written:

```bash
python tools/build_assets.py     # needs the reference Python stack (see below)
python tools/check_asset.py      # re-reads the packed asset and replays every tokenizer golden
python tools/check_bpe.py        # proves the tokenizer algorithm against HuggingFace itself
```

`build_assets.py` needs the reference implementation installed, because the goldens must come from
*it* and not from this module — otherwise the test would only prove the module agrees with itself:

```bash
uv venv --python 3.12 .venv && uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install "laya[serve]" onnxruntime
# plus the checkpoint's support files (~20 MB, public repo convaiinnovations/laya-multilingual):
#   rl_agent_config.json, tokenizer/tokenizer.json, tokenizer/tokenizer_config.json
```

The 1.29 GB graph itself is the official ONNX export of `convaiinnovations/laya-multilingual`
(mmBERT-base, 322M, Apache-2.0), made with the project's own `scripts/export_onnx.py`.

## Limits, stated plainly

* **Zero-shot.** The shipped checkpoint was never fine-tuned on your questions. Wording changes the
  answer — that is not a figure of speech, it is the single biggest lever you have.
* **Question count changes answers too.** A question that is decisive on its own can be outvoted
  inside a large question set; keep the decision rule in your code and treat the model as a signal.
* **Option position matters.** With identical option texts the model still prefers the earlier slot;
  `option_order` exists to average that out.
* **Context is 1024 tokens** by default, and the state is cut to whatever room the question leaves.
  The app tells you when it truncated, and by how many tokens.
* **It cannot write.** No summaries, no translations, no explanations. If you need prose, you need
  an LLM — Laya is the cheap decision layer in front of it.

## Project layout

```
app/src/main/java/com/saeed/layamodule/
  MainActivity.kt          the three tabs, the self-test screen, the log viewer
  SelfTest.kt              on-device parity check against recorded reference answers
  core/RunLog.kt           durable per-run logging + in-app viewing
  laya/                    the module itself (no Android types except ModelLocator)
app/src/main/assets/laya/  tokenizer.bin, config.json, goldens_e2e.json
app/src/test/              JVM tests + the goldens they run against
tools/                     asset builders, the tokenizer reference, the failure reporter
```

Toolchain is the frozen, on-device-verified set: AGP 8.2.2, Kotlin 1.9.22, Gradle 8.5, JDK 17,
compileSdk/targetSdk 35, buildTools 35.0.1, ONNX Runtime Android 1.18.0, Gson 2.10.1.