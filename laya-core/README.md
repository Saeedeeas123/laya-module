# laya-core

The **Laya decision engine as a reusable Android library**. Plain Kotlin + ONNX Runtime: no UI, no
app lifecycle, no logging framework, no dependency on the app that consumes it.

Laya is a *non-autoregressive System 1 decision engine*, not an LLM: it answers **typed questions**
about a piece of text in **one forward pass**. No text is generated, so there is nothing to parse
and nothing to hallucinate.

| primitive | what comes back | what it is for |
|---|---|---|
| `choice` | one label plus the whole distribution over the labels | routing, categorising, intent |
| `score` | the expected level on an ordered scale, plus the distribution | urgency, sentiment, severity |
| `noul` | a calibrated P(true) that *you* threshold | spam, fraud, churn, safety gates |

## Use it

```kotlin
// build.gradle.kts of the consuming app
implementation(project(":laya-core"))                              // inside this repository
implementation("com.saeed.layamodule:laya-core:1.0.0")             // from mavenLocal, see below
```

```kotlin
val engine = LayaEngine.open(
    tokenizerStream = context.assets.open("laya/tokenizer.bin"),   // ships inside this library
    configStream = context.assets.open("laya/config.json"),
    modelPath = "/sdcard/Android/media/<your.package>/laya/laya.fp32.onnx",
)

val question = LayaQuestion.fromJson("department", JsonParser.parseString("""
    {"type":"choice","instructions":"Which department should handle this?",
     "criteria":{"billing":"invoices, payments, refunds","technical":"bugs, outages",
                 "other":"everything else"}}""").asJsonObject)

val result = engine.systemOne("""{"message":"We were billed twice for March."}""", listOf(question))
result.answer("department")?.choice             // "billing"
result.answer("department")?.answerConfidence   // max(p) — gate on this one
result.answer("department")?.probabilities      // the whole distribution
```

`systemOne` is blocking and answers **every** question about one state in a single forward pass —
put several questions in the list rather than calling it several times. Call it off the UI thread.

## What is in the box

| file | role |
|---|---|
| `LayaTokenizer` | the checkpoint's tokenizer, packed into one 13 MB asset, loaded by binary search |
| `LayaQuestion` | a typed question: parsing, validation, option rendering |
| `LayaPrompts` | builds the exact token sequence the checkpoint reads, and the option markers |
| `LayaHead` | turns raw logits into a typed answer (choice / score / noul + confidences) |
| `LayaEngine` | ONNX Runtime session, batching, one forward pass per state |
| `LayaConfig` | the checkpoint's budgets and temperatures |
| `ModelLocator` | finds the 1.29 GB graph and explains every path it tried |

## What it needs from the app

1. **The graph.** 1.29 GB, too big for an APK: the official ONNX export of
   `convaiinnovations/laya-multilingual` (mmBERT-base, 322M, Apache-2.0). `ModelLocator` looks in
   the app's own media directory first (`getExternalMediaDirs()` — no permission needed), then at
   any extra paths the caller passes.
2. **Assets.** `laya/tokenizer.bin` and `laya/config.json` ship inside this library and are
   merged into the consuming app's assets automatically.
3. **A thread.** `LayaEngine.open` takes 3-4 s for the 1.29 GB graph on this device and
   `systemOne` takes ~400-500 ms per question; both block.

Optional but recommended, and what the sample app does: pass `probeStream` (any small ONNX model)
so the engine proves ONNX Runtime works in the process *before* attempting the big graph, and
`memory` / `availableMb` probes so the load is refused with a clear message instead of being killed
when the device is short of memory.

## How it is verified

The module's logic is plain Kotlin, so it is unit-tested **on the JVM** against goldens produced by
the reference Python implementation of the same checkpoint — never against itself:

| layer | oracle | result |
|---|---|---|
| tokenizer | 118 texts tokenized by the real HuggingFace tokenizer | 118/118 exact |
| prompt builder | 70 sequences from `laya.common.build_sequence` | 70/70 exact |
| head decoder | 14 rows from `laya.onnx_agent._decode_answers` | 14/14 within 5e-4 |

```bash
./gradlew :laya-core:testDebugUnitTest --no-daemon
```

The graph itself is verified on the device by the sample app's self-test, which replays recorded
reference answers and compares **decisions** (the chosen label, the level, the probability band).
Numbers differ in the third decimal from the reference because that was recorded with a different
ONNX Runtime build, optimisation level and thread count — measured worst deviation 0.071, typical
0.002-0.02. The decisions are what must agree, and the deviation is reported rather than hidden.

## Publishing it

```bash
./gradlew :laya-core:publishToMavenLocal
# then in a consumer project:
#   repositories { mavenLocal() }
#   implementation("com.saeed.layamodule:laya-core:1.0.0")
```

JitPack cannot serve a private GitHub repository, so `mavenLocal` (or a private Maven host) is the
supported route. The AAR carries the tokenizer asset and declares ONNX Runtime and Gson as runtime
dependencies.

## Known limits

* **Zero-shot.** The shipped checkpoint was never fine-tuned on your questions. Wording changes the
  answer — the single biggest lever you have.
* **Question count changes answers.** A question that is decisive alone can be outvoted inside a
  large question set. Keep the decision rule in your code.
* **Option position matters.** With identical option texts the model prefers the earlier slot;
  `option_order` averages that out.
* **Context is 1024 tokens** by default; the state is cut to whatever room the question leaves.
  `LayaResult.truncated` / `truncatedQuestions` say when that happened.
* **It cannot write.** No summaries, no translations, no explanations.