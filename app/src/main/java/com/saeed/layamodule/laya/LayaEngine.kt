package com.saeed.layamodule.laya

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.LongBuffer

/** One call's answers plus what it cost. */
class LayaResult(
    val answers: List<LayaAnswer>,
    val model: String,
    val elapsedMs: Long,
    /** tokens the model actually read for this call (the state counted once per question). */
    val inputTokens: Int,
    val stateTokens: Int,
    val stateTokensDropped: Int,
    val truncated: Boolean,
    val truncatedQuestions: List<String>,
) {
    fun answer(id: String): LayaAnswer? = answers.firstOrNull { it.questionId == id }
}

/**
 * The Laya decision engine on Android: the official ONNX export of
 * `convaiinnovations/laya-multilingual` (mmBERT-base, 322M) run through ONNX Runtime, with the
 * tokenizer and the prompt/head logic from this module.
 *
 * Inputs the exported graph takes, and the order they are built in, are not guesses — they come
 * from the exporter's own agent (`laya.onnx_agent.ONNXAgent._infer_batch`):
 *
 *     input_ids     [B, S]   the sequences, right-padded with the pad token
 *     attention_mask[B, S]   1 for real tokens
 *     marker_pos    [B, M]   the [MASK] position of each option (padded to at least 2 columns:
 *                            the exported head takes a top-2, and a masked slot scores -1e4)
 *     marker_mask   [B, M]   true where a marker is real
 *     qtype         [B]      0 choice, 1 score, 2 noul
 *
 * and it returns `logits [B, M]` (one score per option) and `act_logits [B, 2]` (the auxiliary
 * action head). One call answers every question about one state, in a single forward pass.
 *
 * Loading is deliberately talkative. A 1.29 GB graph is the single riskiest thing this app does,
 * and the first build died silently inside session creation with nothing in the log to say which
 * call it was in. So every ORT call is now bracketed by a flushed log line plus a memory reading,
 * and a 12.9 MB probe model is loaded first: if the probe loads and the big graph does not, the
 * problem is size/memory; if the probe fails too, the problem is ONNX Runtime itself in this
 * process. Either way the log says which, instead of leaving it to guesswork.
 */
class LayaEngine private constructor(
    val config: LayaConfig,
    val tokenizer: LayaTokenizer,
    private val environment: OrtEnvironment,
    private val session: OrtSession,
    val modelPath: String,
    private val log: (String) -> Unit,
) : AutoCloseable {

    val modelId: String = "laya-rl-agent-onnx"

    /** Answer every question about one state. Blocking; call it off the UI thread. */
    @Synchronized
    fun systemOne(
        state: String,
        questions: List<LayaQuestion>,
        maxLen: Int = config.maxLen,
        headMaxLen: Int = config.headMaxLen,
    ): LayaResult {
        require(questions.isNotEmpty()) { "at least one question is required" }
        questions.forEach { it.validate() }
        val started = System.nanoTime()

        val prompts = questions.map { LayaPrompts.build(tokenizer, state, it, maxLen, headMaxLen) }
        val rows = prompts.size
        val seqLen = prompts.maxOf { it.ids.size }
        val markerCols = maxOf(2, prompts.maxOf { it.markers.size })
        val padId = tokenizer.padId

        val ids = LongArray(rows * seqLen) { padId.toLong() }
        val attention = LongArray(rows * seqLen)
        val markerPos = LongArray(rows * markerCols)
        val markerMask = ByteBuffer.allocate(rows * markerCols).order(ByteOrder.nativeOrder())
        val qtypes = LongArray(rows)
        for (r in 0 until rows) {
            val prompt = prompts[r]
            for (i in prompt.ids.indices) {
                ids[r * seqLen + i] = prompt.ids[i].toLong()
                attention[r * seqLen + i] = 1
            }
            for (i in prompt.markers.indices) {
                markerPos[r * markerCols + i] = prompt.markers[i].toLong()
            }
            for (i in 0 until markerCols) markerMask.put(r * markerCols + i, if (i < prompt.markers.size) 1 else 0)
            qtypes[r] = questions[r].type.index.toLong()
        }
        markerMask.rewind()

        val shape2 = longArrayOf(rows.toLong(), seqLen.toLong())
        val markerShape = longArrayOf(rows.toLong(), markerCols.toLong())

        val inputs = HashMap<String, OnnxTensor>(5)
        val tensors = ArrayList<OnnxTensor>(5)
        try {
            fun keep(t: OnnxTensor): OnnxTensor { tensors.add(t); return t }
            inputs["input_ids"] = keep(OnnxTensor.createTensor(environment, LongBuffer.wrap(ids), shape2))
            inputs["attention_mask"] = keep(OnnxTensor.createTensor(environment, LongBuffer.wrap(attention), shape2))
            inputs["marker_pos"] = keep(OnnxTensor.createTensor(environment, LongBuffer.wrap(markerPos), markerShape))
            inputs["marker_mask"] = keep(OnnxTensor.createTensor(environment, markerMask, markerShape, OnnxJavaType.BOOL))
            inputs["qtype"] = keep(OnnxTensor.createTensor(environment, LongBuffer.wrap(qtypes), longArrayOf(rows.toLong())))

            session.run(inputs).use { result ->
                @Suppress("UNCHECKED_CAST")
                val logits = (result.get("logits").orElseThrow().value) as Array<FloatArray>
                @Suppress("UNCHECKED_CAST")
                val actLogits = (result.get("act_logits").orElseThrow().value) as Array<FloatArray>

                val answers = ArrayList<LayaAnswer>(rows)
                for (r in 0 until rows) {
                    val question = questions[r]
                    val temperature = config.temperatureFor(question.type, question.slotCount)
                    answers.add(LayaHead.decode(question, logits[r], actLogits[r], temperature))
                }

                val dropped = prompts.maxOf { it.stateTokens - it.stateTokensUsed }
                val elapsed = (System.nanoTime() - started) / 1_000_000
                log("decided ${rows} question(s) in ${elapsed}ms over ${prompts[0].stateTokens} state tokens")
                return LayaResult(
                    answers = answers,
                    model = modelId,
                    elapsedMs = elapsed,
                    inputTokens = attention.count { it == 1L },
                    stateTokens = prompts[0].stateTokens,
                    stateTokensDropped = dropped,
                    truncated = dropped > 0,
                    truncatedQuestions = questions.filterIndexed { i, _ ->
                        prompts[i].stateTokens > prompts[i].stateTokensUsed
                    }.map { it.id },
                )
            }
        } finally {
            tensors.forEach { runCatching { it.close() } }
        }
    }

    override fun close() {
        runCatching { session.close() }
    }

    companion object {

        /** The largest a session may grow to before the load is refused instead of attempted. */
        private const val MIN_FREE_MB = 2200L

        /**
         * Load everything, narrating each step. Every call that touches ONNX Runtime is bracketed
         * by a log line, so if the process dies the log names the exact call it died in.
         *
         * @param probeStream a small ONNX model (the 12.9 MB whisper-tiny encoder) used to prove
         *        ONNX Runtime works in this process before the 1.29 GB graph is attempted.
         * @param memory a probe for the app's own memory, logged at each step.
         * @param availableMb free device memory, used to refuse a load that would be killed.
         */
        fun open(
            tokenizerStream: InputStream,
            configStream: InputStream,
            modelPath: String,
            probeStream: InputStream? = null,
            cacheDir: File? = null,
            log: (String) -> Unit = {},
            memory: (() -> String)? = null,
            availableMb: (() -> Long)? = null,
        ): LayaEngine {
            fun step(message: String) {
                log(if (memory == null) message else "$message [${memory()}]")
            }

            step("step 1/8: reading config and tokenizer from the APK")
            val config = LayaConfig.fromStream(configStream)
            val tokenizer = LayaTokenizer.fromStream(tokenizerStream)
            step("step 2/8: tokenizer ${tokenizer.vocabSize} tokens / ${tokenizer.mergeCount} merges, " +
                "config ${config.checkpoint} max_len=${config.maxLen} head_max_len=${config.headMaxLen}")

            // The model file itself: open it for real and read the first bytes. canRead() is not
            // enough on Android — the emulated storage layer can answer one and refuse the other.
            val file = File(modelPath)
            val header = try {
                file.inputStream().use { input ->
                    val head = ByteArray(16)
                    val read = input.read(head)
                    if (read > 0) head.copyOf(read).joinToString("") { "%02x".format(it) } else "empty"
                }
            } catch (t: Throwable) {
                throw IllegalStateException("cannot open the model file at $modelPath: ${t.message}", t)
            }
            step("step 3/8: model file ${file.length()} bytes, first bytes $header")

            val freeMb = availableMb?.invoke() ?: -1L
            if (freeMb in 1 until MIN_FREE_MB) {
                throw IllegalStateException(
                    "only ${freeMb}MB of device memory is free; loading a 1.29GB graph now would be " +
                        "killed by the system. Close some apps and try again.")
            }
            step("step 4/8: free device memory ${freeMb}MB")

            step("step 5/8: creating the ONNX Runtime environment (loads the native library)")
            val environment = OrtEnvironment.getEnvironment()
            step("step 6/8: ORT environment ready")

            // ---- probe: does ONNX Runtime work in THIS process at all? ----
            if (probeStream != null && cacheDir != null) {
                val probeFile = File(cacheDir, "probe.onnx")
                try {
                    if (!probeFile.exists() || probeFile.length() < 1_000_000) {
                        step("step 7a/8: copying the probe model (${probeStream.available()} bytes) to " +
                            probeFile.absolutePath)
                        probeStream.use { input -> probeFile.outputStream().use { input.copyTo(it) } }
                    }
                    step("step 7b/8: probe model is ${probeFile.length()} bytes; opening a session on it")
                    val probeOptions = leanOptions(log)
                    environment.createSession(probeFile.absolutePath, probeOptions).use { probe ->
                        step("step 7c/8: PROBE OK — ONNX Runtime created a session in this process " +
                            "(${probe.inputNames.size} inputs). Deleting the probe.")
                    }
                    probeFile.delete()
                } catch (t: Throwable) {
                    log("step 7/8: PROBE FAILED — ${t::class.java.name}: ${t.message}")
                    log("This means ONNX Runtime itself is failing in this app process, not the big model.")
                    throw t
                }
            } else {
                step("step 7/8: no probe model supplied, skipping")
            }

            step("step 8/8: opening the 1.29GB graph — this is the heavy step, ~70-90 s")
            val options = leanOptions(log)
            val started = System.nanoTime()
            // A heartbeat during the long native call. If the process is killed while ONNX Runtime
            // is loading, the last heartbeat says how far it got and how much memory it was holding
            // — which is the difference between "too big" and "something else entirely".
            val heartbeat = Thread {
                while (!Thread.currentThread().isInterrupted) {
                    try {
                        Thread.sleep(4000)
                    } catch (interrupted: InterruptedException) {
                        return@Thread
                    }
                    val elapsed = (System.nanoTime() - started) / 1_000_000
                    log("  …still loading (${elapsed} ms elapsed)${if (memory != null) " [${memory()}]" else ""}")
                }
            }
            heartbeat.isDaemon = true
            heartbeat.start()
            val session = try {
                environment.createSession(modelPath, options)
            } finally {
                heartbeat.interrupt()
            }
            val loadMs = (System.nanoTime() - started) / 1_000_000
            step("step 8/8: GRAPH LOADED in ${loadMs}ms")
            return LayaEngine(config, tokenizer, environment, session, modelPath, log)
        }

        /**
         * Session options chosen for memory, not speed. The desktop reference enables full graph
         * optimisation and both allocators; on a phone that peak is what gets a process killed, so
         * here the optimisation level is basic, the memory pattern and the CPU arena are off, and
         * the thread count is small (each thread keeps its own scratch buffers).
         */
        private fun leanOptions(log: (String) -> Unit): OrtSession.SessionOptions {
            val options = OrtSession.SessionOptions()
            fun trySet(what: String, block: () -> Unit) {
                try {
                    block()
                } catch (t: Throwable) {
                    log("  (could not set $what: ${t.message})")
                }
            }
            trySet("optimization level") {
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            }
            trySet("memory pattern") { options.setMemoryPatternOptimization(false) }
            trySet("cpu arena") { options.setCPUArenaAllocator(false) }
            trySet("intra-op threads") { options.setIntraOpNumThreads(2) }
            trySet("inter-op threads") { options.setInterOpNumThreads(1) }
            trySet("disable prepacking") { options.addConfigEntry("session.disable_prepacking", "1") }
            log("  session options: basic optimisation, memory pattern off, cpu arena off, " +
                "threads 2/1, prepacking off")
            return options
        }
    }
}