package com.saeed.layamodule.laya

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
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
        /** Load the tokenizer and config from streams (assets) and the graph from a file. */
        fun open(
            tokenizerStream: InputStream,
            configStream: InputStream,
            modelPath: String,
            log: (String) -> Unit = {},
        ): LayaEngine {
            val config = LayaConfig.fromStream(configStream)
            val tokenizer = LayaTokenizer.fromStream(tokenizerStream)
            log("tokenizer loaded: ${tokenizer.vocabSize} tokens, ${tokenizer.mergeCount} merges")
            log("config: ${config.checkpoint} max_len=${config.maxLen} head_max_len=${config.headMaxLen}")

            val environment = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions()
            // The reference implementation enables full graph optimisation; it is functionally
            // neutral and worth ~1.45x on CPU.
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            val started = System.nanoTime()
            val session = environment.createSession(modelPath, options)
            val loadMs = (System.nanoTime() - started) / 1_000_000
            log("graph loaded in ${loadMs}ms from $modelPath")
            return LayaEngine(config, tokenizer, environment, session, modelPath, log)
        }
    }
}