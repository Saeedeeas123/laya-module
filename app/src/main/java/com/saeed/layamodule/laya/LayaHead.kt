package com.saeed.layamodule.laya

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/** One answer, with the numbers a caller needs to act on it. */
class LayaAnswer(
    val questionId: String,
    val type: LayaQuestionType,
    /** choice: the winning label. */
    val choice: String?,
    /** every type: probability per option label, level index, or "false"/"true". */
    val probabilities: Map<String, Float>,
    /** score: the expected level on the rubric's own 0-based scale. */
    val score: Float?,
    /** score: level index -> level text. */
    val legend: Map<String, String>?,
    /** noul: P(true). */
    val noul: Float?,
    /** 1 - normalised entropy (choice/score) or max(p, 1-p) (noul) — the SDK's `confidence`. */
    val confidence: Float,
    /** max(p): the probability mass on the answer reported. Gate on this one. */
    val answerConfidence: Float,
    /** the auxiliary two-way action head over the same encoder. */
    val actProbability: Float,
) {
    /** A one-line summary for logs and lists. */
    fun summary(): String = when (type) {
        LayaQuestionType.CHOICE -> "$choice (p=${probabilities[choice] ?: 0f})"
        LayaQuestionType.SCORE -> "score ${score ?: 0f} of ${(legend?.size ?: 1) - 1}"
        LayaQuestionType.NOUL -> "P(true)=${noul ?: 0f}"
    }
}

/**
 * Turn the model's raw head logits into a typed answer.
 *
 * Ported from `laya.onnx_agent.ONNXAgent._decode_answers`, including the two details that are easy
 * to get silently wrong: the row arrives in *slot* order and must be un-permuted back into the
 * caller's option order, and `confidence` is not one formula — it is normalised entropy on a
 * choice or score answer, but max(p, 1-p) on a noul.
 */
object LayaHead {

    fun decode(
        question: LayaQuestion,
        logits: FloatArray,
        actLogits: FloatArray,
        temperature: Float,
    ): LayaAnswer {
        val k = question.slotCount
        require(logits.size >= k) { "question '${question.id}': expected $k logits, got ${logits.size}" }

        val z = FloatArray(k) { logits[it] / temperature }
        var maxZ = z[0]
        for (i in 1 until k) if (z[i] > maxZ) maxZ = z[i]
        var sum = 0.0
        val slotP = FloatArray(k)
        for (i in 0 until k) {
            slotP[i] = exp((z[i] - maxZ).toDouble()).toFloat()
            sum += slotP[i]
        }
        for (i in 0 until k) slotP[i] = (slotP[i] / sum).toFloat()

        // Back into the caller's option order: slot s showed option slotOrder[s].
        val p = FloatArray(k)
        for (slot in 0 until k) p[question.optionOrder?.get(slot) ?: slot] = slotP[slot]

        val answerConfidence = p.max()
        val keys = question.answerKeys()
        val probabilities = LinkedHashMap<String, Float>(k)
        for (i in 0 until k) probabilities[keys[i]] = p[i]

        val act = softmax(actLogits)

        return when (question.type) {
            LayaQuestionType.CHOICE -> {
                var best = 0
                for (i in 1 until k) if (p[i] > p[best]) best = i
                LayaAnswer(question.id, question.type, keys[best], probabilities, null, null, null,
                    entropyConfidence(p, k), answerConfidence, act)
            }
            LayaQuestionType.SCORE -> {
                var expected = 0.0
                for (i in 0 until k) expected += i * p[i].toDouble()
                val legend = LinkedHashMap<String, String>(k)
                for (i in 0 until k) legend[i.toString()] = question.options[i].first
                LayaAnswer(question.id, question.type, null, probabilities, expected.toFloat(), legend,
                    null, entropyConfidence(p, k), answerConfidence, act)
            }
            LayaQuestionType.NOUL -> LayaAnswer(question.id, question.type, null, probabilities, null,
                null, p[1], max(p[1], 1f - p[1]), answerConfidence, act)
        }
    }

    /** 1 - H(p)/log(k): how concentrated the whole distribution is. Not calibrated. */
    private fun entropyConfidence(p: FloatArray, k: Int): Float {
        if (k < 2) return 1f
        var entropy = 0.0
        for (i in 0 until k) {
            val value = max(p[i].toDouble(), 1e-12)
            entropy -= value * ln(value)
        }
        return (1.0 - entropy / ln(k.toDouble())).coerceIn(0.0, 1.0).toFloat()
    }

    private fun softmax(values: FloatArray): Float {
        var maxValue = values[0]
        for (v in values) if (v > maxValue) maxValue = v
        var sum = 0.0
        val out = FloatArray(values.size)
        for (i in values.indices) {
            out[i] = exp((values[i] - maxValue).toDouble()).toFloat()
            sum += out[i]
        }
        for (i in out.indices) out[i] = (out[i] / sum).toFloat()
        return out[0]
    }

    /** Rounds the way the reference implementation reports numbers (4 decimals). */
    fun round4(value: Float): Float = Math.round(value * 10000f) / 10000f

    /** Absolute difference, for the parity tests. */
    fun differs(a: Float, b: Float, tolerance: Float = 1e-4f): Boolean = abs(a - b) > tolerance
}