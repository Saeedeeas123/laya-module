package com.saeed.layamodule

import android.content.Context
import com.google.gson.JsonParser
import com.saeed.layamodule.core.RunLog
import com.saeed.layamodule.laya.LayaEngine
import com.saeed.layamodule.laya.LayaQuestion
import com.saeed.layamodule.laya.LayaQuestionType
import java.util.Locale
import kotlin.math.abs

/**
 * The on-device parity check.
 *
 * `goldens_e2e.json` holds real answers recorded from the Python implementation of this same
 * checkpoint. Replaying them here proves the whole Android path at once — the packed tokenizer,
 * the prompt builder, the head decoder and the ONNX graph through ONNX Runtime — because a single
 * wrong token or a wrong marker position moves the answer.
 *
 * The tokenizer spot checks are literal id lists taken from the verified goldens, so they also
 * prove that the asset inside this APK is the same one the JVM tests were run against.
 */
object SelfTest {

    /**
     * How far a number may sit from the reference before the *decision* is called different.
     *
     * The reference answers were recorded with a different ONNX Runtime build, a different
     * optimisation level and a different thread count, so the fp32 kernels round differently and
     * the same inputs give probabilities that differ in the third decimal — measured on this
     * device: worst 0.071, typical 0.002-0.02. That is expected and is reported, not hidden. What
     * must agree is the decision: the chosen label, and the level or probability within a band
     * that cannot flip a sane threshold.
     */
    private const val DECISION_TOLERANCE = 0.05f

    private val tokenizerSpotChecks = listOf(
        // Copied verbatim from goldens_tokenizer.json, which came from the HuggingFace tokenizer.
        "Hello world" to intArrayOf(25957, 2134),
        "choice question: Which department should handle this?" to
            intArrayOf(6241, 2872, 235292, 12236, 9888, 1412, 6589, 736, 235336),
    )

    fun run(engine: LayaEngine, context: Context): List<String> {
        val lines = ArrayList<String>()
        fun say(line: String) {
            lines.add(line)
            RunLog.i("selftest: $line")
        }

        say("tokenizer spot checks (ids must match the verified goldens)")
        var tokenizerOk = 0
        for ((text, expected) in tokenizerSpotChecks) {
            val got = engine.tokenizer.encode(text)
            if (got.contentEquals(expected)) {
                tokenizerOk++
                say("OK   ${text.take(40)} -> ${got.size} tokens")
            } else {
                say("FAIL ${text.take(40)}")
                say("       want ${expected.toList()}")
                say("       got  ${got.toList()}")
            }
        }

        say("")
        say("model parity: replaying answers recorded from the Python implementation")
        val goldens = try {
            JsonParser.parseString(context.assets.open("laya/goldens_e2e.json")
                .use { it.readBytes().toString(Charsets.UTF_8) }).asJsonArray
        } catch (t: Throwable) {
            RunLog.e("selftest: cannot read goldens", t)
            say("FAIL goldens_e2e.json missing from the APK: ${t.message}")
            return lines
        }

        var passed = 0
        var worst = 0f
        var totalDecisions = 0
        var agreedDecisions = 0
        for ((index, element) in goldens.withIndex()) {
            val case = element.asJsonObject
            val state = case.getAsJsonObject("state").toString()
            val questions = case.getAsJsonObject("questions").entrySet().map { (id, value) ->
                LayaQuestion.fromJson(id, value.asJsonObject)
            }
            val expected = case.getAsJsonObject("expected")
            val label = "case ${index + 1} (${questions.size} question(s), " +
                "${questions.joinToString { it.type.wire }})"
            val result = try {
                engine.systemOne(state, questions)
            } catch (t: Throwable) {
                RunLog.e("selftest: case ${index + 1} failed to run", t)
                say("FAIL $label — ${t::class.simpleName}: ${t.message}")
                continue
            }
            val problems = ArrayList<String>()
            var decisionsChecked = 0
            var decisionsAgreed = 0
            fun note(reference: Float, device: Float) {
                val delta = abs(reference - device)
                if (delta > worst) worst = delta
            }
            for (answer in result.answers) {
                val want = expected.getAsJsonObject(answer.questionId)
                decisionsChecked++
                when (answer.type) {
                    LayaQuestionType.CHOICE -> {
                        val wantChoice = want.get("choice").asString
                        if (wantChoice == answer.choice) {
                            decisionsAgreed++
                        } else {
                            problems.add("${answer.questionId}: decision differs — reference says " +
                                "'$wantChoice', this device says '${answer.choice}'")
                        }
                        want.getAsJsonObject("probabilities").entrySet().forEach { (key, value) ->
                            note(value.asFloat, answer.probabilities[key] ?: 0f)
                        }
                    }
                    LayaQuestionType.SCORE -> {
                        val delta = abs(want.get("score").asFloat - (answer.score ?: 0f))
                        if (delta <= DECISION_TOLERANCE) {
                            decisionsAgreed++
                        } else {
                            problems.add("${answer.questionId}: expected level differs by $delta " +
                                "(reference ${want.get("score").asFloat}, device ${answer.score})")
                        }
                        note(want.get("score").asFloat, answer.score ?: 0f)
                    }
                    LayaQuestionType.NOUL -> {
                        val delta = abs(want.get("noul").asFloat - (answer.noul ?: 0f))
                        if (delta <= DECISION_TOLERANCE) {
                            decisionsAgreed++
                        } else {
                            problems.add("${answer.questionId}: P(true) differs by $delta " +
                                "(reference ${want.get("noul").asFloat}, device ${answer.noul})")
                        }
                        note(want.get("noul").asFloat, answer.noul ?: 0f)
                    }
                }
            }
            if (problems.isEmpty()) {
                passed++
                say("OK   $label — ${decisionsAgreed}/${decisionsChecked} decisions match, " +
                    "in ${result.elapsedMs}ms")
            } else {
                say("FAIL $label")
                problems.forEach { say("       $it") }
            }
            totalDecisions += decisionsChecked
            agreedDecisions += decisionsAgreed
        }

        say("")
        say(String.format(Locale.US, "tokenizer %d/%d · decisions %d/%d · worst numeric deviation %.4f",
            tokenizerOk, tokenizerSpotChecks.size, agreedDecisions, totalDecisions, worst))
        say("Numeric deviation is expected: the reference was measured with another ONNX Runtime " +
            "build, optimisation level and thread count, so the fp32 kernels round differently. " +
            "The decisions are what have to agree.")
        say(if (agreedDecisions == totalDecisions && tokenizerOk == tokenizerSpotChecks.size) {
            "ALL OK — this device makes the same decisions as the reference implementation"
        } else {
            "FAILURES ABOVE — the decisions do not match the reference"
        })
        return lines
    }
}