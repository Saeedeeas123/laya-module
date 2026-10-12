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

    private const val TOLERANCE = 2e-3f

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
            for (answer in result.answers) {
                val want = expected.getAsJsonObject(answer.questionId)
                when (answer.type) {
                    LayaQuestionType.CHOICE -> {
                        val wantChoice = want.get("choice").asString
                        if (wantChoice != answer.choice) {
                            problems.add("${answer.questionId}: want $wantChoice, got ${answer.choice}")
                        }
                        want.getAsJsonObject("probabilities").entrySet().forEach { (key, value) ->
                            val delta = abs(value.asFloat - (answer.probabilities[key] ?: 0f))
                            if (delta > worst) worst = delta
                            if (delta > TOLERANCE) {
                                problems.add("${answer.questionId}.$key off by $delta")
                            }
                        }
                    }
                    LayaQuestionType.SCORE -> {
                        val delta = abs(want.get("score").asFloat - (answer.score ?: 0f))
                        if (delta > worst) worst = delta
                        if (delta > TOLERANCE) {
                            problems.add("${answer.questionId}: score off by $delta " +
                                "(want ${want.get("score").asFloat}, got ${answer.score})")
                        }
                    }
                    LayaQuestionType.NOUL -> {
                        val delta = abs(want.get("noul").asFloat - (answer.noul ?: 0f))
                        if (delta > worst) worst = delta
                        if (delta > TOLERANCE) {
                            problems.add("${answer.questionId}: noul off by $delta " +
                                "(want ${want.get("noul").asFloat}, got ${answer.noul})")
                        }
                    }
                }
            }
            if (problems.isEmpty()) {
                passed++
                say("OK   $label in ${result.elapsedMs}ms")
            } else {
                say("FAIL $label")
                problems.forEach { say("       $it") }
            }
        }

        say("")
        say(String.format(Locale.US, "tokenizer %d/%d, model parity %d/%d, worst deviation %.5f",
            tokenizerOk, tokenizerSpotChecks.size, passed, goldens.size(), worst))
        say(if (passed == goldens.size() && tokenizerOk == tokenizerSpotChecks.size) {
            "ALL OK — this module computes what the reference implementation computes"
        } else {
            "FAILURES ABOVE — the Android path does not match the reference"
        })
        return lines
    }
}