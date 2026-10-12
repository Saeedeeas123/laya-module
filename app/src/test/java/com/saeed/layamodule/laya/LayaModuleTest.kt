package com.saeed.layamodule.laya

import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * JVM verification of the Laya module against goldens produced by the real Python implementation.
 *
 * Three layers, each with its own oracle:
 *   tokenizer  ids for 118 texts, from the HuggingFace tokenizer (`goldens_tokenizer.json`)
 *   prompt     the full token sequence + option markers, from `laya.common.build_sequence`
 *              (`goldens_prompt.json`)
 *   head       answers decoded from fixed logits, from `laya.onnx_agent._decode_answers`
 *              (`goldens_decode.json`)
 *
 * The model itself is checked on the device by the app's self-test, which replays
 * `goldens_e2e.json` — real answers recorded from the ONNX agent in Python.
 *
 * Run:  ./gradlew :app:testDebugUnitTest --no-daemon
 */
private const val ASSET = "src/main/assets/laya/tokenizer.bin"
private const val TOKENIZER_GOLDENS = "src/test/resources/laya/goldens_tokenizer.json"
private const val PROMPT_GOLDENS = "src/test/resources/laya/goldens_prompt.json"
private const val DECODE_GOLDENS = "src/test/resources/laya/goldens_decode.json"

private fun loadTokenizer(): LayaTokenizer =
    File(ASSET).inputStream().use { LayaTokenizer.fromStream(it) }

private fun readJson(path: String) = JsonParser.parseString(File(path).readText())

class LayaTokenizerTest {

    @Test
    fun `asset loads and exposes the checkpoint's special tokens`() {
        val tok = loadTokenizer()
        assertEquals(256000, tok.vocabSize)
        assertEquals(580604, tok.mergeCount)
        assertEquals(2, tok.clsId)
        assertEquals(1, tok.sepId)
        assertEquals(4, tok.maskId)
        assertEquals(0, tok.padId)
        assertEquals(3, tok.unkId)
        assertEquals("<mask>", tok.maskToken)
        assertEquals("<pad>", tok.token(0))
    }

    @Test
    fun `every golden text tokenizes to the ids the real tokenizer produced`() {
        val tok = loadTokenizer()
        val cases = readJson(TOKENIZER_GOLDENS).asJsonObject.getAsJsonArray("cases")
        var checked = 0
        val failures = StringBuilder()
        for (element in cases) {
            val obj = element.asJsonObject
            val text = obj.get("text").asString
            val want = obj.getAsJsonArray("ids").map { it.asInt }.toIntArray()
            val got = tok.encode(text)
            if (!want.contentEquals(got)) {
                failures.append("\n  ").append(text.take(60))
                    .append("\n    want ").append(want.take(12))
                    .append("\n    got  ").append(got.take(12))
            }
            checked++
        }
        assertEquals("$checked cases checked, failures:$failures", 0, failures.length)
        assertTrue("expected a substantial corpus, got $checked", checked >= 100)
    }

    @Test
    fun `decode round-trips through the byte fallback`() {
        val tok = loadTokenizer()
        // Decoding inserts a space where the pre-tokenizer inserted a marker: at the start of the
        // text (prepend_scheme=always) and after every newline run. That is the shape the model
        // was trained on, so it is asserted explicitly rather than papered over.
        val expected = mapOf(
            "Hello world" to " Hello world",
            "سلام دنیا" to " سلام دنیا",
            "emoji 😀 test" to " emoji 😀 test",
            "line\nbreak" to " line\n break",
        )
        for ((text, want) in expected) {
            assertEquals(want, tok.decode(tok.encode(text)))
        }
    }

    @Test
    fun `empty and whitespace inputs do not crash`() {
        val tok = loadTokenizer()
        assertEquals(0, tok.encode("").size)
        assertTrue(tok.encode("   ").isNotEmpty())
        assertTrue(tok.encode("\n\n").isNotEmpty())
    }
}

class LayaPromptTest {

    @Test
    fun `every golden question builds the sequence the SDK builds`() {
        val tok = loadTokenizer()
        val goldens = readJson(PROMPT_GOLDENS).asJsonArray
        var checked = 0
        val failures = StringBuilder()
        for (element in goldens) {
            val obj = element.asJsonObject
            val question = LayaQuestion.fromJson("q", obj.getAsJsonObject("question"))
            val state = obj.get("state").asString
            val maxLen = obj.get("max_len").asInt
            val headMaxLen = obj.get("head_max_len").asInt
            val wantIds = obj.getAsJsonArray("ids").map { it.asInt }.toIntArray()
            val wantMarkers = obj.getAsJsonArray("markers").map { it.asInt }.toIntArray()
            val wantType = obj.get("qtype").asInt

            val prompt = LayaPrompts.build(tok, state, question, maxLen, headMaxLen)
            if (!wantIds.contentEquals(prompt.ids)) {
                failures.append("\n  ids differ for ").append(question.type.wire)
                    .append(" maxLen=").append(maxLen).append(" headMaxLen=").append(headMaxLen)
                    .append("\n    want ").append(wantIds.take(16)).append(" (").append(wantIds.size).append(")")
                    .append("\n    got  ").append(prompt.ids.take(16)).append(" (").append(prompt.ids.size).append(")")
            } else if (!wantMarkers.contentEquals(prompt.markers)) {
                failures.append("\n  markers differ: want ").append(wantMarkers.toList())
                    .append(" got ").append(prompt.markers.toList())
            } else if (wantType != question.type.index) {
                failures.append("\n  qtype differs: want ").append(wantType)
                    .append(" got ").append(question.type.index)
            }
            checked++
        }
        assertEquals("$checked sequences checked, failures:$failures", 0, failures.length)
        assertTrue(checked >= 40)
    }

    @Test
    fun `option_order changes the slot order but not the option set`() {
        val tok = loadTokenizer()
        val question = LayaQuestion.fromJson("q", JsonParser.parseString(
            """{"type":"choice","instructions":"Which team?",
                "criteria":["billing","technical","sales"],"option_order":[2,0,1]}""").asJsonObject)
        val prompt = LayaPrompts.build(tok, "hello", question, 1024, 256)
        assertEquals(listOf(2, 0, 1), prompt.slotOrder.toList())
        assertEquals(3, prompt.markers.size)
    }
}

class LayaHeadTest {

    @Test
    fun `every golden row decodes to the SDK's answer`() {
        val goldens = readJson(DECODE_GOLDENS).asJsonArray
        var checked = 0
        val failures = StringBuilder()
        for (element in goldens) {
            val obj = element.asJsonObject
            val question = LayaQuestion.fromJson("q", obj.getAsJsonObject("question"))
            val logits = obj.getAsJsonArray("logits").map { it.asFloat }.toFloatArray()
            val actLogits = obj.getAsJsonArray("act_logits").map { it.asFloat }.toFloatArray()
            val expected = obj.getAsJsonObject("expected")

            val answer = LayaHead.decode(question, logits, actLogits, 1.0f)

            val tolerance = 5e-4f
            when (answer.type) {
                LayaQuestionType.CHOICE -> {
                    val wantChoice = expected.get("choice").asString
                    if (wantChoice != answer.choice) {
                        failures.append("\n  choice: want ").append(wantChoice)
                            .append(" got ").append(answer.choice)
                    }
                    for ((key, value) in expected.getAsJsonObject("probabilities").entrySet()) {
                        val got = answer.probabilities[key] ?: -1f
                        if (LayaHead.differs(value.asFloat, got, tolerance)) {
                            failures.append("\n  p[").append(key).append("]: want ")
                                .append(value.asFloat).append(" got ").append(got)
                        }
                    }
                }
                LayaQuestionType.SCORE -> {
                    val wantScore = expected.get("score").asFloat
                    if (LayaHead.differs(wantScore, answer.score ?: -1f, tolerance)) {
                        failures.append("\n  score: want ").append(wantScore)
                            .append(" got ").append(answer.score)
                    }
                }
                LayaQuestionType.NOUL -> {
                    val wantNoul = expected.get("noul").asFloat
                    if (LayaHead.differs(wantNoul, answer.noul ?: -1f, tolerance)) {
                        failures.append("\n  noul: want ").append(wantNoul)
                            .append(" got ").append(answer.noul)
                    }
                }
            }
            val wantAct = expected.get("act_probability").asFloat
            if (LayaHead.differs(wantAct, answer.actProbability, tolerance)) {
                failures.append("\n  act: want ").append(wantAct).append(" got ").append(answer.actProbability)
            }
            checked++
        }
        assertEquals("$checked rows checked, failures:$failures", 0, failures.length)
        assertTrue(checked >= 8)
    }

    @Test
    fun `noul confidence is the probability of the more likely side`() {
        val question = LayaQuestion.fromJson("q", JsonParser.parseString(
            """{"type":"noul","instructions":"Is this a refund request?"}""").asJsonObject)
        val yes = LayaHead.decode(question, floatArrayOf(0f, 4f), floatArrayOf(0f, 1f), 1f)
        assertTrue("P(true) should dominate", (yes.noul ?: 0f) > 0.9f)
        assertEquals(yes.noul!!, yes.confidence, 1e-4f)
        val no = LayaHead.decode(question, floatArrayOf(4f, 0f), floatArrayOf(0f, 1f), 1f)
        assertTrue("P(true) should be small", (no.noul ?: 1f) < 0.1f)
        assertEquals(1f - no.noul!!, no.confidence, 1e-4f)
    }

    @Test
    fun `choice confidence is entropy based, not max probability`() {
        val question = LayaQuestion.fromJson("q", JsonParser.parseString(
            """{"type":"choice","instructions":"Pick","criteria":["a","b","c","d"]}""").asJsonObject)
        val confident = LayaHead.decode(question, floatArrayOf(12f, 0f, 0f, 0f), floatArrayOf(0f, 1f), 1f)
        val flat = LayaHead.decode(question, floatArrayOf(0f, 0f, 0f, 0f), floatArrayOf(0f, 1f), 1f)
        assertTrue(confident.confidence > 0.9f)
        assertTrue(flat.confidence < 0.01f)
        // four equal options: every one of them holds a quarter of the probability mass
        assertEquals(0.25f, flat.answerConfidence, 1e-4f)
        assertEquals("a", confident.choice)
        assertFalse(LayaHead.differs(confident.answerConfidence, 1f, 1e-3f))
    }
}