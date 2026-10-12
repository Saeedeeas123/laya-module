package com.saeed.layamodule.laya

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.InputStream

/**
 * The checkpoint's own configuration (`config.json`, packed next to the tokenizer asset from the
 * checkpoint's `rl_agent_config.json`).
 *
 * The budgets and the temperatures are properties of the exported weights, not preferences:
 * `max_len` and `head_max_len` decide how the prompt is cut, and the temperatures are what turn
 * raw logits into the calibrated probabilities the checkpoint was fitted to report.
 */
class LayaConfig(
    val checkpoint: String,
    val encoder: String?,
    val maxLen: Int,
    val headMaxLen: Int,
    val temperature: FloatArray,
    val temperatureByOptions: Map<String, Float>,
) {

    /** The bucket a fitted temperature would be keyed by: type plus option count. */
    fun temperatureFor(type: LayaQuestionType, slotCount: Int): Float {
        val size = when {
            slotCount <= 2 -> "2"
            slotCount <= 5 -> "3-5"
            slotCount <= 10 -> "6-10"
            else -> "11+"
        }
        return temperatureByOptions["${type.wire}:$size"] ?: temperature[type.index]
    }

    companion object {
        fun fromStream(input: InputStream): LayaConfig {
            val json = input.use { it.readBytes().toString(Charsets.UTF_8) }
            val obj: JsonObject = JsonParser.parseString(json).asJsonObject
            val temperatures = obj.getAsJsonArray("temperature")?.map { it.asFloat }?.toFloatArray()
                ?: floatArrayOf(1f, 1f, 1f)
            require(temperatures.size == 3) { "temperature must have three entries (choice, score, noul)" }
            val byOptions = LinkedHashMap<String, Float>()
            obj.getAsJsonObject("temperature_by_options")?.entrySet()?.forEach { (key, value) ->
                byOptions[key] = value.asFloat
            }
            return LayaConfig(
                checkpoint = obj.get("checkpoint")?.asString ?: "unknown",
                encoder = obj.get("encoder")?.asString,
                maxLen = obj.get("max_len")?.asInt ?: 1024,
                headMaxLen = obj.get("head_max_len")?.asInt ?: 256,
                temperature = temperatures,
                temperatureByOptions = byOptions,
            )
        }
    }
}