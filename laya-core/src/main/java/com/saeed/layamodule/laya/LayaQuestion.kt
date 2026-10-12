package com.saeed.layamodule.laya

import com.google.gson.JsonElement
import com.google.gson.JsonObject

/** The three typed primitives Laya answers. The type index is what the model's head expects. */
enum class LayaQuestionType(val index: Int, val wire: String) {
    CHOICE(0, "choice"),
    SCORE(1, "score"),
    NOUL(2, "noul");

    companion object {
        fun from(wire: String): LayaQuestionType =
            values().firstOrNull { it.wire == wire.lowercase() }
                ?: throw IllegalArgumentException("unknown question type '$wire' (choice, score, noul)")
    }
}

/**
 * One typed question, in the shape the public API takes:
 *
 *     {"type": "choice", "instructions": "...", "criteria": {"billing": "invoices, refunds"}}
 *     {"type": "score",  "instructions": "...", "criteria": ["not urgent", "soon", "blocking"]}
 *     {"type": "noul",   "instructions": "...", "criteria": {"false": "...", "true": "..."}}
 *
 * Optional: `labels` (noul only) renames the two model-facing slots without changing the meaning
 * of the returned probability, and `option_order` permutes the slots the options are shown in.
 */
class LayaQuestion(
    val id: String,
    val type: LayaQuestionType,
    val instructions: String,
    /** choice: label -> description (description may be null); score: the ordered levels. */
    val options: List<Pair<String, String?>>,
    /** noul only: what "false" and "true" mean, if the caller spelled them out. */
    val falseMeaning: String? = null,
    val trueMeaning: String? = null,
    /** noul only: the model-facing slot names. Always exactly two, distinct. */
    val falseLabel: String = "false",
    val trueLabel: String = "true",
    val optionOrder: IntArray? = null,
) {

    /** How many slots the model scores. Noul is always two: [false, true]. */
    val slotCount: Int
        get() = when (type) {
            LayaQuestionType.NOUL -> 2
            else -> options.size
        }

    /** The option texts, in label-index order, exactly as the reference implementation renders them. */
    fun renderOptions(): List<String> = when (type) {
        LayaQuestionType.CHOICE -> options.map { (label, description) ->
            if (description == null || description.isEmpty()) label else "$label: $description"
        }
        LayaQuestionType.SCORE -> options.mapIndexed { i, (level, _) -> "level $i: $level" }
        LayaQuestionType.NOUL -> listOf(
            "$falseLabel: " + (falseMeaning ?: "no, the statement does not hold"),
            "$trueLabel: " + (trueMeaning ?: "yes, the statement holds"),
        )
    }

    /** The labels the answer comes back keyed by: option labels, level indices, or false/true. */
    fun answerKeys(): List<String> = when (type) {
        LayaQuestionType.CHOICE -> options.map { it.first }
        LayaQuestionType.SCORE -> options.indices.map { it.toString() }
        LayaQuestionType.NOUL -> listOf("false", "true")
    }

    fun validate() {
        require(instructions.isNotBlank()) { "question '$id': instructions must not be blank" }
        when (type) {
            LayaQuestionType.CHOICE -> {
                require(options.size >= 2) { "question '$id': choice needs at least two options" }
                require(options.map { it.first }.toSet().size == options.size) {
                    "question '$id': choice option labels must be distinct"
                }
            }
            LayaQuestionType.SCORE -> {
                require(options.size >= 2) { "question '$id': score needs at least two levels" }
            }
            LayaQuestionType.NOUL -> {
                require(falseLabel.isNotBlank() && trueLabel.isNotBlank() && falseLabel != trueLabel) {
                    "question '$id': noul labels must be two distinct non-empty strings"
                }
            }
        }
        optionOrder?.let {
            require(it.size == slotCount && it.toSet().size == slotCount &&
                it.all { index -> index in 0 until slotCount }) {
                "question '$id': option_order must be a permutation of 0..${slotCount - 1}"
            }
        }
    }

    companion object {
        fun fromJson(id: String, obj: JsonObject): LayaQuestion {
            val type = LayaQuestionType.from(obj.get("type").asString)
            val instructions = obj.get("instructions")?.asString
                ?: throw IllegalArgumentException("question '$id': instructions is required")
            val criteria = obj.get("criteria")

            var options: List<Pair<String, String?>> = emptyList()
            var falseMeaning: String? = null
            var trueMeaning: String? = null

            when (type) {
                LayaQuestionType.CHOICE -> {
                    val element = criteria
                        ?: throw IllegalArgumentException("question '$id': choice needs criteria")
                    options = when {
                        element.isJsonArray -> element.asJsonArray.map { it.asString to null }
                        element.isJsonObject -> element.asJsonObject.entrySet().map { (label, value) ->
                            label to renderCriterion(value)
                        }
                        else -> throw IllegalArgumentException("question '$id': criteria must be an object or array")
                    }
                }
                LayaQuestionType.SCORE -> {
                    val element = criteria
                        ?: throw IllegalArgumentException("question '$id': score needs criteria (an ordered list)")
                    require(element.isJsonArray) { "question '$id': score criteria must be an ordered list" }
                    options = element.asJsonArray.map { renderCriterion(it)!! to null }
                }
                LayaQuestionType.NOUL -> {
                    if (criteria != null && criteria.isJsonObject) {
                        for ((key, value) in criteria.asJsonObject.entrySet()) {
                            when (key.lowercase()) {
                                "false" -> falseMeaning = renderCriterion(value)
                                "true" -> trueMeaning = renderCriterion(value)
                                else -> throw IllegalArgumentException(
                                    "question '$id': noul criteria keys must be 'false' and 'true'")
                            }
                        }
                    }
                }
            }

            val labels = obj.get("labels")
            var falseLabel = "false"
            var trueLabel = "true"
            if (labels != null && !labels.isJsonNull) {
                require(type == LayaQuestionType.NOUL) { "question '$id': labels is only for noul" }
                val map = labels.asJsonObject
                require(map.has("false") && map.has("true") && map.size() == 2) {
                    "question '$id': labels must map exactly 'false' and 'true'"
                }
                falseLabel = map.get("false").asString.trim()
                trueLabel = map.get("true").asString.trim()
            }

            val order = obj.get("option_order")?.takeIf { !it.isJsonNull }?.let { element ->
                element.asJsonArray.map { it.asInt }.toIntArray()
            }

            return LayaQuestion(id, type, instructions, options, falseMeaning, trueMeaning,
                falseLabel, trueLabel, order).also { it.validate() }
        }

        /** Strings pass through; numbers and other scalars are rendered as their JSON text. */
        fun renderCriterion(value: JsonElement?): String? = when {
            value == null || value.isJsonNull -> null
            value.isJsonPrimitive -> value.asString
            else -> value.toString()
        }
    }
}