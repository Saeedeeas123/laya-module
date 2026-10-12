package com.saeed.layamodule.laya

/**
 * A question turned into the exact token sequence the checkpoint was trained to read:
 *
 *     [CLS] <type> question: <instructions> [SEP] [MASK] option0 [MASK] option1 ... [SEP] <state> [SEP]
 *
 * `markers` holds the `[MASK]` position of each option, in slot order; those are the positions the
 * model scores. Ported from `laya.common.build_head` / `build_sequence` and verified against
 * goldens produced by that code (`goldens_prompt.json`).
 */
class LayaPrompt(
    val ids: IntArray,
    val markers: IntArray,
    val slotOrder: IntArray,
    val optionCount: Int,
    val stateTokens: Int,
    val stateTokensUsed: Int,
) {
    val truncated: Boolean get() = stateTokensUsed < stateTokens
}

object LayaPrompts {

    /** Options are capped at 48 tokens each before the head budget is measured. */
    private const val OPTION_TOKEN_CAP = 48
    private const val MIN_HEAD_TOKENS = 8
    private const val MIN_OPTION_BUDGET = 16

    fun build(
        tokenizer: LayaTokenizer,
        state: String,
        question: LayaQuestion,
        maxLen: Int,
        headMaxLen: Int,
    ): LayaPrompt {
        val mask = tokenizer.maskToken
        val options = question.renderOptions()
        val order = question.optionOrder ?: IntArray(options.size) { it }

        var headIds = tokenizer.encode(
            "${question.type.wire} question: ${question.instructions.replace(mask, " ")}"
        )

        val optionIds = ArrayList<IntArray>(order.size)
        for (slot in order) {
            val tokens = tokenizer.encode(" " + options[slot].replace(mask, " "))
            val capped = if (tokens.size > OPTION_TOKEN_CAP) tokens.copyOf(OPTION_TOKEN_CAP) else tokens
            optionIds.add(IntArray(capped.size + 1).also { it[0] = tokenizer.maskId; capped.copyInto(it, 1) })
        }

        var optionBudget = headMaxLen - optionIds.sumOf { it.size }
        if (optionBudget < MIN_OPTION_BUDGET) {
            // Not enough room for the options as written: give each an equal slice instead of
            // letting the last ones fall off the end of the sequence.
            val per = maxOf(4, (headMaxLen - MIN_OPTION_BUDGET) / maxOf(1, optionIds.size))
            for (i in optionIds.indices) {
                val current = optionIds[i]
                if (current.size > per) optionIds[i] = current.copyOf(per)
            }
            optionBudget = headMaxLen - optionIds.sumOf { it.size }
        }
        headIds = headIds.copyOf(minOf(headIds.size, maxOf(MIN_HEAD_TOKENS, optionBudget)))

        val ids = ArrayList<Int>(maxLen)
        ids.add(tokenizer.clsId)
        headIds.forEach { ids.add(it) }
        ids.add(tokenizer.sepId)
        val markers = IntArray(optionIds.size)
        for (i in optionIds.indices) {
            markers[i] = ids.size
            optionIds[i].forEach { ids.add(it) }
        }
        ids.add(tokenizer.sepId)

        val room = maxOf(0, maxLen - ids.size - 1)
        val stateIds = tokenizer.encode(state.replace(mask, " "))
        val used = minOf(stateIds.size, room)
        for (i in 0 until used) ids.add(stateIds[i])
        ids.add(tokenizer.sepId)

        val finalIds = if (ids.size > maxLen) ids.subList(0, maxLen).toIntArray() else ids.toIntArray()
        val finalMarkers = markers.filter { it < maxLen }.toIntArray()
        require(finalMarkers.size == question.slotCount) {
            "question '${question.id}': only ${finalMarkers.size} of its ${question.slotCount} option " +
                "markers fit in max_len=$maxLen with head_max_len=$headMaxLen spent on the question; " +
                "lower head_max_len, raise max_len, or use fewer options"
        }
        return LayaPrompt(finalIds, finalMarkers, order, question.slotCount, stateIds.size, used)
    }
}