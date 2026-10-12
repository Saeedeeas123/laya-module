package com.saeed.layamodule.laya

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

/**
 * The laya-multilingual tokenizer, implemented from the checkpoint's own tokenizer.json.
 *
 * Pipeline (read out of the checkpoint, not assumed):
 *   normalizer      replace " " with the marker ▁ (U+2581)
 *   pre_tokenizer   Metaspace(marker, prepend_scheme=always, split=true)
 *   model           BPE, 256000 tokens, 580604 merges, byte_fallback
 *   post_processor  <bos> A <eos>, applied only when add_special_tokens is true — laya never
 *                   asks for it, because the prompt builder places the special tokens itself.
 *
 * The vocabulary and the merge table are packed into one asset (`tokenizer.bin`) by
 * `tools/build_assets.py`, and every lookup here is a binary search over that asset, so loading
 * costs one file read and no dictionary building. The algorithm is verified against 118 goldens
 * produced by the real HuggingFace tokenizer: see `tools/check_asset.py` (the same algorithm in
 * Python, run against the packed asset alone) and `LayaTokenizerTest` (this code, on the JVM).
 */
class LayaTokenizer private constructor(
    private val blob: ByteArray,
    private val offsets: IntArray,      // n_vocab + 1 offsets into blob, in id order
    private val byToken: IntArray,      // vocab ids sorted by token bytes, for lookup
    private val mergeLeft: IntArray,
    private val mergeRight: IntArray,
    private val mergeResult: IntArray,
    private val mergeRank: IntArray,
    val vocabSize: Int,
    val mergeCount: Int,
    val clsId: Int,
    val sepId: Int,
    val maskId: Int,
    val padId: Int,
    val unkId: Int,
    val maskToken: String,
) {

    /** The token text for an id. */
    fun token(id: Int): String =
        String(blob, offsets[id], offsets[id + 1] - offsets[id], StandardCharsets.UTF_8)

    /**
     * Lexicographic byte comparison of a vocabulary token against `want`: negative, zero or
     * positive. Length decides only after every shared byte matched — ordering by length first
     * would put "b" before "aa" and break the binary search, which is exactly what the first
     * version did (every character then came back as <unk>).
     */
    private fun compareToken(id: Int, want: ByteArray): Int {
        val from = offsets[id]
        val len = offsets[id + 1] - from
        val shared = if (len < want.size) len else want.size
        for (i in 0 until shared) {
            val a = blob[from + i].toInt() and 0xFF
            val b = want[i].toInt() and 0xFF
            if (a != b) return if (a < b) -1 else 1
        }
        return when {
            len == want.size -> 0
            len < want.size -> -1
            else -> 1
        }
    }

    /** Binary search the id space for a token, or -1 when the vocabulary has no such token. */
    fun vocabId(text: String): Int {
        val want = text.toByteArray(StandardCharsets.UTF_8)
        var lo = 0
        var hi = vocabSize - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val id = byToken[mid]
            val cmp = compareToken(id, want)
            if (cmp == 0) return id
            if (cmp < 0) lo = mid + 1 else hi = mid - 1
        }
        return -1
    }

    /** The rank and result of merging two adjacent tokens, or null when they never merge. */
    private fun mergeRankOf(left: Int, right: Int): Int {
        var lo = 0
        var hi = mergeCount - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val l = mergeLeft[mid]
            val r = mergeRight[mid]
            if (l == left && r == right) return mid
            if (l < left || (l == left && r < right)) lo = mid + 1 else hi = mid - 1
        }
        return -1
    }

    // ------------------------------------------------------------------ encoding
    /**
     * Metaspace(marker, prepend_scheme=always, split=true), matching the real pre-tokenizer:
     * a piece starts at the marker or at a newline run; a newline run is a piece of its own; and
     * any piece that would start with an ordinary character gets the marker prepended.
     *
     * Read off the library (`pre_tokenizer.pre_tokenize_str` and `convert_ids_to_tokens`), not
     * guessed: '   three spaces' -> ['▁','▁','▁three','▁spaces'], 'a  b' -> ['▁a','▁','▁b'],
     * "new\nline" -> ['▁new', "\n", '▁line'], "\n\n...deep" -> ["\n\n...", '▁deep'], and a tab is
     * not a boundary at all.
     */
    private fun preTokenize(marked: String): List<String> {
        val pieces = ArrayList<String>(8)
        val current = StringBuilder()
        var i = 0
        val n = marked.length
        while (i < n) {
            val ch = marked[i]
            if (ch == '\n') {
                if (current.isNotEmpty()) {
                    pieces.add(current.toString())
                    current.setLength(0)
                }
                var j = i
                while (j < n && marked[j] == '\n') j++
                pieces.add(marked.substring(i, j))
                i = j
                continue
            }
            if (ch == MARKER) {
                if (current.isNotEmpty()) pieces.add(current.toString())
                current.setLength(0)
                current.append(MARKER)
                i++
                continue
            }
            if (current.isEmpty()) current.append(MARKER)
            current.append(ch)
            i++
        }
        if (current.isNotEmpty()) pieces.add(current.toString())
        return pieces
    }

    /**
     * Merge one pre-token by lowest merge rank. Characters the vocabulary does not contain are
     * emitted as byte tokens immediately: no merge names a byte token (checked when the asset is
     * built), so they can never take part in one.
     */
    private fun bpe(piece: String): IntArray {
        val symbols = IntArray(piece.length)
        val rawBytes = ArrayList<ByteArray>()
        var count = 0
        // Iterate by code point, not by Char: a Kotlin String is UTF-16, so an emoji is two
        // Chars and tokenizing those separately would produce lone surrogates. The Python
        // reference iterates code points, and the goldens include emoji.
        var index = 0
        while (index < piece.length) {
            val codePoint = piece.codePointAt(index)
            val ch = String(Character.toChars(codePoint))
            index += Character.charCount(codePoint)
            val id = vocabId(ch)
            if (id >= 0) {
                symbols[count++] = id
            } else {
                // a virtual slot: negative, so it never matches a merge
                symbols[count++] = -(rawBytes.size + 1)
                rawBytes.add(ch.toByteArray(StandardCharsets.UTF_8))
            }
        }
        var len = count
        while (len > 1) {
            var bestRank = Int.MAX_VALUE
            var bestAt = -1
            for (i in 0 until len - 1) {
                if (symbols[i] < 0 || symbols[i + 1] < 0) continue
                val at = mergeRankOf(symbols[i], symbols[i + 1])
                if (at >= 0 && mergeRank[at] < bestRank) {
                    bestRank = mergeRank[at]
                    bestAt = at
                }
            }
            if (bestAt < 0) break
            val left = mergeLeft[bestAt]
            val right = mergeRight[bestAt]
            val result = mergeResult[bestAt]
            // merge the leftmost occurrence of that pair, as the reference implementation does
            var at = -1
            for (i in 0 until len - 1) {
                if (symbols[i] == left && symbols[i + 1] == right) { at = i; break }
            }
            if (at < 0) break
            symbols[at] = result
            for (i in at + 1 until len - 1) symbols[i] = symbols[i + 1]
            len--
        }
        val out = ArrayList<Int>(len + 4)
        for (i in 0 until len) {
            val sym = symbols[i]
            if (sym >= 0) {
                out.add(sym)
            } else {
                for (b in rawBytes[-sym - 1]) {
                    val byteToken = vocabId(String.format("<0x%02X>", b.toInt() and 0xFF))
                    out.add(if (byteToken >= 0) byteToken else unkId)
                }
            }
        }
        return out.toIntArray()
    }

    /** Tokenize text. Special tokens are never added: the prompt builder places them. */
    fun encode(text: String): IntArray {
        val out = ArrayList<Int>(text.length / 3 + 8)
        for (piece in preTokenize(text.replace(' ', MARKER))) {
            for (id in bpe(piece)) out.add(id)
        }
        return out.toIntArray()
    }

    /** Decode ids back to text, undoing the byte fallback and the marker. */
    fun decode(ids: IntArray): String {
        val out = StringBuilder()
        val pending = java.io.ByteArrayOutputStream()
        fun flush() {
            if (pending.size() > 0) {
                out.append(String(pending.toByteArray(), StandardCharsets.UTF_8))
                pending.reset()
            }
        }
        for (id in ids) {
            val text = if (id in 0 until vocabSize) token(id) else ""
            if (text.length == 6 && text.startsWith("<0x") && text.endsWith(">")) {
                pending.write(text.substring(3, 5).toInt(16))
            } else {
                flush()
                out.append(text)
            }
        }
        flush()
        return out.toString().replace(MARKER, ' ')
    }

    companion object {
        const val MARKER = '\u2581'
        private const val MAGIC = "LYATOK1\n"

        /** Load from an asset stream. The whole file is read once (about 13 MB). */
        fun fromStream(input: InputStream): LayaTokenizer {
            val data = input.use { it.readBytes() }
            val buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(8)
            buf.get(magic)
            require(String(magic, StandardCharsets.US_ASCII) == MAGIC) {
                "not a Laya tokenizer asset (bad magic)"
            }
            val vocabSize = buf.int
            val mergeCount = buf.int
            val clsId = buf.int
            val sepId = buf.int
            val maskId = buf.int
            val padId = buf.int
            val unkId = buf.int
            val blobLen = buf.int
            val blob = ByteArray(blobLen)
            buf.get(blob)
            val offsets = IntArray(vocabSize + 1)
            for (i in offsets.indices) offsets[i] = buf.int
            val byToken = IntArray(vocabSize)
            for (i in byToken.indices) byToken[i] = buf.int
            val mergeLeft = IntArray(mergeCount)
            val mergeRight = IntArray(mergeCount)
            val mergeResult = IntArray(mergeCount)
            val mergeRank = IntArray(mergeCount)
            for (i in 0 until mergeCount) {
                mergeLeft[i] = buf.int
                mergeRight[i] = buf.int
                mergeResult[i] = buf.int
                mergeRank[i] = buf.int
            }
            val maskToken = String(blob, offsets[maskId], offsets[maskId + 1] - offsets[maskId],
                StandardCharsets.UTF_8)
            return LayaTokenizer(blob, offsets, byToken, mergeLeft, mergeRight, mergeResult,
                mergeRank, vocabSize, mergeCount, clsId, sepId, maskId, padId, unkId, maskToken)
        }
    }
}