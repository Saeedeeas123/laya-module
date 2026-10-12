"""Read `tokenizer.bin` the way Kotlin will, and reproduce the goldens from it alone.

This is the format's own test: it loads ONLY the packed asset (no tokenizer.json, no
HuggingFace), does the vocabulary and merge lookups by binary search over raw bytes exactly
as the Kotlin loader does, and re-encodes every golden text. If this passes, the asset is
sufficient and the Kotlin port has a proven specification to follow.

Run:  .venv/bin/python tools/check_asset.py
"""
from __future__ import annotations

import json
import os
import struct
import sys

ASSET = "/root/projects/laya-module/app/src/main/assets/laya/tokenizer.bin"
GOLDENS = "/root/projects/laya-module/app/src/test/resources/laya/goldens_tokenizer.json"

SPIKE = "\u2581"


class AssetTokenizer:
    """The Kotlin loader's algorithm, in Python."""

    def __init__(self, path: str) -> None:
        with open(path, "rb") as handle:
            self.data = handle.read()
        if self.data[:8] != b"LYATOK1\n":
            raise SystemExit("bad magic")
        off = 8
        (self.n_vocab, self.n_merges) = struct.unpack_from("<II", self.data, off)
        off += 8
        (self.cls_id, self.sep_id, self.mask_id, self.pad_id, self.unk_id) = \
            struct.unpack_from("<IIIII", self.data, off)
        off += 20
        (blob_len,) = struct.unpack_from("<I", self.data, off)
        off += 4
        self.blob = self.data[off:off + blob_len]
        off += blob_len
        self.offsets = struct.unpack_from("<%dI" % (self.n_vocab + 1), self.data, off)
        off += 4 * (self.n_vocab + 1)
        self.by_token = struct.unpack_from("<%dI" % self.n_vocab, self.data, off)
        off += 4 * self.n_vocab
        self.merges = []
        for _ in range(self.n_merges):
            self.merges.append(struct.unpack_from("<IIII", self.data, off))
            off += 16
        self.mask_token = self.token(self.mask_id)

    # ---------------------------------------------------------------- vocabulary
    def token(self, tid: int) -> str:
        return self.blob[self.offsets[tid]:self.offsets[tid + 1]].decode("utf-8", "replace")

    def token_bytes(self, tid: int) -> bytes:
        return self.blob[self.offsets[tid]:self.offsets[tid + 1]]

    def vocab_id(self, text: str) -> int:
        """Binary search the sorted-by-token index, comparing raw UTF-8 bytes."""
        want = text.encode("utf-8")
        lo, hi = 0, self.n_vocab - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            tid = self.by_token[mid]
            have = self.token_bytes(tid)
            if have == want:
                return tid
            if have < want:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    def merge(self, left: int, right: int):
        """Binary search the (left, right) ordered merge table."""
        lo, hi = 0, self.n_merges - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            l, r, result, rank = self.merges[mid]
            if l == left and r == right:
                return rank, result
            if (l, r) < (left, right):
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    # ------------------------------------------------------------------ encoding
    def pre_tokenize(self, text: str) -> list[str]:
        pieces, current, i, n = [], "", 0, len(text)
        while i < n:
            ch = text[i]
            if ch == "\n":
                if current:
                    pieces.append(current)
                    current = ""
                j = i
                while j < n and text[j] == "\n":
                    j += 1
                pieces.append(text[i:j])
                i = j
                continue
            if ch == SPIKE:
                if current:
                    pieces.append(current)
                current = SPIKE
                i += 1
                continue
            if not current:
                current = SPIKE
            current += ch
            i += 1
        if current:
            pieces.append(current)
        return pieces

    def bpe(self, piece: str) -> list[int]:
        """Merge on integer ids; characters outside the vocabulary become byte tokens."""
        symbols: list[int] = []
        raw: dict[int, bytes] = {}
        next_virtual = -1
        for ch in piece:
            tid = self.vocab_id(ch)
            if tid >= 0:
                symbols.append(tid)
            else:
                # never merges (no merge names a byte token), so give it a virtual slot
                raw[next_virtual] = ch.encode("utf-8")
                symbols.append(next_virtual)
                next_virtual -= 1
        while len(symbols) > 1:
            best_rank, best_at, best_result = None, -1, None
            for i in range(len(symbols) - 1):
                if symbols[i] < 0 or symbols[i + 1] < 0:
                    continue
                found = self.merge(symbols[i], symbols[i + 1])
                if found is None:
                    continue
                rank, result = found
                if best_rank is None or rank < best_rank:
                    best_rank, best_at, best_result = rank, i, result
            if best_at < 0:
                break
            symbols = symbols[:best_at] + [best_result] + symbols[best_at + 2:]
        out: list[int] = []
        for sym in symbols:
            if sym >= 0:
                out.append(sym)
            else:
                for byte in raw[sym]:
                    out.append(self.vocab_id("<0x%02X>" % byte))
        return out

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self.pre_tokenize(text.replace(" ", SPIKE)):
            ids.extend(self.bpe(piece))
        return ids


def main() -> int:
    tok = AssetTokenizer(ASSET)
    print(f"loaded asset: vocab={tok.n_vocab} merges={tok.n_merges} "
          f"specials cls={tok.cls_id} sep={tok.sep_id} mask={tok.mask_id} pad={tok.pad_id} "
          f"unk={tok.unk_id}")
    print(f"spot checks: token(0)={tok.token(0)!r} vocab_id('hello')={tok.vocab_id('hello')} "
          f"vocab_id('nonexistent-zzz')={tok.vocab_id('nonexistent-zzz')}")

    with open(GOLDENS, "r", encoding="utf-8") as handle:
        goldens = json.load(handle)

    bad = 0
    for case in goldens["cases"]:
        got = tok.encode(case["text"])
        if got != case["ids"]:
            bad += 1
            if bad <= 3:
                print(f"MISMATCH {case['text'][:60]!r}\n  want {case['ids'][:12]}\n  got  {got[:12]}")
    print(f"\n{len(goldens['cases']) - bad}/{len(goldens['cases'])} goldens reproduced from the asset alone")

    diverged = sum(1 for case in goldens["known_divergences"] if tok.encode(case["text"]) != case["ids"])
    print(f"known divergences still divergent: {diverged}/{len(goldens['known_divergences'])}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())