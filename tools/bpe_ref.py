"""Reference implementation of Laya's tokenizer pipeline, written from tokenizer.json.

Why this exists: the Android module must tokenize in Kotlin, with no Python and no
HuggingFace. Writing the algorithm here first means the algorithm itself gets debugged
against the real tokenizer at Python speed, and Kotlin only ever receives a design that is
already proven to match. `tools/check_bpe.py` is the harness that proves it.

The pipeline, read out of `tokenizer.json` (not guessed):
    normalizer      Replace(" " -> "▁")
    pre_tokenizer   Metaspace(replacement="▁", prepend_scheme="always", split=true)
    model           BPE(vocab=256000, merges=580604, byte_fallback=true, ignore_merges=false)
    post_processor  TemplateProcessing <bos> A <eos>  -- only when add_special_tokens=True

Laya always calls the tokenizer with add_special_tokens=False, so the post-processor never
runs and the special tokens are placed by the prompt builder instead.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

SPIKE = "\u2581"          # the SentencePiece space marker, written as ▁ below


class RefTokenizer:
    """A from-scratch BPE tokenizer for the laya-multilingual checkpoint."""

    def __init__(self, tokenizer_json: str, tokenizer_config_json: Optional[str] = None) -> None:
        with open(tokenizer_json, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
        model = spec["model"]
        self.vocab: Dict[str, int] = model["vocab"]
        self.merges: List[Tuple[str, str]] = [(m[0], m[1]) for m in model["merges"]]
        self.ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(self.merges)}
        self.byte_fallback: bool = bool(model.get("byte_fallback"))
        self.ignore_merges: bool = bool(model.get("ignore_merges"))
        self.unk_token: str = model["unk_token"]
        self.fuse_unk: bool = bool(model.get("fuse_unk", False))
        # added tokens are matched literally, before the model runs
        self.added: Dict[str, int] = {a["content"]: a["id"] for a in spec.get("added_tokens", [])}
        self.byte_tokens = {i: "<0x%02X>" % i for i in range(256)}

        cfg = {}
        if tokenizer_config_json:
            with open(tokenizer_config_json, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        self.cls_token_id = cfg.get("cls_token_id", 2)
        self.sep_token_id = cfg.get("sep_token_id", 1)
        self.mask_token_id = cfg.get("mask_token_id", 4)
        self.mask_token = cfg.get("mask_token", "<mask>")
        self.pad_token_id = cfg.get("pad_token_id", 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 3)

    # ------------------------------------------------------------------ pipeline
    def normalize(self, text: str) -> str:
        return text.replace(" ", SPIKE)

    def pre_tokenize(self, text: str) -> List[str]:
        """Metaspace(replacement=▁, prepend_scheme=always, split=true), read off the real one.

        The behaviour below was not guessed: it was read back from the library with
        `backend_tokenizer.pre_tokenizer.pre_tokenize_str` and `convert_ids_to_tokens` on
        tricky inputs, because three plausible readings of the config were all wrong.

            '   three spaces'  -> ['▁', '▁', '▁three', '▁spaces']
            'a  b'             -> ['▁a', '▁', '▁b']
            'new\\nline'        -> ['▁new', '\\n', '▁line']       (a newline run is its own piece
                                                             and the text after it restarts
                                                             with a marker)
            '\\n\\n...deep'      -> ['\\n\\n...', '▁deep']        (no marker before a newline run)
            '▁▁▁▁▁'            -> ['▁', '▁', '▁', '▁', '▁']       (one piece per lone marker)
            'tab\\there'        -> ['▁tab\\there']               (a tab is NOT a boundary)

        So: a piece starts at a marker or at a newline run; a newline run is a piece of its own;
        and any piece that would start with an ordinary character gets the marker prepended.
        """
        pieces: List[str] = []
        current = ""
        i = 0
        n = len(text)
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

    def bpe(self, piece: str) -> List[str]:
        """Merge one pre-token by lowest merge rank, then byte-fallback what is left."""
        if self.ignore_merges and piece in self.vocab:
            return [piece]
        symbols: List[str] = list(piece)
        if not symbols:
            return []
        while len(symbols) > 1:
            best_rank = None
            best_at = -1
            for i in range(len(symbols) - 1):
                rank = self.ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_at = rank, i
            if best_at < 0:
                break
            symbols = symbols[:best_at] + [symbols[best_at] + symbols[best_at + 1]] + symbols[best_at + 2:]
        return symbols

    def to_ids(self, symbols: Sequence[str]) -> List[int]:
        out: List[int] = []
        for symbol in symbols:
            if symbol in self.vocab:
                out.append(self.vocab[symbol])
                continue
            if symbol in self.added:
                out.append(self.added[symbol])
                continue
            if self.byte_fallback:
                for byte in symbol.encode("utf-8"):
                    name = self.byte_tokens[byte]
                    out.append(self.vocab.get(name, self.unk_token_id))
                continue
            out.append(self.unk_token_id)
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids: List[int] = []
        for piece in self.pre_tokenize(self.normalize(text)):
            ids.extend(self.to_ids(self.bpe(piece)))
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        reverse = {v: k for k, v in self.vocab.items()}
        out = ""
        for i in ids:
            token = reverse.get(i, "")
            if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
                out += bytes([int(token[3:5], 16)]).decode("utf-8", errors="replace")
            else:
                out += token
        return out.replace(SPIKE, " ")