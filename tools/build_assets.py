"""Build the Android assets and the goldens the Kotlin module is verified against.

Produces, under the Android project:
  app/src/main/assets/laya/tokenizer.bin     the tokenizer, in a format Kotlin can load
  app/src/main/assets/laya/config.json       specials, budgets, temperatures, question types
  app/src/main/assets/laya/goldens_e2e.json  real model answers, for the on-device self-test
  app/src/test/resources/laya/goldens_tokenizer.json   text -> ids
  app/src/test/resources/laya/goldens_prompt.json      state+question -> input_ids/markers/qtype
  app/src/test/resources/laya/goldens_decode.json      logits -> answers

Every golden is produced by the real implementation (HuggingFace tokenizer, `laya.common`,
the ONNX agent), never by hand, so a Kotlin mismatch is a Kotlin bug and not a stale fixture.

Run:  .venv/bin/python tools/build_assets.py
"""
from __future__ import annotations

import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bpe_ref import RefTokenizer  # noqa: E402

PY = "/root/projects/layalab"
MODEL_DIR = os.path.join(PY, "models", "laya-multilingual")
TOK_DIR = os.path.join(MODEL_DIR, "tokenizer")
ONNX_PATH = "/sdcard/Android/media/com.saeed.cosmo/laya/laya.fp32.onnx"
PROJECT = "/root/projects/laya-module"
ASSETS = os.path.join(PROJECT, "app", "src", "main", "assets", "laya")
RESOURCES = os.path.join(PROJECT, "app", "src", "test", "resources", "laya")

SPIKE = "\u2581"


def build_asset(ref: RefTokenizer, out_path: str) -> dict:
    """Pack the vocabulary, the merge ranks and the special tokens into one loadable file."""
    vocab = ref.vocab
    n_vocab = max(vocab.values()) + 1
    tokens = [None] * n_vocab
    for token, tid in vocab.items():
        tokens[tid] = token
    missing = [i for i, t in enumerate(tokens) if t is None]
    if missing:
        raise SystemExit(f"vocabulary has gaps at {missing[:5]} — the id space is not dense")

    # merges are defined over token strings; both sides AND the result must be real tokens, so
    # Kotlin can merge on plain integer ids instead of concatenating strings
    left_ids, right_ids, result_ids, ranks = [], [], [], []
    for rank, (left, right) in enumerate(ref.merges):
        if left not in vocab or right not in vocab:
            raise SystemExit(f"merge {rank} names a token outside the vocabulary: {left!r} {right!r}")
        if left + right not in vocab:
            raise SystemExit(f"merge {rank} has no result token in the vocabulary: {left + right!r}")
        left_ids.append(vocab[left])
        right_ids.append(vocab[right])
        result_ids.append(vocab[left + right])
        ranks.append(rank)
    # sorted by (left, right) so Kotlin can binary-search a rank
    order = sorted(range(len(left_ids)), key=lambda i: (left_ids[i], right_ids[i]))
    pairs = [(left_ids[i], right_ids[i], result_ids[i], ranks[i]) for i in order]
    for a, b in zip(pairs, pairs[1:]):
        if a[:2] == b[:2]:
            raise SystemExit(f"duplicate merge pair {a[:2]} — ranks would be ambiguous")

    blob = bytearray()
    offsets = [0]
    for token in tokens:
        blob += token.encode("utf-8")
        offsets.append(len(blob))
    by_token = sorted(range(n_vocab), key=lambda i: tokens[i].encode("utf-8"))

    with open(out_path, "wb") as handle:
        handle.write(b"LYATOK1\n")
        handle.write(struct.pack("<II", n_vocab, len(pairs)))
        handle.write(struct.pack("<IIIII", ref.cls_token_id, ref.sep_token_id, ref.mask_token_id,
                                 ref.pad_token_id, ref.unk_token_id))
        handle.write(struct.pack("<I", len(blob)))
        handle.write(bytes(blob))
        handle.write(struct.pack("<%dI" % len(offsets), *offsets))
        handle.write(struct.pack("<%dI" % n_vocab, *by_token))
        for left, right, result, rank in pairs:
            handle.write(struct.pack("<IIII", left, right, result, rank))
    return {"n_vocab": n_vocab, "n_merges": len(pairs), "blob_bytes": len(blob),
            "file_bytes": os.path.getsize(out_path)}


def main() -> int:
    import laya
    from laya.common import QTYPES, build_sequence, render_options, unpermute_probs
    from laya.onnx_agent import ONNXAgent
    from transformers import AutoTokenizer

    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(RESOURCES, exist_ok=True)

    hf = AutoTokenizer.from_pretrained(TOK_DIR)
    ref = RefTokenizer(os.path.join(TOK_DIR, "tokenizer.json"),
                       os.path.join(TOK_DIR, "tokenizer_config.json"))
    agent = ONNXAgent(MODEL_DIR, onnx_path=ONNX_PATH)
    cfg = agent.cfg
    print(f"laya {laya.__version__} | cfg max_len={cfg['max_len']} head_max_len={cfg['head_max_len']}")

    # ------------------------------------------------------------------ 1. the asset
    info = build_asset(ref, os.path.join(ASSETS, "tokenizer.bin"))
    print("tokenizer.bin", info)

    with open(os.path.join(ASSETS, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "format": 1,
            "checkpoint": "convaiinnovations/laya-multilingual",
            "encoder": cfg.get("encoder"),
            "max_len": cfg.get("max_len", 1024),
            "head_max_len": cfg.get("head_max_len", 256),
            "temperature": cfg.get("temperature", [1.0, 1.0, 1.0]),
            "temperature_by_options": cfg.get("temperature_by_options", {}),
            "qtypes": QTYPES,
            "cls_token_id": ref.cls_token_id,
            "sep_token_id": ref.sep_token_id,
            "mask_token_id": ref.mask_token_id,
            "pad_token_id": ref.pad_token_id,
            "unk_token_id": ref.unk_token_id,
            "mask_token": ref.mask_token,
        }, handle, indent=2)

    # ------------------------------------------------------------- 2. tokenizer goldens
    from check_bpe import CASES, KNOWN_DIVERGENCES

    samples = []
    cases_file = os.path.join(PY, "app", "cases.py")
    if os.path.exists(cases_file):
        sys.path.insert(0, os.path.join(PY, "app"))
        try:
            from cases import LABS
            for lab in LABS:
                for sample in lab["samples"]:
                    samples.append(sample["state"])
        except Exception as exc:  # noqa: BLE001
            print("note: lab samples unavailable:", exc)

    prompts = []
    for qtype in ("choice", "score", "noul"):
        for ins in ("Which department should handle this?", "How urgent is this?",
                    "Does the user ask for a refund?", "آیا این پیام مشکوک است؟"):
            prompts.append(f"{qtype} question: {ins}")
    for extra in (" billing: invoices, payments, refunds", " level 0: not urgent",
                  " false: no, the statement does not hold", "true: yes, the statement holds",
                  " level 3: blocking, everything is down"):
        prompts.append(extra)

    texts = list(dict.fromkeys(CASES + samples + prompts))
    tokenizer_goldens = [{"text": t, "ids": hf(t, add_special_tokens=False)["input_ids"]}
                         for t in texts]
    with open(os.path.join(RESOURCES, "goldens_tokenizer.json"), "w", encoding="utf-8") as handle:
        json.dump({"cases": tokenizer_goldens,
                   "known_divergences": [{"text": t, "ids": hf(t, add_special_tokens=False)["input_ids"]}
                                         for t in KNOWN_DIVERGENCES]},
                  handle, ensure_ascii=False, indent=1)
    print(f"goldens_tokenizer.json: {len(tokenizer_goldens)} cases, "
          f"{len(KNOWN_DIVERGENCES)} known divergences")

    # ---------------------------------------------------------------- 3. prompt goldens
    states = [
        "Hi, we were billed twice for March. Please refund the duplicate today.",
        "سلام، مبلغ اشتراک ماه گذشته دو بار از حساب من کم شده است.",
        "Your API has been returning 500 errors for the last hour.",
        '{"message": "La aplicación se cierra cada vez que abro la configuración."}',
        "x" * 3000,
    ]
    # Written in the public shape a caller uses, so the Kotlin test exercises the real parser;
    # the SDK's own `_to_internal` produces what `build_sequence` takes.
    public_questions = [
        {"type": "choice", "instructions": "Which department should handle this request?",
         "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                      "other": "everything else"}},
        {"type": "score", "instructions": "How urgent is this?",
         "criteria": ["not urgent", "soon", "blocking"]},
        {"type": "noul", "instructions": "Does the user request a refund?"},
        {"type": "noul", "instructions": "Is this suspicious?",
         "criteria": {"false": "an ordinary message", "true": "a fraud attempt"}},
        {"type": "noul", "instructions": "Is this positive?",
         "criteria": {"false": "negative", "true": "positive"}, "labels": {"false": "B", "true": "A"}},
        {"type": "choice", "instructions": "What is this about?",
         "criteria": {f"label_{i}": f"description number {i} with a few words" for i in range(12)}},
        {"type": "choice", "instructions": "Which team?",
         "criteria": ["billing", "technical", "sales"], "option_order": [2, 0, 1]},
    ]
    internal_questions = [ONNXAgent._to_internal(q) for q in public_questions]
    prompt_goldens = []
    for state in states:
        for public, internal in zip(public_questions, internal_questions):
            for max_len, head_max_len in ((cfg["max_len"], cfg["head_max_len"]), (256, 96)):
                ids, markers, stats = build_sequence(hf, state, internal, max_len, head_max_len,
                                                     option_order=internal.get("option_order"),
                                                     return_stats=True)
                prompt_goldens.append({
                    "state": state, "question": public, "max_len": max_len,
                    "head_max_len": head_max_len, "ids": ids, "markers": markers,
                    "qtype": QTYPES[internal["t"]], "options": stats,
                    "rendered_options": render_options(internal),
                })
    with open(os.path.join(RESOURCES, "goldens_prompt.json"), "w", encoding="utf-8") as handle:
        json.dump(prompt_goldens, handle, ensure_ascii=False, indent=1)
    print(f"goldens_prompt.json: {len(prompt_goldens)} sequences")

    # ---------------------------------------------------------------- 4. decode goldens
    # Fixed logits, decoded by the SDK's own formulas: this is what proves the Kotlin
    # softmax / argmax / expected-level / noul maths, independently of the model.
    import numpy as np

    rng = np.random.default_rng(7)
    decode_goldens = []
    for public, internal in zip(public_questions, internal_questions):
        k = len(render_options(internal))
        for scale in (1.0, 4.0):
            logits = (rng.normal(size=k) * scale).astype(np.float32)
            act = np.array([rng.normal(), rng.normal()], dtype=np.float32)
            z = logits / 1.0
            p = np.exp(z - z.max())
            p = p / p.sum()
            # The row comes back in slot order; the SDK un-permutes it back into the caller's
            # option order before reading it, so the golden must too.
            p = unpermute_probs(p, internal.get("option_order"))
            act_p = np.exp(act - act.max())
            act_p = act_p / act_p.sum()
            expected = {"act_probability": round(float(act_p[0]), 4)}
            if internal["t"] == "choice":
                keys = list(internal["crit"].keys())
                expected.update({
                    "type": "choice", "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                })
            elif internal["t"] == "score":
                expected.update({
                    "type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                    "legend": {str(i): c for i, c in enumerate(internal["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                })
            else:
                expected.update({"type": "noul", "noul": round(float(p[1]), 4),
                                 "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4)})
            decode_goldens.append({"question": public, "logits": [float(v) for v in logits],
                                   "act_logits": [float(v) for v in act], "expected": expected})
    with open(os.path.join(RESOURCES, "goldens_decode.json"), "w", encoding="utf-8") as handle:
        json.dump(decode_goldens, handle, ensure_ascii=False, indent=1)
    print(f"goldens_decode.json: {len(decode_goldens)} rows")

    # ------------------------------------------------------------------ 5. e2e goldens
    e2e_questions = [
        {"department": {"type": "choice", "instructions": "Which department should handle this?",
                        "criteria": {"billing": "invoices, payments, refunds",
                                     "technical": "bugs, outages, system errors",
                                     "other": "everything else"}},
         "urgent": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "soon", "blocking"]},
         "refund": {"type": "noul", "instructions": "Does the user request a refund?"}},
        {"is_scam": {"type": "noul",
                     "instructions": "Is this message trying to trick the reader into sending money, clicking a link, or revealing personal details?"},
         "topic": {"type": "choice", "instructions": "What is this message about?",
                   "criteria": {"billing": "charges or refunds", "delivery": "parcels",
                                "account": "login or access", "other": "anything else"}}},
        {"sentiment": {"type": "score", "instructions": "How does the sender feel?",
                       "criteria": ["neutral", "mildly unhappy", "frustrated", "angry"]}},
    ]
    e2e_states = [
        {"message": "Hi, we were billed twice for March. Please refund the duplicate today."},
        {"message": "سلام، مبلغ اشتراک ماه گذشته دو بار از حساب من کم شده است. لطفاً برگردانید."},
        {"message": "Your parcel is held at customs. Pay the 25,000 T fee within 12 hours: http://x.example"},
        {"message": "مبلغ ۵۰۰٫۰۰۰ ریال از حساب شما کسر شد. موجودی: ۱۲٫۴۵۰٫۰۰۰ ریال"},
        {"message": "It works, I suppose. The export is slower than it used to be."},
    ]
    e2e = []
    for state in e2e_states:
        for qs in e2e_questions:
            result = agent.system_one(state, qs)
            e2e.append({"state": state, "questions": qs, "expected": result["answers"]})
    with open(os.path.join(ASSETS, "goldens_e2e.json"), "w", encoding="utf-8") as handle:
        json.dump(e2e, handle, ensure_ascii=False, indent=1)
    print(f"goldens_e2e.json: {len(e2e)} real model calls")

    print("\nASSETS BUILT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())