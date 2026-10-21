"""Grounded phrase mapping and target-to-text bias for the edit forward.

Parser adapted from capitan01R's MIT-licensed Krea2T Enhancer.
See LICENSE-phrase-weights-MIT. Integration code follows this repository's license.
"""
import math
import numbers
from pathlib import Path

OPTION_KEY = "krea2edit_phrase_weights"


def map_phrases(clean, spans, template, tokens, tokenizer):
    """Map exact native placeholder tokens; image expansion is resolved afterwards."""
    rendered = template.format(clean)
    offset = len(template.split("{}", 1)[0])
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    batches = tokens.get("qwen3vl_4b")
    if batches is None or len(batches) != 1 or len(batches[0]) != len(ids):
        raise ValueError("Use the native Krea2 Qwen3-VL 4B encoder.")
    image_indices = []
    for i, (pair, expected) in enumerate(zip(batches[0], ids)):
        value = pair[0]
        if isinstance(value, dict):
            if value.get("type") != "image" or expected != 151655:
                raise ValueError("Unsupported embedded token in phrase layout.")
            image_indices.append(i)
        elif not isinstance(value, numbers.Integral) or int(value) != expected:
            raise ValueError("Native Krea2 tokenizer differs from the phrase layout.")
    starts = [i for i, value in enumerate(ids) if value == 151644]
    if len(starts) < 2 or ids[starts[1] + 1:starts[1] + 3] != [872, 198]:
        raise ValueError("Unsupported Krea2 system/user prefix.")
    trim = starts[1] + 3
    mapped, owners = [], set()
    for span in spans:
        start, end = offset + span["start"], offset + span["end"]
        rows = []
        for i, (a, b) in enumerate(offsets):
            if a >= end or b <= start:
                continue
            if rendered[a:start].strip() or rendered[end:b].strip():
                raise ValueError("Mark complete tokenizer pieces, not partial words.")
            if i < trim or (image_indices and i <= image_indices[-1]) or i in owners:
                raise ValueError("Weight separate text phrases after all image placeholders.")
            owners.add(i)
            rows.append(i)
        if not rows:
            raise ValueError("Weighted phrase contains no tokens.")
        mapped.append({**span, "raw_rows": rows})
    return {"raw_count": len(ids), "trim": trim, "images": image_indices, "phrases": mapped}


def finish_mapping(layout, count):
    """Native Krea2 removes only a prefix; all image expansions precede the prompt."""
    plain = layout["raw_count"] - layout["trim"]
    if (not layout["images"] and count != plain) or count < plain:
        raise ValueError("Encoded conditioning rows differ from the native Krea2 layout.")
    biases, report = [], []
    for span in layout["phrases"]:
        rows = [count - (layout["raw_count"] - i) for i in span["raw_rows"]]
        if any(i < 0 or i >= count for i in rows):
            raise ValueError("Phrase row outside conditioning.")
        if span["weight"] != 1:
            bias = math.log(span["weight"]) if span["weight"] > 0 else -80.0
            biases.extend((i, bias) for i in rows)
        report.append({k: v for k, v in span.items() if k != "raw_rows"} | {"rows": rows})
    return tuple(biases), report


def encode_phrases(clip, clean, spans, tokens, template):
    import comfy.hooks
    from comfy.text_encoders import krea2
    from transformers import Qwen2TokenizerFast

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        Path(krea2.__file__).parent / "qwen25_tokenizer", local_files_only=True)
    layout = map_phrases(clean, spans, template, tokens, tokenizer)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    output, reports = [], []
    for tensor, metadata in conditioning:
        if tensor.ndim != 3 or tensor.shape[-1] != 12 * 2560:
            raise ValueError("Expected Krea2's native twelve-layer text conditioning.")
        biases, phrases = finish_mapping(layout, tensor.shape[1])
        item = [[tensor, metadata]]
        if biases:
            hook = comfy.hooks.TransformerOptionsHook(
                {OPTION_KEY: {"count": tensor.shape[1], "biases": biases}},
                hook_scope=comfy.hooks.EnumHookScope.HookedOnly)
            hook.hook_id = OPTION_KEY
            group = comfy.hooks.HookGroup()
            group.add(hook)
            item = comfy.hooks.set_hooks_for_conditioning(item, group)
        output.extend(item)
        reports.append({"active": bool(biases), "token_count": tensor.shape[1], "phrases": phrases})
    return output, reports


def apply_phrase_bias(attn_bias, payload, txtlen, srclen, tgtlen, combined):
    if payload is None:
        return attn_bias
    if payload["count"] != txtlen:
        raise ValueError("Text rows changed after grounded phrase encoding.")
    biases = payload["biases"]
    if not biases:
        return attn_bias
    if any(row < 0 or row >= txtlen for row, _ in biases):
        raise ValueError("Phrase row outside text conditioning.")
    length = txtlen + srclen + tgtlen
    if attn_bias is None:
        attn_bias = combined.new_zeros((1, 1, length, length))
    else:
        attn_bias = attn_bias.clone()
    for row, value in biases:
        attn_bias[:, :, txtlen + srclen:, row] += value
    return attn_bias
