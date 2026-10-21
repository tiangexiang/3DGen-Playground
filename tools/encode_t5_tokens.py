"""Encode captions as frozen FLAN-T5 token sequences.

Outputs keys.json (caption stems), tokens.npy (N,L,D float16), mask.npy
(N,L uint8), and meta.json. Existing compatible outputs resume from keys.json.
Padding is zeroed; the training model replaces it with its learned mask token.

Use python prepare.py to share the training configuration, or run this
script directly with --help for batch size, device, and sharding options."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import training_paths as settings  # noqa: E402

T5_HIDDEN = {"google/flan-t5-small": 512, "google/flan-t5-base": 768,
             "google/flan-t5-large": 1024, "google/flan-t5-xl": 2048}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--t5-name", default=settings.T5_NAME)
    p.add_argument("--output", default=str(settings.T5_TOKEN_ROOT))
    p.add_argument("--obj-list", default=str(settings.OBJ_LIST))
    p.add_argument("--captions", default=str(settings.CAPTIONS_PATH))
    p.add_argument("--prompt-length", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--device", default=None)
    p.add_argument("--flush-every", type=int, default=20_000,
                   help="Persist keys.json + flush memmaps every N captions.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    return p.parse_args()


def _open_outputs(out_dir: Path, n_total: int, seq_len: int, hidden: int):
    """Create or reattach the memmaps. Shape mismatch raises rather than corrupts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens_path, mask_path = out_dir / "tokens.npy", out_dir / "mask.npy"
    keys_path = out_dir / "keys.json"
    expected = {
        tokens_path: (np.float16, (n_total, seq_len, hidden)),
        mask_path:   (np.uint8,   (n_total, seq_len)),
    }
    done_keys: list[str] = []
    if all(p.exists() for p in expected):
        for path, (dtype, shape) in expected.items():
            ex = np.load(path, mmap_mode="r")
            if ex.dtype != dtype or ex.shape != shape:
                raise ValueError(
                    f"Existing {path} has dtype={ex.dtype}, shape={ex.shape}; expected "
                    f"dtype={dtype}, shape={shape}. Delete {out_dir} to start fresh."
                )
            del ex
        if keys_path.exists():
            done_keys = json.loads(keys_path.read_text())
        mode = "r+"
    elif any(p.exists() for p in expected):
        raise ValueError(f"{out_dir} is partially populated. Delete it to start fresh.")
    else:
        mode = "w+"
    tokens_mm = np.lib.format.open_memmap(tokens_path, mode=mode, dtype=np.float16,
                                          shape=(n_total, seq_len, hidden))
    mask_mm = np.lib.format.open_memmap(mask_path, mode=mode, dtype=np.uint8,
                                        shape=(n_total, seq_len))
    return tokens_mm, mask_mm, done_keys


def _persist_keys(out_dir: Path, keys: list[str]) -> None:
    keys_path = out_dir / "keys.json"
    tmp = keys_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(keys))
    os.replace(tmp, keys_path)


def _write_meta(out_dir: Path, meta: dict) -> None:
    """Record what produced this directory, so the loader can refuse a mismatch."""
    tmp = (out_dir / "meta.json").with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out_dir / "meta.json")


class T5Encoder:
    def __init__(self, name: str, device: torch.device, dtype: torch.dtype, max_len: int):
        from transformers import AutoTokenizer, T5EncoderModel
        self.max_len = max_len
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=max_len)
        self.model = T5EncoderModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden = self.model.config.d_model

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
        tok = self.tokenizer(texts, padding="max_length", max_length=self.max_len,
                             truncation=True, return_tensors="pt")
        ids = tok["input_ids"].to(self.device)
        mask = tok["attention_mask"].to(self.device)
        out = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        # Zero the padded rows. They are replaced by the learned mask_token at
        # training time anyway, but leaving T5's arbitrary pad activations on disk
        # invites a silent bug the day someone forgets to apply the mask.
        out = out * mask[:, :, None].to(out.dtype)
        n_trunc = int((tok["attention_mask"].sum(1) == self.max_len).sum())
        return (out.float().cpu().numpy().astype(np.float16),
                mask.to(torch.uint8).cpu().numpy(), n_trunc)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    print(f"Device: {device}  dtype: {args.dtype}  model: {args.t5_name}", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    obj_data = json.loads(Path(args.obj_list).read_text())
    obj_keys = {t.removesuffix(".tar.gz") for t in obj_data.values()}
    captions = json.loads(Path(args.captions).read_text())
    print(f"  {len(obj_keys):,} object stems, {len(captions):,} captions", flush=True)

    selected = sorted(obj_keys & captions.keys())
    print(f"Caption coverage: {len(selected):,}/{len(obj_keys):,} "
          f"({len(selected) / max(1, len(obj_keys)) * 100:.2f}%)", flush=True)

    if args.num_shards > 1:
        before = len(selected)
        selected = [k for i, k in enumerate(selected) if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {before:,} -> {len(selected):,}")
    if args.limit is not None:
        selected = selected[: args.limit]
    n_total = len(selected)
    if n_total == 0:
        raise ValueError("No captions to encode after filtering / sharding.")

    hidden = T5_HIDDEN.get(args.t5_name)
    encoder = T5Encoder(args.t5_name, device, dtype, args.prompt_length)
    if hidden is not None and encoder.hidden != hidden:
        raise ValueError(f"{args.t5_name} reports d_model={encoder.hidden}, expected {hidden}")
    hidden = encoder.hidden

    tokens_mm, mask_mm, done = _open_outputs(out_dir, n_total, args.prompt_length, hidden)
    if done:
        if done != selected[: len(done)]:
            raise ValueError(
                f"Resume mismatch: keys.json prefix does not match the selected key "
                f"list. Delete {out_dir} to start fresh."
            )
        start = len(done)
        print(f"Resuming from index {start:,} / {n_total:,}", flush=True)
    else:
        start = 0
        print(f"Fresh run: encoding {n_total:,} captions -> {out_dir} "
              f"({n_total * args.prompt_length * hidden * 2 / 1e9:.1f} GB)", flush=True)

    _write_meta(out_dir, {
        "t5_name": args.t5_name, "prompt_length": args.prompt_length,
        "hidden": hidden, "n_total": n_total, "dtype": "float16",
        "encoder_dtype": args.dtype, "captions": str(args.captions),
        "obj_list": str(args.obj_list), "num_shards": args.num_shards,
        "shard_id": args.shard_id,
    })

    t0 = time.time()
    truncated = 0
    for i in range(start, n_total, args.batch_size):
        batch_keys = selected[i: i + args.batch_size]
        texts = [captions[k] for k in batch_keys]
        tok, msk, n_tr = encoder.encode(texts)
        tokens_mm[i: i + len(batch_keys)] = tok
        mask_mm[i: i + len(batch_keys)] = msk
        truncated += n_tr
        done_now = i + len(batch_keys)
        if done_now % args.flush_every < args.batch_size or done_now >= n_total:
            tokens_mm.flush(); mask_mm.flush()
            _persist_keys(out_dir, selected[:done_now])
            el = time.time() - t0
            rate = (done_now - start) / max(el, 1e-6)
            eta = (n_total - done_now) / max(rate, 1e-6)
            print(f"  {done_now:,}/{n_total:,}  {rate:.0f}/s  "
                  f"elapsed {el / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)

    tokens_mm.flush(); mask_mm.flush()
    _persist_keys(out_dir, selected)
    print(f"Done: {n_total:,} captions in {(time.time() - t0) / 60:.1f} min. "
          f"Truncated at {args.prompt_length} tokens: {truncated:,} "
          f"({100 * truncated / max(n_total, 1):.2f}%)", flush=True)


if __name__ == "__main__":
    main()
