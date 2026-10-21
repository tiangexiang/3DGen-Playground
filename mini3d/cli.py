#!/usr/bin/env python
"""mini3d entry point: prepare | train | sample | eval.

    python -m mini3d train  --config-file configs/train.yml
    python -m mini3d sample --config-file configs/train.yml --checkpoint ... --prompt "..."
    python -m mini3d eval   --config-file configs/train.yml --checkpoint ...

Config precedence: dataclass defaults < --config-file (which may chain via
`_base`) < explicit CLI flags.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import fields
from pathlib import Path

from .config import TrainConfig


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config-file", default=str(Path(__file__).resolve().parents[1] / "configs/train.yml"))
    p.add_argument("--print-config", action="store_true", help="Print resolved configuration and exit")
    p.add_argument("--output-dir")
    p.add_argument("--run-name")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override any TrainConfig field, e.g. --set num_steps=200 optimizer=muon")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")


def _coerce(cfg: TrainConfig, key: str, raw: str):
    import json as _json
    if key not in {item.name for item in fields(cfg)}:
        raise KeyError(f"unknown config key: {key}")
    cur = getattr(cfg, key)
    if isinstance(cur, bool):
        if raw.lower() not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
            raise ValueError(f"{key} expects a boolean, got {raw!r}")
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(raw)
    if isinstance(cur, float):
        return float(raw)
    if cur is None:
        import yaml
        return yaml.safe_load(raw)
    if isinstance(cur, str):
        return None if raw.lower() in ("none", "null") else raw
    return _json.loads(raw)


def build_cfg(args) -> TrainConfig:
    cfg = TrainConfig()
    if args.config_file:
        cfg.update_from_yaml(args.config_file)
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        setattr(cfg, k, _coerce(cfg, k, v))
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.run_name:
        cfg.run_name = args.run_name
    if args.no_wandb:
        cfg.wandb = False
    return cfg.resolve()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Build text tokens and camera framing from training config")
    _add_common(p_prepare)
    p_prepare.add_argument("--only", choices=["text", "framing"])
    p_prepare.add_argument("--text-batch-size", type=int, default=256)
    p_prepare.add_argument("--device", help="Device for text encoding, e.g. cuda or cpu")

    p_train = sub.add_parser("train")
    _add_common(p_train)
    p_sample = sub.add_parser("sample")
    _add_common(p_sample)
    p_sample.add_argument("--checkpoint", required=True)
    p_sample.add_argument("--checkpoint-target", choices=["model", "ema"], default="ema")
    p_sample.add_argument("--prompt", action="append", default=None)
    p_sample.add_argument("--steps", type=int, default=None)
    p_sample.add_argument("--cfg-scale", type=float, default=None)
    p_sample.add_argument("--out", default=None)
    p_eval = sub.add_parser("eval")
    _add_common(p_eval)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--checkpoint-target", choices=["model", "ema"], default="ema")
    p_eval.add_argument("--num-samples", type=int, default=None)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
    )
    try:
        cfg = build_cfg(args)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))

    if args.print_config:
        import json
        print(json.dumps(cfg.to_dict(), indent=2))
        return

    if args.command == "prepare":
        from tools.prepare_training_data import prepare
        prepare(cfg, only=args.only, text_batch_size=args.text_batch_size, device=args.device)
    elif args.command == "train":
        from mini3d.train import train
        train(cfg)
    elif args.command == "sample":
        from mini3d.sample import sample_cli
        sample_cli(cfg, args)
    elif args.command == "eval":
        from mini3d.sample import eval_cli
        eval_cli(cfg, args)


if __name__ == "__main__":
    main()
