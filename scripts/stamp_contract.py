#!/usr/bin/env python3
"""Write a task's command-block contract into an exported ONNX's metadata.

    uv run scripts/stamp_contract.py out.onnx Mjlab-BallFollow-Flat-MicroDuck

Run this after `scripts/export.py`. The exported graph is `obs[1,61] ->
actions[1,14]` no matter what task produced it, so an ONNX file on its own gives
you no way to know what its command block means — and in this repo four tasks
redefine those same 13 slots. robotd's own metadata records the command *names*
(`twist`, `head_pose`, `body_pose`) but not their meaning.

Stamping puts the answer in the file itself:

    command_spec_task            Mjlab-BallFollow-Flat-MicroDuck
    cmd_twist_0_name             target_x_body
    cmd_twist_0_unit             m
    cmd_twist_0_range            -2.0,2.0
    cmd_twist_0_desc             target offset, robot forward axis
    ... (all 13 slots)

so whoever writes the command block on the robot can read it back:

    uv run scripts/check_contract.py --onnx out.onnx

Safe to run twice — stamping is idempotent, it just overwrites the keys.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from mjlab_microduck.command_contract import (  # noqa: E402
    known_tasks,
    spec_for_task,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx", type=Path, help="exported policy")
    parser.add_argument("task_id", help=f"task to stamp. Known: "
                                        f"{', '.join(known_tasks())}")
    parser.add_argument("--list", action="store_true",
                        help="print the known task ids and exit")
    args = parser.parse_args()

    if args.list:
        for tid in known_tasks():
            print(tid)
        return

    spec = spec_for_task(args.task_id)
    if spec is None:
        sys.exit(
            f"error: no CommandSpec for {args.task_id!r}.\n"
            f"       Register one from the env cfg module:\n"
            f"         from mjlab_microduck.command_contract import register, CommandSpec\n"
            f"         register(CommandSpec(task_id='{args.task_id}', ...))\n"
            f"       Known: {', '.join(known_tasks())}"
        )

    if not args.onnx.is_file():
        sys.exit(f"error: no such file: {args.onnx}")

    import onnx

    model = onnx.load(str(args.onnx))
    metadata = spec.to_metadata()

    # Overwrite in place so re-stamping does not duplicate keys.
    existing = {p.key: i for i, p in enumerate(model.metadata_props)}
    for key, value in metadata.items():
        entry = model.metadata_props.add() if key not in existing \
            else model.metadata_props[existing[key]]
        entry.key = key
        entry.value = str(value)

    onnx.save(model, str(args.onnx))

    print(f"stamped {args.onnx.name} with the {args.task_id} contract "
          f"({len(metadata)} keys)")
    print()
    print(spec.describe())


if __name__ == "__main__":
    main()
