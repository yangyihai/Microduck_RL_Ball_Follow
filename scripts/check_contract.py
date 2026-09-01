#!/usr/bin/env python3
"""Check a trained policy against the contracts it has to satisfy to deploy.

    uv run scripts/check_contract.py Mjlab-BallFollow-Flat-MicroDuck
    uv run scripts/check_contract.py --onnx out.onnx

The runtime will reject a policy with the wrong observation width, but several
other ways of producing a policy that loads fine and then behaves wrongly are
completely silent. This checks the ones that have actually bitten this repo:

  1. **actor obs must be 61** — robotd's `Observation::build` is width-checked,
     so this is the one failure that is at least loud. Checked anyway, because
     it is cheap and it is the first thing to rule out.
  2. **the command block must be declared** — the twist slots are redefined per
     task in this repo (velocity / sitstand flag / ball-follow target), all
     61 wide, all exporting cleanly. Feed one the others' command block and the
     robot does something confidently wrong. A task with no `CommandSpec` is a
     task whose command block nobody has written down.
  3. **the symmetry table must match the declared mirror signs** — the velocity
     table mirrors twist as [vx, -vy, -vyaw] and was reused for BallFollow,
     which negates the stand-off distance, i.e. it trained the policy that the
     mirror image of "hold 0.35 m" is "hold -0.35 m". Widths matched, training
     ran, nothing errored. `check_mirror_signs` catches exactly this.
  4. **the exported ONNX must be obs[1,61] -> actions[1,14]** — the same check
     robotd performs at load, run before deployment rather than on the robot.

Exits non-zero on any failure, so it can gate a CI step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from mjlab_microduck.command_contract import (  # noqa: E402
    OBS_LEN,
    check_mirror_signs,
    spec_for_task,
    validate_width,
)

# duck-control/src/obs.rs
ACTION_LEN = 14


def check_onnx(path: Path) -> list[str]:
    import onnxruntime as ort

    problems: list[str] = []
    try:
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return [f"{path.name}: not a loadable ONNX graph: {exc}"]

    def width(items) -> int:
        for item in items:
            if len(item.shape) == 2:
                return int(item.shape[1])
        return -1

    obs, act = width(session.get_inputs()), width(session.get_outputs())
    if obs != OBS_LEN:
        problems.append(f"{path.name}: obs width {obs}, expected {OBS_LEN}")
    if act != ACTION_LEN:
        problems.append(f"{path.name}: action width {act}, expected {ACTION_LEN}")

    # A policy that carries its own spec documents itself; report what it says.
    # custom_metadata_map is already a plain {key: str} dict, not protobufs.
    meta = dict(session.get_modelmeta().custom_metadata_map or {})
    task = meta.get("command_spec_task")
    if task:
        print(f"  ONNX declares task: {task}")
    else:
        print("  ONNX carries no command_spec_task (run scripts/stamp_contract.py)")
    return problems


def check_task(task_id: str) -> list[str]:
    import torch

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    problems: list[str] = []

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(env_cfg, device="cuda:0")

    obs, _ = env.reset()
    width = obs["actor"].shape[-1]
    print(f"  actor obs: {width}D")
    problems += validate_width(width)

    spec = spec_for_task(task_id)
    print(f"  command spec: {'declared' if spec else 'MISSING'}")
    if spec is None:
        problems.append(
            f"{task_id}: no CommandSpec registered. The command block's meaning "
            f"is undocumented, so nothing can check that whatever fills it "
            f"matches what this policy was trained on. Register one with "
            f"command_contract.register() from the env cfg module."
        )
        return problems

    print()
    print(spec.describe())

    # Symmetry, if the task uses it. A wrong table here is silent during
    # training, so it is worth checking explicitly rather than assuming.
    runner_cfg = None
    try:
        from mjlab.tasks.registry import load_rl_cfg

        algo = getattr(load_rl_cfg(task_id), "algorithm", None)
        sym = getattr(algo, "symmetry_cfg", None)
        if sym:
            runner_cfg = sym
    except Exception:
        pass

    if runner_cfg:
        import importlib

        dotted = runner_cfg.get("data_augmentation_func")
        if dotted:
            mod_name, fn_name = dotted.rsplit(".", 1)
            fn = getattr(importlib.import_module(mod_name), fn_name)
            # Augment a zero obs: the returned mirror half carries the sign
            # vector the function applies.
            from tensordict import TensorDict

            dummy = torch.zeros(2, width, device=env.device)
            out, _ = fn(
                env,
                TensorDict(
                    {"actor": dummy, "critic": torch.zeros(2, 76, device=env.device)},
                    batch_size=[2],
                    device=env.device,
                ),
                torch.zeros(2, ACTION_LEN, device=env.device),
            )
            # Recover the applied signs by mirroring a known non-zero vector.
            probe = torch.arange(1, width + 1, dtype=torch.float32,
                                 device=env.device).unsqueeze(0)
            out2, _ = fn(
                env,
                TensorDict(
                    {"actor": probe, "critic": torch.zeros(1, 76, device=env.device)},
                    batch_size=[1],
                    device=env.device,
                ),
                None,
            )
            mirrored = out2["actor"][1]
            perm = torch.abs(mirrored).long() - 1
            signs = torch.where(mirrored >= 0, 1.0, -1.0)
            # Undo the permutation to get per-slot signs in original order.
            restored = torch.zeros_like(signs)
            restored[perm] = signs
            print()
            print("  symmetry table vs declared mirror signs:")
            problems += check_mirror_signs(spec, restored)
            if not problems:
                print("    agrees")

    env.close()
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_id", nargs="?",
                        help="registered task to build and check")
    parser.add_argument("--onnx", type=Path, help="exported policy to check")
    parser.add_argument("--json", action="store_true",
                        help="emit problems as JSON (for CI)")
    args = parser.parse_args()

    if not args.task_id and not args.onnx:
        parser.error("give a task id or --onnx")

    problems: list[str] = []

    if args.task_id:
        print(f"=== {args.task_id} ===")
        problems += [f"{args.task_id}: {p}" for p in check_task(args.task_id)]

    if args.onnx:
        print(f"\n=== {args.onnx} ===")
        problems += [f"{args.onnx.name}: {p}"
                     for p in check_onnx(args.onnx)]

    print()
    if problems:
        if args.json:
            print(json.dumps({"ok": False, "problems": problems}, indent=2))
        else:
            for p in problems:
                print(f"FAIL: {p}")
            print(f"\n{len(problems)} problem(s)")
        sys.exit(1)

    if args.json:
        print(json.dumps({"ok": True, "problems": []}))
    else:
        print("all contract checks passed")


if __name__ == "__main__":
    main()
