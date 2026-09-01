#!/usr/bin/env python3
"""Install a trained ONNX policy where the local Microduck runtime can load it.

Trained policies are useless until `robotd` can find them. This is the missing
step between `scripts/export.py` (which writes an ONNX next to your terminal)
and the daemon, which resolves `[policy]` paths in `deploy/robotd.toml`.

It does three things, in this order:

1. **Validates** the graph is `obs[1,61] -> actions[1,14]` — the same check
   `robotd` performs at load. Catching it here costs a second; catching it on a
   robot costs a fall.
2. **Installs** the file into a local drop directory with a manifest recording
   where it came from (task, checkpoint, timestamp), because a policy with no
   provenance is indistinguishable from any other.
3. **Points the config at it** — either by printing the `[policy]` lines to
   paste, or by patching a `robotd.toml` in place with `--write-config`.

Usage:

    uv run scripts/deploy_local.py walk /path/to/output.onnx
    uv run scripts/deploy_local.py walk output.onnx \\
        --task-id Mjlab-Velocity-Flat-MicroDuck --checkpoint logs/.../model_2999.pt
    uv run scripts/deploy_local.py walk output.onnx --write-config ../Microduck/deploy/robotd.toml

Why the install directory is *outside* both repos: the runtime's `xtask` test
`every_policy_in_the_repo_is_packaged` requires every `.onnx` under
`Microduck/policies/` to appear in all three packaging manifests (dev.yml,
_build-release.yml, dev-push.sh). Dropping a freshly trained file in there
turns a local experiment into a failing CI test. The drop directory is a
sibling of both repos, and `robotd.toml` takes absolute paths by design — see
`Microduck/policies/README.md`, "Trying your own".
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mjlab_microduck.command_contract import spec_for_task  # noqa: E402

# An uncommented `key = value` line, used to find where a section's keys end.
KEY_LINE = re.compile(r"\s*[A-Za-z_][A-Za-z0-9_]*\s*=")

# Mirrors duck-control's OBS_LEN / ACTION_LEN (Microduck/duck-control/src/obs.rs).
# robotd rejects anything else at load with "observation width is N, expected 61".
OBS_LEN = 61
ACTION_LEN = 14

# The `[policy]` slot names in robotd-params, i.e. what a role may be called.
ROLES = (
    "walk",
    "stand",
    "sitstand",
    "ground_pick",
    "kick_left",
    "kick_right",
    "roulade",
)

# What each slot drives, so `--list-roles` says something useful.
ROLE_HELP = {
    "walk": "locomotion / velstand gait",
    "stand": "standing + body-pose control (also the stand-up policy)",
    "sitstand": "commanded sit <-> stand (posture flag in the twist vx slot)",
    "ground_pick": "ground pick; in roller mode this slot holds the crouch",
    "kick_left": "left-leg ball kick",
    "kick_right": "right-leg ball kick",
    "roulade": "forward roll",
}


def check_policy(path: Path) -> tuple[int, int]:
    """Return (obs_width, action_width), refusing graphs robotd would refuse."""
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime raises its own hierarchy, all unreadable
        raise SystemExit(f"error: {path.name} is not a loadable ONNX graph: {exc}") from exc

    def width(items) -> int:
        # Shapes are [1, N]; anything without a second dim is not this family.
        for item in items:
            shape = item.shape
            if len(shape) == 2:
                return int(shape[1])
        return -1

    obs = width(session.get_inputs())
    act = width(session.get_outputs())
    if (obs, act) != (OBS_LEN, ACTION_LEN):
        raise SystemExit(
            f"error: {path.name}: observation width is {obs}, expected {OBS_LEN} "
            f"(action width {act}, expected {ACTION_LEN}).\n"
            "       Only ONNX files produced by scripts/export.py belong here: the\n"
            "       observation normalizer is baked into that graph, and a hand-converted\n"
            "       checkpoint would be fed unnormalized observations on the robot."
        )
    return obs, act


def read_stamped_task(path: Path) -> str | None:
    """Task id an ONNX declares in its own metadata, if it was stamped."""
    import onnx

    try:
        meta = onnx.load(str(path), load_external_data=False).metadata_props
    except Exception:
        return None
    for prop in meta:
        if prop.key == "command_spec_task":
            return prop.value
    return None


def installed_stamped(path: Path) -> bool:
    return read_stamped_task(path) is not None


def install(onnx: Path, install_dir: Path, role: str, provenance: dict) -> Path:
    install_dir.mkdir(parents=True, exist_ok=True)
    target = install_dir / f"{role}.onnx"
    if target.resolve() == onnx.resolve():
        raise SystemExit(f"error: {onnx} is already the installed {role} policy")
    shutil.copy2(onnx, target)
    (install_dir / f"{role}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return target


def build_snippet(policies: dict[str, Path]) -> str:
    lines = ["[policy]"]
    for role, path in policies.items():
        lines.append(f'{role} = "{path}"')
    return "\n".join(lines) + "\n"


def patch_config(config: Path, policies: dict[str, Path]) -> None:
    """Set the given keys inside the existing `[policy]` table.

    Inserting a second `[policy]` header would be invalid TOML, and the file's
    own header comment asks that values stay commented unless a robot genuinely
    needs them — so an already-present (or commented) key is replaced in place
    and a missing one is appended to the section.
    """
    lines = config.read_text().splitlines(keepends=True)
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "[policy]"), None
    )
    if start is None:
        raise SystemExit(f"error: {config} has no [policy] section")

    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i].lstrip().startswith("[") and lines[i].rstrip().endswith("]")
        ),
        len(lines),
    )

    for role, path in policies.items():
        key = f'{role} = "{path}"'
        existing = next(
            (
                i
                for i in range(start + 1, end)
                if lines[i].lstrip().removeprefix("#").lstrip().startswith(f"{role} ")
            ),
            None,
        )
        if existing is not None:
            lines[existing] = key + "\n"
            continue
        # Append after the last real key in the section, so the line lands next
        # to its neighbours rather than below the section's trailing comments.
        last_key = max(
            (i for i in range(start + 1, end) if KEY_LINE.match(lines[i])),
            default=start,
        )
        lines.insert(last_key + 1, key + "\n")
        end += 1

    config.write_text("".join(lines))


def main() -> None:
    default_install = Path(__file__).resolve().parent.parent.parent / "deployed_policies"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Optional only so `--list-roles` works without them; validated below.
    parser.add_argument("role", choices=ROLES, nargs="?", help="the [policy] slot to fill")
    parser.add_argument("onnx", type=Path, nargs="?", help="ONNX file from scripts/export.py")
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=default_install,
        help=f"where to install (default: {default_install})",
    )
    parser.add_argument("--task-id", help="training task id, recorded in the manifest")
    parser.add_argument("--checkpoint", help="checkpoint the policy came from")
    parser.add_argument(
        "--write-config",
        type=Path,
        metavar="ROBOTD_TOML",
        help="patch this robotd.toml in place instead of only printing the lines",
    )
    parser.add_argument(
        "--list-roles",
        action="store_true",
        help="print the roles and exit",
    )
    args = parser.parse_args()

    if args.list_roles:
        for role in ROLES:
            print(f"{role:12s} {ROLE_HELP[role]}")
        return

    if args.role is None or args.onnx is None:
        parser.error("role and onnx are required (or use --list-roles)")

    onnx = args.onnx.expanduser()
    if not onnx.is_file():
        raise SystemExit(f"error: no such file: {onnx}")

    obs, act = check_policy(onnx)
    print(f"checked {onnx.name}: obs[1,{obs}] -> actions[1,{act}]")

    # Record what the command block means, so the manifest says what has to be
    # written into it. Without this, a deployed .onnx is a 61->14 graph and
    # nothing else: four tasks in this repo redefine those 13 slots, all export
    # identically, and feeding one the wrong block makes the robot do something
    # confidently wrong with no error anywhere.
    command_spec = None
    stamped_task = read_stamped_task(onnx)
    stamped = stamped_task is not None
    spec = spec_for_task(args.task_id) if args.task_id else None
    if spec is None:
        # Fall back to whatever the ONNX itself declares (scripts/stamp_contract.py).
        if stamped_task:
            spec = spec_for_task(stamped_task)
    elif stamped and stamped_task != spec.task_id:
        # The file claims one task, --task-id says another. Both 61 wide, so
        # nothing downstream would notice; say it now.
        print(f"warning: ONNX is stamped {stamped_task} but --task-id is "
              f"{spec.task_id}. Using {spec.task_id}; re-stamp if that is wrong.")

    if spec is not None:
        command_spec = {
            "task_id": spec.task_id,
            "slots": [
                {
                    "obs_index": 48 + i,
                    "name": s.name,
                    "unit": s.unit,
                    "range": list(s.range),
                    "description": s.description,
                }
                for i, s in enumerate(spec.slots)
            ],
            "notes": spec.notes,
        }
        print(f"command block: {spec.task_id}")
        for s, meta in zip(spec.slots, command_spec["slots"]):
            print(f"  obs[{meta['obs_index']}] {s.name:16s} "
                  f"{s.unit:6s} {s.description}")
        if not stamped:
            print("  (not stamped into the ONNX — "
                  "run scripts/stamp_contract.py to make the file self-describing)")
    elif args.task_id:
        print(f"note: no CommandSpec for {args.task_id!r}; the manifest will not "
              f"say how to fill the command block")

    provenance = {
        "role": args.role,
        "installed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_onnx": str(onnx.resolve()),
        "task_id": args.task_id,
        "checkpoint": args.checkpoint,
        "command_block": command_spec,
    }
    target = install(onnx, args.install_dir, args.role, provenance)

    print(f"installed {target}")
    print(f"manifest   {target.with_suffix('.json')}")

    policies = {args.role: target}
    if args.write_config:
        config = args.write_config.expanduser()
        if not config.is_file():
            raise SystemExit(f"error: no such config: {config}")
        patch_config(config, policies)
        print(f"patched    {config} -> [policy] {args.role}")
        print("robotd reads this once at startup: systemctl restart robotd")
    else:
        print("\nAdd to [policy] in robotd.toml (absolute paths are supported):\n")
        print(build_snippet(policies), end="")


if __name__ == "__main__":
    main()
