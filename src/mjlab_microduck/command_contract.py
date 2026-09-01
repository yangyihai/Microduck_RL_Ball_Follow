"""Machine-readable meaning of the 61D observation's command block.

Why this exists
---------------
The unified 61D obs is

    48D proprioception  +  13D command block  [twist(3), head_pose(4), body_pose(6)]

and the runtime (`duck-control/src/obs.rs`) copies `twist/head/body` straight
into that block without looking at what they mean. That is deliberate — it is
what makes one policy hot-swappable for another — but it also means **nothing
checks that the numbers being piped in are the numbers the policy was trained
on.**

The command block is already redefined per task in this repo:

    velocity   twist = [lin_vel_x, lin_vel_y, ang_vel_z]
    sitstand   twist = [sit_stand_flag, 0, 0]   (a posture flag, not a velocity)
    ball_kick  twist = [0, 0, 0]                (kick flag lives in the body slots)
    ball_follow twist = [target_x_body, target_y_body, hold_distance]

All four are 61 wide, all four export fine, all four load on the robot — and
feeding one the others' command block produces a robot that does something
confidently wrong. The width check passes; only the *meaning* is broken.

This module makes the meaning explicit and checkable:

- :class:`CommandSpec`   — what each of the 13 slots means, with units and how
  each behaves under a left-right reflection.
- :func:`spec_for_task`  — look up a registered task's spec.
- :func:`validate`       — catch the mistakes that are silent today (see
  :func:`check_mirror_signs` for the one that bit this repo already).
- :func:`to_metadata`    — serialise into ONNX metadata so a deployed .onnx
  carries its own command documentation, and whoever writes the command block
  can read back what to fill in.

Adding a task
-------------
Register a spec next to your env cfg. If your task redefines the twist slots,
you MUST declare it — that is the whole point. Copy `BALL_FOLLOW_SPEC` as a
template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Block geometry, fixed by the runtime's Observation::build.
COMMAND_TWIST_LEN = 3
COMMAND_HEAD_LEN = 4
COMMAND_BODY_LEN = 6
COMMAND_LEN = COMMAND_TWIST_LEN + COMMAND_HEAD_LEN + COMMAND_BODY_LEN  # 13

# Full actor observation width (duck-control/src/obs.rs OBS_LEN).
OBS_LEN = 61
PROPRIO_LEN = OBS_LEN - COMMAND_LEN  # 48

# Slot offsets inside the actor observation.
TWIST_SLICE = slice(48, 51)
HEAD_SLICE = slice(51, 55)
BODY_SLICE = slice(55, 61)


@dataclass(frozen=True)
class Slot:
    """One number in the command block.

    mirror_sign is how the value transforms under a left-right reflection of the
    robot: +1 unchanged, -1 negated. It is not decoration — see
    :func:`check_mirror_signs`.
    """

    name: str
    description: str
    unit: str
    range: tuple[float, float]
    mirror_sign: int = 1

    def __post_init__(self) -> None:
        if self.mirror_sign not in (1, -1):
            raise ValueError(
                f"slot {self.name!r}: mirror_sign must be +1 or -1, "
                f"got {self.mirror_sign}"
            )
        lo, hi = self.range
        if lo > hi:
            raise ValueError(
                f"slot {self.name!r}: range is inverted ({lo}, {hi})")


@dataclass(frozen=True)
class CommandSpec:
    """The meaning of one task's 13-slot command block."""

    task_id: str
    twist: tuple[Slot, ...]
    head: tuple[Slot, ...]
    body: tuple[Slot, ...]
    # Free-form note recorded in the ONNX metadata. Say where the command comes
    # from on the real robot, since that is the part only the task author knows.
    notes: str = ""

    def __post_init__(self) -> None:
        if len(self.twist) != COMMAND_TWIST_LEN:
            raise ValueError(
                f"{self.task_id}: twist has {len(self.twist)} slots, "
                f"expected {COMMAND_TWIST_LEN}"
            )
        if len(self.head) != COMMAND_HEAD_LEN:
            raise ValueError(
                f"{self.task_id}: head has {len(self.head)} slots, "
                f"expected {COMMAND_HEAD_LEN}"
            )
        if len(self.body) != COMMAND_BODY_LEN:
            raise ValueError(
                f"{self.task_id}: body has {len(self.body)} slots, "
                f"expected {COMMAND_BODY_LEN}"
            )

    # -- access ---------------------------------------------------------------
    @property
    def slots(self) -> tuple[Slot, ...]:
        """All 13 slots in observation order."""
        return (*self.twist, *self.head, *self.body)

    @property
    def length(self) -> int:
        return len(self.slots)

    @property
    def mirror_signs(self) -> tuple[int, ...]:
        """Expected sign flip per slot, in observation order."""
        return tuple(s.mirror_sign for s in self.slots)

    # -- serialisation --------------------------------------------------------
    def to_metadata(self) -> dict[str, str]:
        """Flatten into ONNX metadata (string values only).

        Names are written as `cmd_twist_0_name`, `cmd_twist_0_unit`, ... next to
        the runtime's own keys, so a deployed policy documents itself.
        """
        out: dict[str, str] = {
            "command_spec_task": self.task_id,
            "command_spec_mirror_signs": ",".join(
                str(s) for s in self.mirror_signs),
            "command_spec_notes": self.notes,
        }
        for group, slots in (("twist", self.twist),
                             ("head", self.head),
                             ("body", self.body)):
            for i, slot in enumerate(slots):
                base = f"cmd_{group}_{i}"
                out[f"{base}_name"] = slot.name
                out[f"{base}_unit"] = slot.unit
                out[f"{base}_range"] = f"{slot.range[0]},{slot.range[1]}"
                out[f"{base}_desc"] = slot.description
        return out

    def describe(self) -> str:
        """Human-readable table: what to write into the command block."""
        lines = [f"Command block for {self.task_id}", ""]
        offset = PROPRIO_LEN
        for group, slots in (("twist", self.twist),
                             ("head", self.head),
                             ("body", self.body)):
            lines.append(f"  {group}:")
            for i, slot in enumerate(slots):
                lo, hi = slot.range
                lines.append(
                    f"    obs[{offset + i:2d}] {slot.name:16s} "
                    f"{slot.unit:8s} [{lo:g}, {hi:g}]  {slot.description}"
                )
            offset += len(slots)
        if self.notes:
            lines += ["", f"  note: {self.notes}"]
        return "\n".join(lines)


# ── Registered specs ──────────────────────────────────────────────────────────
# Head and body slots mean the same thing in every task so far, so they are
# defined once and shared. Only the twist slot varies.

def _head_slots() -> tuple[Slot, ...]:
    return (
        Slot("neck_pitch", "neck pitch target", "rad", (-0.05, 0.05), +1),
        Slot("head_pitch", "head pitch target", "rad", (-0.05, 0.05), +1),
        Slot("head_yaw", "head yaw target", "rad", (-0.07, 0.07), -1),
        Slot("head_roll", "head roll target", "rad", (-0.015, 0.015), -1),
    )


def _body_slots() -> tuple[Slot, ...]:
    # Order and mirror signs are fixed by two things that must agree: the
    # runtime's Observation::build fills x, y, z, roll, pitch, yaw (with x, y
    # and yaw hardcoded to 0.0 — unbound in training), and symmetry.py negates
    # y, roll, yaw for the reflection. Do not reorder these without changing
    # both; check_mirror_signs will catch it if you do.
    return (
        Slot("body_x", "unbound in training, always 0", "-", (0.0, 0.0), +1),
        Slot("body_y", "unbound in training, always 0", "-", (0.0, 0.0), -1),
        Slot("body_z", "standing height offset", "m", (-0.05, 0.05), +1),
        Slot("body_roll", "body roll offset", "rad", (-0.05, 0.05), -1),
        Slot("body_pitch", "body pitch offset", "rad", (-0.05, 0.05), +1),
        Slot("body_yaw", "unbound in training, always 0", "-", (0.0, 0.0), -1),
    )


VELOCITY_SPEC = CommandSpec(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    twist=(
        Slot("lin_vel_x", "forward velocity command", "m/s", (-0.5, 0.5), +1),
        Slot("lin_vel_y", "lateral velocity command", "m/s", (-0.5, 0.5), -1),
        Slot("ang_vel_z", "yaw rate command", "rad/s", (-1.5, 1.5), -1),
    ),
    head=_head_slots(),
    body=_body_slots(),
    notes="Baseline: a velocity command. >0 magnitude selects walking.",
)

SITSTAND_SPEC = CommandSpec(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    twist=(
        Slot("posture_flag", "1.0 = stand, -1.0 = sit", "flag", (-1.0, 1.0), +1),
        Slot("unused", "always zero", "-", (0.0, 0.0), +1),
        Slot("unused", "always zero", "-", (0.0, 0.0), +1),
    ),
    head=_head_slots(),
    body=_body_slots(),
    notes="twist[0] is a posture flag, NOT a velocity. A velocity policy fed "
          "this block would read it as a forward speed.",
)

BALL_FOLLOW_SPEC = CommandSpec(
    task_id="Mjlab-BallFollow-Flat-MicroDuck",
    twist=(
        Slot("target_x_body", "target offset, robot forward axis",
             "m", (-2.0, 2.0), +1),
        Slot("target_y_body", "target offset, robot left axis",
             "m", (-2.0, 2.0), -1),
        Slot("hold_distance", "stand-off distance to hold",
             "m", (0.25, 0.45), +1),
    ),
    head=_head_slots(),
    body=_body_slots(),
    notes="Target comes from OUTSIDE the policy — the actor has no target "
          "sensor, so the command block must be filled by whatever is tracking "
          "it (camera, motion capture, UI). hold_distance has no left/right "
          "sign; do not mirror it.",
)


# Sugars the same spec for the other registered task ids of a family.
def _family(base: CommandSpec, *task_ids: str) -> dict[str, CommandSpec]:
    return {
        tid: CommandSpec(
            task_id=tid,
            twist=base.twist,
            head=base.head,
            body=base.body,
            notes=base.notes,
        )
        for tid in task_ids
    }


_REGISTRY: dict[str, CommandSpec] = {}
for _spec in (VELOCITY_SPEC, SITSTAND_SPEC, BALL_FOLLOW_SPEC):
    _REGISTRY[_spec.task_id] = _spec
_REGISTRY.update(_family(
    VELOCITY_SPEC,
    "Mjlab-Velocity-Rough-MicroDuck",
    "Mjlab-VelStand-Flat-MicroDuck",
    "Mjlab-VelStand-Rough-MicroDuck",
))
_REGISTRY.update(_family(
    SITSTAND_SPEC, "Mjlab-SitStand-Rough-MicroDuck"))


def register(spec: CommandSpec) -> None:
    """Register (or replace) a task's command spec. Call from your env module."""
    _REGISTRY[spec.task_id] = spec


def spec_for_task(task_id: str) -> CommandSpec | None:
    return _REGISTRY.get(task_id)


def known_tasks() -> list[str]:
    return sorted(_REGISTRY)


# ── Validation ────────────────────────────────────────────────────────────────

def check_mirror_signs(spec: CommandSpec,
                       symmetry_signs: Any,
                       slice_start: int = PROPRIO_LEN) -> list[str]:
    """Does a symmetry table agree with this spec's declared mirror signs?

    This is the check that would have caught a real bug in this repo. The
    velocity symmetry table mirrors the twist slot as [vx, -vy, -vyaw], and it
    was reused for BallFollow — which negates the stand-off distance, i.e. it
    trains the policy that the mirror image of "keep 0.35 m" is "keep -0.35 m",
    a command it can never be given. Widths matched, nothing errored, the policy
    just learned something wrong.

    Call this from your task's self-check with the sign vector your symmetry
    function applies to the actor observation.
    """
    import torch

    if isinstance(symmetry_signs, torch.Tensor):
        signs = symmetry_signs.detach().cpu().tolist()
    else:
        signs = list(symmetry_signs)

    problems: list[str] = []
    if len(signs) < slice_start + spec.length:
        return [f"symmetry sign vector has {len(signs)} entries, need at "
                f"least {slice_start + spec.length}"]

    actual = signs[slice_start:slice_start + spec.length]
    for slot, want, got in zip(spec.slots, spec.mirror_signs, actual):
        # Compare signs, tolerate magnitudes other than exactly 1.0.
        got_sign = 1 if float(got) >= 0 else -1
        if got_sign != want:
            problems.append(
                f"slot {slot.name!r} (obs[{slice_start + spec.slots.index(slot)}]): "
                f"spec declares mirror_sign {want:+d} but the symmetry table "
                f"applies {got_sign:+d}"
            )
    return problems


def validate_width(actor_obs_width: int) -> list[str]:
    """The runtime rejects anything but 61; catch it before deployment."""
    if actor_obs_width != OBS_LEN:
        return [f"actor obs is {actor_obs_width}D, the runtime requires "
                f"{OBS_LEN}D"]
    return []
