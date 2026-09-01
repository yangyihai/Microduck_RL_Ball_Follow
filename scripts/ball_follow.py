#!/usr/bin/env python3
"""Drag the yellow ball; the duck walks after it and stops at a set distance.

Every so often it stops and does a trick — sits down, rolls over, touches the
ground, kicks — then gets up and carries on following. Both the trick and the
gap between tricks are random.

    uv run scripts/ball_follow.py                 # follow, with random tricks
    uv run scripts/ball_follow.py --no-tricks     # just follow
    uv run scripts/ball_follow.py --overlay       # show the guide line and ring

Mouse

    left-drag on the ball   move it (it stays at its height, on a ground plane)
    left-drag elsewhere     orbit the camera
    right-drag / wheel      pan / zoom
    Q or Esc                quit,  R reset

Why this has its own window instead of `mujoco.viewer.launch_passive`: the
passive viewer exposes no mouse API at all (`Handle` has cam/opt/perturb/
user_scn/sync and nothing else), so there is no way to be told where the mouse
is. A ball you cannot grab is not this demo.

The controller — read this before tuning it
-------------------------------------------
The walking policy does NOT respond proportionally to velocity commands in this
rehearsal. Measured on `alpha_walking.onnx` at 50 Hz (see CLAUDE.md):

    cmd vx    0.15   0.20   0.22   0.25   0.30
    achieved  0.000  0.000  0.090  0.103  0.122   m/s

    cmd vyaw  0.3    0.6    1.0    1.5
    achieved  0.001  0.000  0.447  0.746            rad/s

    cmd vx   -0.20  -0.25  -0.30
    achieved  0.000  0.000  0.000                   m/s   ← cannot reverse

So there is a **dead zone**: commands below ~0.22 m/s and ~0.8 rad/s produce
literally nothing, and reverse does not exist. A proportional controller that
commands `0.3 * error` would spend its whole life inside the dead zone and the
robot would stand there. Hence:

- Commands are issued at a magnitude that clears the dead zone (bang-bang with a
  floor), not proportional to the error. The error only decides *whether* to
  move, and hysteresis decides when to stop, so the robot does not chatter
  across the boundary.
- "Too close" means *stop*, not *back away* — the gait has no reverse. The robot
  will not close the gap further on its own; drag the ball away and it follows.

These are measured properties of this model and this rehearsal, not of the task.
If the policy or the actuator model changes, re-measure before trusting them.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import glfw
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_policy as ip  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCENE_XML = REPO / "src/mjlab_microduck/robot/microduck/scene_track.xml"
POLICY_DIR = REPO.parent / "Microduck" / "policies"
DEFAULT_POLICY = POLICY_DIR / "alpha_walking.onnx"

DECIMATION = 4          # physics steps per control tick; 5 ms × 4 = 50 Hz
CONTROL_HZ = 50.0

# --- Command magnitudes, above the measured dead zones -----------------------
CMD_VX_GO = 0.25        # min forward command that actually produces motion
CMD_VX_MAX = 0.30
CMD_VYAW_GO = 1.0       # min yaw command that actually produces rotation
CMD_VYAW_MAX = 1.5

# --- Hysteresis (m / rad) ----------------------------------------------------
APPROACH_ON = 0.08      # start walking when this far outside the target
APPROACH_OFF = 0.03     # stop once this close to it
TURN_ON = 0.20          # start turning when this far off heading
TURN_OFF = 0.07         # stop turning once this well aligned
FACE_LIMIT = 0.60       # do not walk forward while this far off heading

BALL_Z = 0.10           # drag plane height, matches the mocap body
DRAG_LIMIT = 3.0        # keep the target ball within this radius of the origin
GRAB_PX = 26            # extra screen-space slack for grabbing the ball

# Where the real (kickable) ball waits. Far outside DRAG_LIMIT so it can never
# lie in the robot's walking path.
BALL_PARK = (20.0, 20.0, 0.035)


# --- Following ---------------------------------------------------------------
class BallFollower:
    """Turn the ball's position into velocity commands for the walking policy."""

    def __init__(self, target_distance: float = 0.35):
        self.target_distance = target_distance
        self.approaching = False
        self.turning = False
        # Reported for the on-screen readout and for trick gating.
        self.distance = float("nan")
        self.yaw_error = float("nan")
        self.too_close = False

    @staticmethod
    def _yaw(quat: np.ndarray) -> float:
        w, x, y, z = quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def command(self, robot_xy: np.ndarray, quat: np.ndarray,
                ball_xy: np.ndarray) -> tuple[float, float, float]:
        """Return (vx, vy, vyaw) in the robot's body frame."""
        dx, dy = float(ball_xy[0] - robot_xy[0]), float(ball_xy[1] - robot_xy[1])
        dist = math.hypot(dx, dy)

        # World -> body. Measured: +vx is world +x at yaw 0, +vyaw increases yaw.
        yaw = self._yaw(quat)
        c, s = math.cos(yaw), math.sin(yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        yaw_err = math.atan2(local_y, local_x)

        # Turn: hysteresis, then a command big enough to clear the dead zone.
        if self.turning:
            if abs(yaw_err) < TURN_OFF:
                self.turning = False
        elif abs(yaw_err) > TURN_ON:
            self.turning = True

        vyaw = 0.0
        if self.turning:
            # Scale with the error, but never below CMD_VYAW_GO. The floor is
            # the whole point: the error shrinks as the turn proceeds, so an
            # unclamped `GO + k*(|err| - TURN_ON)` decays below the ~0.8 rad/s
            # dead zone while `turning` is still set (it only clears at
            # TURN_OFF, which is below TURN_ON). The robot then issues a command
            # too small to turn, so the error never shrinks and it stays
            # "turning" forever — observed stuck at 0.08 rad for 40 s.
            mag = min(CMD_VYAW_MAX,
                      max(CMD_VYAW_GO, CMD_VYAW_GO + 1.2 * (abs(yaw_err) - TURN_ON)))
            vyaw = math.copysign(mag, yaw_err)

        # Approach: hysteresis around the target distance. err < 0 means the ball
        # is closer than the target; there is no reverse, so that means stop.
        err = dist - self.target_distance
        if self.approaching:
            if err < APPROACH_OFF:
                self.approaching = False
        elif err > APPROACH_ON:
            self.approaching = True

        # Never walk while turning. The dead zone means any turn that works at
        # all is a hard one (~0.45 rad/s), and walking through one tips the
        # robot over — it fell every time right after a trick, when it was still
        # settling onto its feet. Turn to face first, then walk.
        vx = 0.0
        if self.approaching and not self.turning and abs(yaw_err) < FACE_LIMIT:
            # Same floor as vyaw, same reason: without it the command decays
            # below the ~0.22 m/s dead zone as the gap closes, while
            # `approaching` is still set, and the robot stalls short of target.
            vx = min(CMD_VX_MAX,
                     max(CMD_VX_GO, CMD_VX_GO + 0.5 * (err - APPROACH_ON)))

        self.distance, self.yaw_error = dist, yaw_err
        self.too_close = err < -APPROACH_ON
        return vx, 0.0, vyaw

    def settled(self) -> bool:
        """True when the robot is parked and not walking anywhere.

        Tricks are gated on this: a trick fired mid-stride would be a fall, not
        a flourish.

        Deliberately ignores `turning`. While parked, the hysteresis keeps
        nudging heading between TURN_OFF and TURN_ON, so `turning` flickers
        constantly — requiring it to be clear meant the trick timer was reset
        almost every tick and tricks stopped firing entirely (observed: two
        tricks in the first 22 s, then none for the rest of the run). Turning in
        place is safe to interrupt; only walking is not.
        """
        return not self.approaching and abs(self.yaw_error) < TRICK_YAW_TOL

    def status(self) -> str:
        if self.turning:
            return "turning to face the ball"
        if self.too_close:
            return "too close - holding (no reverse gait)"
        if self.approaching:
            return "following"
        return "at target distance"


def set_command(policy, vx: float, vy: float, vyaw: float) -> None:
    """Write the velocity command without going through set_vel_cmd.

    `PolicyInference.set_vel_cmd` prints, which is fine for a keypress and
    useless at 50 Hz — the yaw term changes every tick while turning, so it
    would print 50 lines a second. Equivalent here because only the walking
    policy is loaded for following, so the walk/stand switch inside
    set_vel_cmd would be a no-op anyway.
    """
    policy.vel_cmd[:] = (vx, vy, vyaw)
    policy._update_command()


# --- Tricks ------------------------------------------------------------------
TRICKS = ("sit", "ground_pick", "roulade", "kick_left", "kick_right")
TRICK_LABEL = {
    "sit": "sitting down",
    "ground_pick": "touching the ground",
    "roulade": "rolling",
    "kick_left": "kicking (left)",
    "kick_right": "kicking (right)",
}

TRICK_GAP_MIN = 6.0     # seconds of settled following before a trick
TRICK_GAP_MAX = 14.0
TRICK_YAW_TOL = 0.30    # must be facing roughly at the ball first
SIT_HOLD = 2.0          # how long to stay seated
STAND_UP = 3.5          # time for the sitstand policy to rise before walking
# Durations measured from leg joint velocity through each trick: the kick's
# motion is over by ~0.3 s and the roll's by ~1.6 s. These leave margin without
# leaving the robot frozen after the trick is done. (infer_policy's own default
# of 3.0 s for a kick means standing still for 2.7 s after the kick landed.)
ROULADE_DURATION = 2.0
KICK_DURATION = 0.8
# After a trick the robot is still settling onto its feet. Walking (or turning,
# which the dead zone makes unavoidably abrupt) straight away tips it over —
# measured: it fell on the first step after standing up. So stand still first.
RECOVER_TIME = 2.0
MAX_DT = 0.10           # a stall must not fast-forward a trick


class TrickScheduler:
    """Fire a random trick now and then, then hand control back to following.

    Every trick *ends* differently, and that asymmetry is the whole reason this
    is a state machine rather than a timer:

      roulade / kick_*  auto-return: `update_behavior` counts the duration down
                        and `_end_behavior` puts the walk policy back.
      ground_pick       auto-return via phases: `update_ground_pick_phase`
                        hands over at 70% of the period (2.8 s of 4 s).
      sit               a posture FLAG, not a timer. Nothing flips it back, so
                        this scheduler must, and the policy then needs STAND_UP
                        seconds to physically get up before walking is safe.

    Tricks are only fired while the robot is settled (see BallFollower.settled):
    the command is ignored during a trick anyway, so firing mid-stride would
    just mean the robot stops where it stands and then has to re-acquire.
    """

    def __init__(self, rng: random.Random, enabled: bool = True,
                 gap: tuple[float, float] = (TRICK_GAP_MIN, TRICK_GAP_MAX),
                 only: tuple[str, ...] | None = None):
        self.rng = rng
        self.enabled = enabled
        self.gap = gap
        self.only = tuple(only) if only else None
        self.wait = rng.uniform(*gap)
        self.timer = 0.0
        self.active = None      # name of the running trick, or None
        self.stage = None       # "doing" | "rising" (sit) | "recover"
        self.done_count = 0

    # -- plumbing ------------------------------------------------------------
    def available(self, policy, name: str) -> bool:
        """Is this trick both requested and actually loadable?

        A missing ONNX is skipped rather than fatal: the point of the demo is
        following, and every trick is optional.
        """
        if self.only is not None and name not in self.only:
            return False
        return {
            "sit": policy.sit_session,
            "ground_pick": policy.ground_pick_session,
            # kick_left / kick_right / roulade live in behavior_sessions.
            "roulade": policy.behavior_sessions.get("roulade"),
            "kick_left": policy.behavior_sessions.get("kick_left"),
            "kick_right": policy.behavior_sessions.get("kick_right"),
        }.get(name) is not None

    @staticmethod
    def busy(policy) -> bool:
        """Is any trick currently owning the policy?"""
        return (policy.behavior_mode is not None
                or policy.ground_pick_mode
                or policy.sit_mode)

    @property
    def idle(self) -> bool:
        """True when the follower may steer.

        False through the trick *and* through the recovery stand afterwards:
        during recovery the robot must be left alone to steady itself, so the
        follower keeps its hands off rather than issuing a zero command that
        would fight the gait.
        """
        return self.active is None and self.stage != "recover"

    @property
    def recovering(self) -> bool:
        return self.stage == "recover"

    def label(self) -> str | None:
        if self.active is None:
            return None
        return TRICK_LABEL.get(self.active, self.active)

    # -- main ----------------------------------------------------------------
    def update(self, dt: float, policy, follower: BallFollower, data=None) -> None:
        if not self.enabled:
            return
        if self.active is not None:
            self._advance(dt, policy, data)
            return
        if self.stage == "recover":
            self.timer += dt
            if self.timer >= RECOVER_TIME:
                self.stage = None
                self.timer = 0.0
            return
        # No trick running: only charge the timer while parked at the target.
        if self.busy(policy) or not follower.settled():
            self.timer = 0.0
            return
        self.timer += dt
        if self.timer >= self.wait:
            self.fire(policy)

    def fire(self, policy, name: str | None = None) -> str | None:
        """Start a trick right now: a specific one, or a random available one.

        Returns the name fired, or None if nothing was available (or if the
        named trick has no policy loaded).
        """
        if name is not None:
            if not self.available(policy, name):
                return None
            choices = [name]
        else:
            choices = [t for t in TRICKS if self.available(policy, t)]
            if not choices:
                self.timer = 0.0
                self.wait = self.rng.uniform(*self.gap)
                return None
        name = self.rng.choice(choices)

        if name == "sit":
            policy.toggle_sit()
        elif name == "ground_pick":
            policy.trigger_ground_pick()
        else:
            policy.trigger_behavior(name)

        self.active = name
        self.stage = "doing"
        self.timer = 0.0
        return name

    def _advance(self, dt: float, policy, data=None) -> None:
        self.timer += dt

        if self.active == "sit":
            # sit is a flag, so it needs flipping back by hand — and the robot
            # needs time to actually stand up before walking can take over.
            if self.stage == "doing":
                if self.timer >= SIT_HOLD:
                    policy.toggle_sit()      # flag -> 0: the policy stands up
                    self.stage = "rising"
                    self.timer = 0.0
            elif self.stage == "rising" and self.timer >= STAND_UP:
                self._hand_back(policy, data)
            return

        # Auto-returning tricks: finished once the policy cleared its mode.
        if not self.busy(policy):
            # `data` must reach here: a kick leaves a real ball lying where the
            # robot is about to walk, and only parking it again keeps the path
            # clear. Pass nothing and the ball is never picked up.
            self._hand_back(policy, data)

    def _hand_back(self, policy, data=None) -> None:
        """Put the walk policy back in charge and re-arm a random gap."""
        if policy.walking_session is not None:
            policy.current_policy = "walking"
            policy.ort_session = policy.walking_session
        # Zero the command and let the walk policy hold a stand still: the robot
        # has just finished a trick and is not steady enough to be steered yet.
        policy.vel_cmd[:] = 0.0
        policy._update_command()
        if data is not None:
            park_ball(policy, data)
        self.active = None
        self.stage = "recover"
        self.timer = 0.0
        self.done_count += 1
        self.wait = self.rng.uniform(*self.gap)


def park_ball(policy, data) -> None:
    """Move the kickable ball back out of the way.

    `trigger_behavior` teleports it in front of the kicking foot; if it were
    left there the robot would keep tripping over it while walking.
    """
    adr = policy.ball_qpos_adr
    if adr is None:
        return
    data.qpos[adr:adr + 7] = [*BALL_PARK, 1.0, 0.0, 0.0, 0.0]
    qvel_adr = policy.ball_qvel_adr
    if qvel_adr is not None:
        data.qvel[qvel_adr:qvel_adr + 6] = 0.0


# --- Camera / mouse ----------------------------------------------------------
class View:
    """Minimal orbit camera plus screen<->world projection for ball dragging."""

    def __init__(self, model):
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 1.7
        self.cam.azimuth = 160.0
        self.cam.elevation = -22.0
        self.cam.lookat[:] = [0.0, 0.0, 0.15]
        # The vertical FOV lives on the model, not on MjvCamera — this is what
        # mjv_updateScene builds the frustum from.
        self.fovy = float(model.vis.global_.fovy)
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(model, maxgeom=max(model.ngeom + 80, 1000))
        self.width, self.height = 1280, 720
        self.grabbed = False

    def basis(self):
        gl = self.scn.camera[0]
        fwd = np.array(gl.forward, dtype=np.float64)
        up = np.array(gl.up, dtype=np.float64)
        pos = np.array(gl.pos, dtype=np.float64)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        return pos, fwd, up, right

    def project(self, point: np.ndarray) -> tuple[float, float, float]:
        """World point -> (pixel x, pixel y, view depth). Depth drives ball size."""
        pos, fwd, up, right = self.basis()
        rel = np.asarray(point, dtype=np.float64) - pos
        z = float(np.dot(rel, fwd))
        if z <= 1e-6:
            return -1e6, -1e6, z
        t = math.tan(math.radians(self.fovy) * 0.5)
        ndc_x = float(np.dot(rel, right)) / (z * t * self.width / self.height)
        ndc_y = float(np.dot(rel, up)) / (z * t)
        return ((ndc_x * 0.5 + 0.5) * self.width,
                (1.0 - (ndc_y * 0.5 + 0.5)) * self.height, z)

    def ray_to_plane(self, px: float, py: float, z: float) -> np.ndarray | None:
        """Mouse pixel -> world point on the horizontal plane at height z."""
        pos, fwd, up, right = self.basis()
        t = math.tan(math.radians(self.fovy) * 0.5)
        aspect = self.width / self.height
        ndc_x = (px / self.width) * 2.0 - 1.0
        ndc_y = 1.0 - (py / self.height) * 2.0
        direction = fwd + ndc_x * t * aspect * right + ndc_y * t * up
        if abs(direction[2]) < 1e-6:
            return None
        s = (z - pos[2]) / direction[2]
        if s <= 0:                      # the plane is behind the camera
            return None
        return pos + s * direction

    def grab_hit(self, px: float, py: float, ball: np.ndarray,
                 radius: float = 0.04) -> bool:
        """Would a click at this pixel have grabbed the ball?

        A screen-space hit test around the ball's projected disc, with GRAB_PX of
        slack: the ball is small on screen from a distance, and nobody aims that
        precisely.
        """
        bx, by, depth = self.project(ball)
        if depth <= 0:
            return False
        t = math.tan(math.radians(self.fovy) * 0.5)
        ball_px = radius / (depth * t) * (self.height * 0.5)
        return math.hypot(px - bx, py - by) <= max(ball_px, 12.0) + GRAB_PX

    def orbit(self, dx: float, dy: float) -> None:
        self.cam.azimuth -= dx * 0.3
        self.cam.elevation = float(np.clip(self.cam.elevation - dy * 0.3, -89.0, 89.0))

    def pan(self, dx: float, dy: float) -> None:
        _, fwd, up, right = self.basis()
        scale = self.cam.distance * 0.0015
        self.cam.lookat[:] += (-dx * right + dy * up) * scale

    def zoom(self, amount: float) -> None:
        self.cam.distance = float(np.clip(self.cam.distance * (1.0 - amount * 0.1), 0.3, 12.0))


def add_line(scn, a, b, rgba, width=2.0) -> None:
    """One line segment into the scene's spare geoms, via mjv_connector."""
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    # mjv_connector only fills type/size/pos/mat; its own doc says to call
    # mjv_initGeom first for everything else. Skip it and the spare geoms stay
    # uninitialised, and mjr_render segfaults on them — process dies with SIGSEGV
    # and no Python traceback. Cost me a bisect; keep the init.
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3, dtype=np.float64),
                        np.zeros(3, dtype=np.float64),
                        np.eye(3, dtype=np.float64).flatten(),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width,
                         np.asarray(a, dtype=np.float64),
                         np.asarray(b, dtype=np.float64))
    geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
    scn.ngeom += 1


def draw_overlay(scn, robot_xy, ball: np.ndarray, target_distance: float) -> None:
    """Debug guides: link line, drop line, and the target-distance ring.

    Off by default (add --overlay). Useful when tuning the controller — the ring
    is the only way to see what distance the robot is actually aiming for — but
    it is visual clutter for the demo.
    """
    add_line(scn, ball, [ball[0], ball[1], 0.0], [1.0, 0.85, 0.05, 0.55], 1.0)
    add_line(scn, [robot_xy[0], robot_xy[1], ball[2]], ball, [0.4, 0.9, 1.0, 0.7], 1.5)

    segments, radius = 48, target_distance
    prev = None
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        p = np.array([ball[0] + radius * math.cos(a),
                      ball[1] + radius * math.sin(a), 0.005], dtype=np.float64)
        if prev is not None:
            add_line(scn, prev, p, [0.3, 1.0, 0.5, 0.6], 1.0)
        prev = p


def load_policy(model, data, walking: str, tricks: bool, action_scale: float,
                head_lp: float, legs_lp: float):
    """Build the controller, loading whatever trick policies exist.

    Trick durations are always passed: infer_policy's own defaults (a 3.0 s
    kick) leave the robot standing still long after the motion is over.

    A missing trick ONNX is skipped rather than fatal — the point of the demo is
    following, and every trick is optional.
    """
    kwargs = {
        "roulade_duration": ROULADE_DURATION,
        "kick_duration": KICK_DURATION,
    }
    if tricks:
        for name, filename in {
            "sitstand": "alpha_sitstand.onnx",
            "ground_pick": "alpha_ground_pick.onnx",
            "roulade": "roulade.onnx",
            "kick_left": "ball_kick_left.onnx",
            "kick_right": "ball_kick_right.onnx",
        }.items():
            path = POLICY_DIR / filename
            if path.is_file():
                kwargs[name + "_onnx_path"] = str(path)
            else:
                print(f"note: no {filename} - the {name} trick will be skipped")

    return ip.PolicyInference(
        model, data,
        walking_onnx_path=walking,
        action_scale=action_scale,
        head_lowpass=head_lp, legs_lowpass=legs_lp,
        new_cmd_obs=True, use_projected_gravity=True,
        **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drag the yellow ball; the duck follows and does random tricks.")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY),
                        help="walking policy ONNX")
    parser.add_argument("--target-distance", type=float, default=0.35,
                        help="how far from the ball the robot stops (m)")
    parser.add_argument("--action-scale", type=float, default=0.9,
                        help="robotd's walking default is 0.9")
    parser.add_argument("--head-lowpass", type=float, default=0.5,
                        help="robotd's value; the alpha policies are trained with it")
    parser.add_argument("--legs-lowpass", type=float, default=0.7)
    parser.add_argument("--no-tricks", action="store_true",
                        help="just follow the ball, never stop for a trick")
    parser.add_argument("--trick", action="append", choices=TRICKS, dest="tricks",
                        help="restrict tricks to this one; repeatable")
    parser.add_argument("--trick-gap", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=[TRICK_GAP_MIN, TRICK_GAP_MAX],
                        help="seconds of following between tricks (default: 6 14)")
    parser.add_argument("--seed", type=int, help="fix the random sequence")
    parser.add_argument("--overlay", action="store_true",
                        help="draw the link line and the target-distance ring")
    parser.add_argument("--xml", default=str(SCENE_XML))
    args = parser.parse_args()

    if not os.path.exists(args.policy):
        raise SystemExit(f"error: policy not found: {args.policy}")

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    policy = load_policy(model, data, args.policy, not args.no_tricks,
                         args.action_scale, args.head_lowpass, args.legs_lowpass)

    view = View(model)
    follower = BallFollower(args.target_distance)
    rng = random.Random(args.seed)
    tricks = TrickScheduler(rng, enabled=not args.no_tricks,
                            gap=tuple(args.trick_gap), only=args.tricks)

    def reset() -> None:
        free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        a = model.jnt_qposadr[free]
        data.qpos[a:a + 3] = [0.0, 0.0, 0.125]
        data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        for i, qi in enumerate(policy.joint_qpos_indices):
            data.qpos[qi] = policy.default_pose[i]
        data.qvel[:] = 0.0
        data.ctrl[:] = policy.default_pose
        data.mocap_pos[0] = [0.70, 0.0, BALL_Z]
        park_ball(policy, data)
        # Drop any trick that was mid-flight when R was pressed, or the posture
        # flag would stay flipped and the robot would sit there forever.
        # `is_sitstand` is deliberately NOT touched: it is set at load time and
        # _update_command needs it to write the sit flag (cmd[0] = sit_mode).
        # Clearing it makes every later sit a silent no-op.
        policy.behavior_mode = None
        policy.behavior_time_left = 0.0
        policy.ground_pick_mode = False
        policy.ground_pick_phase = 0.0
        if policy.sit_mode:
            policy.toggle_sit()          # flip the flag back the normal way
        if policy.walking_session is not None:
            policy.current_policy = "walking"
            policy.ort_session = policy.walking_session
        policy.previous_targets = None
        policy.vel_cmd[:] = 0.0
        policy._update_command()
        tricks.active = None
        tricks.stage = None
        tricks.timer = 0.0
        mujoco.mj_forward(model, data)

    reset()

    print(__doc__.strip().splitlines()[0])
    print(f"target distance: {args.target_distance:.2f} m   policy: {args.policy}")
    if tricks.enabled:
        usable = [t for t in TRICKS if tricks.available(policy, t)]
        if usable:
            print(f"tricks: {', '.join(usable)} "
                  f"(every {args.trick_gap[0]:.0f}-{args.trick_gap[1]:.0f} s)")
        else:
            print("tricks: none available (no trick policies found)")
    else:
        print("tricks: off")
    print("left-drag the yellow ball to move it; Q to quit")

    if not glfw.init():
        raise SystemExit("error: could not initialise glfw")
    window = glfw.create_window(view.width, view.height,
                                "Microduck - follow the yellow ball", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("error: could not create a window - is DISPLAY set?")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    viewport = mujoco.MjrRect(0, 0, view.width, view.height)

    mouse = {"x": 0.0, "y": 0.0, "left": False, "right": False, "middle": False}

    def on_mouse_button(win, button, action, mods):
        down = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["left"] = down
            view.grabbed = down and view.grab_hit(
                mouse["x"], mouse["y"], data.mocap_pos[0])
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            mouse["right"] = down
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            mouse["middle"] = down

    def on_cursor(win, xpos, ypos):
        dx, dy = xpos - mouse["x"], ypos - mouse["y"]
        mouse["x"], mouse["y"] = xpos, ypos
        if view.grabbed and mouse["left"]:
            hit = view.ray_to_plane(xpos, ypos, BALL_Z)
            if hit is not None:
                r = math.hypot(hit[0], hit[1])
                if r > DRAG_LIMIT:      # do not let the ball escape the arena
                    hit[0] *= DRAG_LIMIT / r
                    hit[1] *= DRAG_LIMIT / r
                data.mocap_pos[0] = [hit[0], hit[1], BALL_Z]
        elif mouse["left"]:
            view.orbit(dx, dy)
        elif mouse["right"] or mouse["middle"]:
            view.pan(dx, dy)

    def on_scroll(win, xoff, yoff):
        view.zoom(yoff)

    def on_key(win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_R:
            reset()
            print("reset")

    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    last = glfw.get_time()
    last_report = 0.0
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Real elapsed time, clamped: if the window stalls (a drag, a laptop
        # sleep) dt would be huge and fast-forward a trick to completion.
        now = glfw.get_time()
        dt = min(max(now - last, 0.0), MAX_DT)
        last = now

        # Trick timers run on wall time, as in infer_policy.py. They must tick
        # before infer(): ground_pick advances its phase and writes the command.
        policy.update_ground_pick_phase(dt)
        policy.update_behavior(dt)
        tricks.update(dt, policy, follower, data)

        if tricks.idle:
            vx, vy, vyaw = follower.command(
                data.qpos[0:2], data.qpos[3:7], data.mocap_pos[0])
            set_command(policy, vx, vy, vyaw)

        policy.apply_action(policy.infer())
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        fb_w, fb_h = glfw.get_framebuffer_size(window)
        viewport.width, viewport.height = fb_w, fb_h
        view.width, view.height = fb_w, fb_h

        mujoco.mjv_updateScene(model, data, view.opt, None, view.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, view.scn)
        if args.overlay:
            draw_overlay(view.scn, data.qpos[0:2], data.mocap_pos[0],
                         args.target_distance)

        mujoco.mjr_render(viewport, view.scn, ctx)

        # Real-time pacing: the loop is a 50 Hz controller, not a benchmark.
        target = 1.0 / CONTROL_HZ
        if glfw.get_time() - last < target:
            glfw.wait_events_timeout(target - (glfw.get_time() - last))

        glfw.swap_buffers(window)

        if glfw.get_time() - last_report > 0.5:
            last_report = glfw.get_time()
            trick = f"  [{tricks.label()}]" if tricks.label() else ""
            print(f"ball {follower.distance:.2f} m (target {args.target_distance:.2f})  "
                  f"{follower.status()}{trick}", flush=True)

    glfw.terminate()


if __name__ == "__main__":
    main()
