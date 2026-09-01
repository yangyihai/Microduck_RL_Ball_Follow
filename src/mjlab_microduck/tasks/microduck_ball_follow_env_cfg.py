"""Microduck BallFollow task — walk to a target and hold a stand-off distance.

This is the training-side counterpart of `scripts/ball_follow.py`. That script
demonstrates the behaviour by bolting a hand-written controller onto the
pretrained walking policy; this env trains a single policy to do it end to end.

Why train it at all, when the scripted version works?
-----------------------------------------------------
Because everything awkward in that script exists only to work around the fact
that it is *not* one policy. The scripted controller has to:

  - issue bang-bang commands, because the walking policy ignores anything below
    ~0.22 m/s and ~0.8 rad/s (see CLAUDE.md);
  - add hysteresis, or the robot chatters across that dead zone;
  - refuse to walk while turning, because the dead zone makes every effective
    turn abrupt enough to tip it over;
  - stand still for 2 s after every trick, or it falls on the first step;
  - treat "too close" as "stop", because the gait has no reverse.

None of that is inherent to following a target — it is the cost of driving a
velocity-tracking policy through a velocity interface to solve a
position-seeking problem. A policy trained on the task itself reads the target
offset and outputs joint targets directly: no dead zone, no hysteresis, no
recovery pause, and "too close" can mean "back off" instead of "freeze".

What the policy sees
--------------------
The unified 61D actor layout is preserved exactly:

    48D proprioception  (unchanged from the velocity env)
    13D command block   [twist(3), head_pose(4), body_pose(6)]

The 3 twist slots are REDEFINED as

    [target_x_body, target_y_body, hold_distance]

so the width stays 61 and robotd's obs check, the ONNX shape and the export
path are untouched. Redefining the twist slot is an established move here —
sitstand uses twist[0] as a sit/stand flag — and the runtime is what fills the
command block anyway (duck-control/src/obs.rs), so a target position arrives
through the same channel a velocity command would.

Two consequences to keep in mind:

  1. Hot-swap only works between policies sharing a command semantics. A
     velocity policy fed a target offset will read target_x as a forward speed.
     That is already true of sitstand vs velocity.
  2. Something upstream must supply the target position. On the real robot that
     means whatever is detecting the target (camera, motion capture, a UI
     picking a point on a map). The ball_kick env keeps its actor deliberately
     ball-blind for the same reason: no onboard ball sensing. Here the target
     is the task, so the assumption is explicit.

The target is a virtual world point held by the command term and drawn with
debug_vis — no prop. The demo's yellow ball is a mocap body with collisions off
for the same reason: a ball the robot can kick away stops being a target.

Training shape
--------------
Continuous (not episodic): 20 s episodes, and the target is re-sampled every
4-8 s, so one episode sees several "the operator dragged it somewhere else"
events rather than a single approach. Gait quality terms, DR, obs noise and
delays are all inherited from `make_microduck_velocity_env_cfg` — do not
rebuild those from scratch (see CLAUDE.md "Building a new env").
"""

import math
from copy import deepcopy

# ── Target sampling ──────────────────────────────────────────────────────────
# Where a fresh target is placed, relative to the robot as it stands at that
# moment (polar: distance along its heading, bearing). Bearing spans the full
# circle so the policy learns to approach from any direction, including from
# behind — it has to turn around, which is the hard case.
TARGET_DISTANCE_RANGE = (0.3, 1.6)      # m from the robot
TARGET_BEARING_RANGE = (-math.pi, math.pi)
TARGET_HEIGHT = 0.10                    # m; marker height, only drawn

# Stand-off distance to hold. Wide enough that the policy learns a distance to
# keep rather than a fixed pose to strike; narrow enough to stay learnable.
HOLD_DISTANCE_RANGE = (0.25, 0.45)      # m

# Seconds between target moves. Short vs the 20 s episode so a single episode
# contains several re-acquisitions.
TARGET_RESAMPLE_S = (4.0, 8.0)

# ── Reward shaping ───────────────────────────────────────────────────────────
# std, in metres of stand-off error. Sets how sharply the band is enforced, and
# therefore how precisely the robot has to park. 0.15 m is loose enough to be
# reachable from the first metres of random gait, tight enough to read as
# "holding a distance" once converged.
DISTANCE_STD = 0.15
# std, in radians of bearing error (0 = target straight ahead).
FACING_STD = 0.5
# Half-width of the band inside which loitering is penalised (m).
HOLD_STILL_TOLERANCE = 0.08

# ── Domain randomisation ─────────────────────────────────────────────────────
# Inherited wholesale from the velocity env (proven to transfer). The only
# local choice is episode length and the push schedule below.
EPISODE_LENGTH_S = 20.0

# Pushes: same ramp timing as velocity/standup. A target-seeking policy has to
# recover and re-acquire, which is exactly what this trains.
ENABLE_VELOCITY_PUSHES = True
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.3, 0.3)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.command_contract import BALL_FOLLOW_SPEC, register
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import (
    PpoWithSymmetryCfg,
    SYMMETRY_BALL_FOLLOW_CFG,
)

# Left/right mirror of a target on one side is a valid target on the other, so
# symmetry augmentation applies — unlike ball_kick, which is one-footed.
ENABLE_SYMMETRY = True

# What the command block means for this task. Registered on import so
# contracts.check_task (scripts/check_contract.py) can validate it and so an
# exported ONNX can carry it. The runtime copies twist/head/body into the obs
# without interpreting them, so this declaration is the only thing that says
# what to put there.
register(BALL_FOLLOW_SPEC)


def make_microduck_ball_follow_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck BallFollow environment configuration."""
    # Build on the MICRODUCK velocity recipe, not mjlab's bare template: this
    # task is locomotion, so it needs that env's gait rewards, contact/height
    # sensors, DR stack, obs noise and delays. Starting from mjlab's
    # make_velocity_env_cfg would mean re-wiring the scene sensors by hand —
    # the foot-height ray sensor needs microduck's site names, and silently
    # gets them wrong (empty ObjRef) if you inherit the bare template.
    cfg = make_microduck_velocity_env_cfg()

    # Walk model, flat terrain, and episode length are all inherited; only the
    # terrain generator is dropped (rough ground would confound the stand-off
    # objective with terrain noise).
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Commands: the twist slot becomes the target ──────────────────────────
    # Replacing commands["twist"] is enough to redefine the obs: the base
    # template's "command" observation term reads `generated_commands` for
    # command_name="twist", so it now emits [target_x, target_y, hold_distance]
    # with no change to the obs width (61).
    cfg.commands["twist"] = microduck_mdp.BallFollowCommandCfg(
        target_distance_range=TARGET_DISTANCE_RANGE,
        target_bearing_range=TARGET_BEARING_RANGE,
        target_height=TARGET_HEIGHT,
        hold_distance_range=HOLD_DISTANCE_RANGE,
        resampling_time_range=TARGET_RESAMPLE_S,
        debug_vis=not play,   # no marker clutter when watching a policy
    )

    # ── Rewards: swap velocity tracking for target seeking ───────────────────
    # Velocity tracking is meaningless here — there is no velocity command to
    # track. Everything else about walking well is kept.
    for name in ("track_linear_velocity", "track_angular_velocity"):
        if name in cfg.rewards:
            del cfg.rewards[name]

    # The objective: sit on the commanded stand-off band. Two-sided by design —
    # "get closer" alone walks into the target, "stay away" alone loiters at the
    # spawn radius.
    cfg.rewards["ball_follow_distance"] = RewardTermCfg(
        func=microduck_mdp.ball_follow_distance_reward,
        weight=2.0,
        params={"command_name": "twist", "std": DISTANCE_STD},
    )

    # Face the target. Distance alone is satisfied from any direction, so
    # without this the robot can sidle up crabwise or backwards.
    cfg.rewards["ball_follow_facing"] = RewardTermCfg(
        func=microduck_mdp.ball_follow_facing_reward,
        weight=1.0,
        params={"command_name": "twist", "std": FACING_STD},
    )

    # Stop shuffling once parked. The distance term saturates inside the band,
    # which otherwise leaves nothing to prefer standing still.
    cfg.rewards["ball_follow_hold_still"] = RewardTermCfg(
        func=microduck_mdp.ball_follow_hold_still_penalty,
        weight=-0.5,
        params={
            "command_name": "twist",
            "tolerance": HOLD_STILL_TOLERANCE,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # "standing envs" is a velocity-env idea: it zeroes the velocity command for
    # a fraction of envs so the policy learns to stand still. Here the command
    # IS the target, and zeroing it would mean "the target is on top of me",
    # which is a different (and confusing) situation rather than an idle one.
    # Drop the curriculum — it reads cfg.commands["twist"].rel_standing_envs,
    # which a target command does not have.
    cfg.curriculum.pop("standing_envs", None)

    # Head / body command slots are left exactly as the velocity env sets them
    # (live, non-zero ranges with tracking rewards), so those 10 input neurons
    # stay alive and the head stays steerable. A follow-up task could point the
    # head at the target; this one leaves the head command independent.

    # ── Terminations ─────────────────────────────────────────────────────────
    # fell_over is inherited from velocity and stays: a fallen robot cannot
    # follow anything, and continuing the episode would just train recovery
    # under a reward that no longer describes the task.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ───────────────────────────────────────────────────────────────
    # No target prop to reset — the command term owns the target and places it
    # relative to the reset robot pose on its own (see _place_pending).
    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # Ramp pushes in once the gait exists — same lesson as standup: a
        # full-strength shove at iter 0 taxes discovering the walk itself.
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 500 * 24,  "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 1000 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    # ── Terrain: flat only ──────────────────────────────────────────────────
    # Following a target across rough ground is a different (harder) task and
    # would confound the stand-off objective with terrain noise.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]

    return cfg


# ── RL runner config ─────────────────────────────────────────────────────────
MicroduckBallFollowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # NOT the velocity table: it negates the third twist slot (ang_vel_z),
        # which here is the stand-off distance and must not flip sign.
        symmetry_cfg=SYMMETRY_BALL_FOLLOW_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="ball_follow",
    run_name="ball_follow",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
