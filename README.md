# Microduck RL

<div align="center">

**Reinforcement learning for Microduck — an 800 g, 25 cm bipedal robot — built on
MuJoCo Warp, plus a target-following task, a command-block contract layer, and
an interactive drag-the-ball demo.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-4C1?logo=apache&logoColor=white)](LICENSE)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo%20Warp-EA4C1D)](https://github.com/google-deepmind/mujoco)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%20required-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=black)](README.md)
[![uv](https://img.shields.io/badge/env-uv-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![mjlab](https://img.shields.io/badge/Built%20on-mjlab-0090F5)](https://github.com/mujocolab/mjlab)
[![ruff](https://img.shields.io/badge/code%20style-ruff-D7FF64?logo=astral&logoColor=black)](https://docs.astral.sh/ruff/)

</div>

---

RL training environments for [Microduck](https://github.com/pollen-robotics/microduck),
built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) with PPO.
Policies train here at 50 Hz, export to ONNX, and run on the real robot via the
runtime in [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck).

<img width="2215" height="884" alt="Microduck" src="https://github.com/user-attachments/assets/5db7cc83-b3ce-4f7c-83f0-0572a63baed7" />

Real-robot montage — walking, standup, roulade, roller skating:
https://github.com/user-attachments/assets/50c3d537-8db2-4005-9d9c-3472faeec4d0

The repo encodes the full sim2real recipe: [BAM](https://github.com/Rhoban/bam)
actuator physics, domain randomization, backlash simulation, and the
reward-design lessons that made it work
(see [CLAUDE.md](CLAUDE.md) for the distilled playbook).

---

## Table of Contents

- [What's in this fork](#whats-in-this-fork)
- [Quickstart](#quickstart)
- [Interactive Demo: Follow the Yellow Ball](#interactive-demo-follow-the-yellow-ball)
- [Training Your Own Model](#training-your-own-model)
- [Deploying a Trained Policy](#deploying-a-trained-policy)
- [Tasks](#tasks)
- [Backlash Variants](#backlash-variants)
- [Actuator Model](#actuator-model)
- [Robot Model](#robot-model)
- [Project Structure](#project-structure)
- [Important Conventions](#important-conventions)
- [Gotchas](#gotchas)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## What's in this fork

This repository is [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
plus a target-following task and the tooling around it. Upstream is intact —
every original task, model, and convention still works as documented.

| Addition | What it does |
|---|---|
| **`Mjlab-BallFollow-Flat-MicroDuck`** | A trained task: walk to a target and hold a commanded stand-off distance, re-acquiring whenever the target moves. |
| **`scripts/ball_follow.py`** | Interactive demo — drag a yellow ball with the mouse and the robot follows it, breaking into random tricks. |
| **`command_contract.py`** | Declares what the 13 command-block slots mean, per task. The runtime copies them into the observation without interpreting them, so this is the only record of what to write there. |
| **`scripts/check_contract.py`** | Validates a task or ONNX against the deployment contracts before it reaches a robot. |
| **`scripts/stamp_contract.py`** | Writes the command spec into an exported ONNX's metadata, so the file documents itself. |
| **`scripts/deploy_local.py`** | Installs an exported ONNX where the local `robotd` loads it, with a provenance manifest. |
| **`TargetProvider`** | Pluggable interface for where the target comes from — random, drifting, scripted, or your own callback. |

Two of these exist because of bugs that were **silent**: a symmetry table reused
across command semantics, and a command block whose meaning was only written in
prose. Both produced policies that exported cleanly and behaved wrongly. See
[Command-block contracts](#command-block-contracts).

---

## Quickstart

Requires a CUDA GPU (training runs through MuJoCo Warp) and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yangyihai/Microduck_RL_Ball_Follow
cd Microduck_RL_Ball_Follow
uv sync                     # ~2 GB of CUDA wheels on first run
```

<details>
<summary><b>Troubleshooting <code>uv sync</code> on a slow or mirrored network</b></summary>

`uv sync` can fail part-way on a slow link: the default 30 s HTTP timeout is
shorter than one large wheel takes to arrive.

```bash
export UV_HTTP_TIMEOUT=600    # one-off, for this shell
```

Or permanently — lowering concurrency helps more than it sounds, because 50
parallel connections on a slow link each get an unusable slice:

```toml
# ~/.config/uv/uv.toml
concurrent-downloads = 8
http-timeout = 600
```

**If you are behind a PyPI mirror:** `uv` does **not** read
`~/.config/pip/pip.conf`, so a mirror configured for pip alone is silently
ignored and `uv` goes straight to `files.pythonhosted.org`. Put the mirror in
`~/.config/uv/uv.toml` instead:

```toml
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

Note that `uv` records the index it resolved against in `uv.lock`, so a mirrored
sync rewrites every package URL in it. Versions and hashes are unchanged — it is
a local artifact that must never be committed (it would break CI and HF Jobs).
Restore with `git show HEAD:uv.lock > uv.lock`.

</details>

Then:

```bash
# train the walking policy (uses your GPU; ~1-2 h for a usable gait at 4096 envs)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# watch a trained policy in the viewer
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# export to ONNX for deployment
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>

# drive the exported policy in CPU MuJoCo with the keyboard
uv run scripts/infer_policy.py --walking output.onnx
```

Resume training from a checkpoint:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

### Without a GPU

Add `--hf-jobs` to any training command to run it on Hugging Face Jobs instead
of locally (see `scripts/hf/README.md`).

---

## Interactive Demo: Follow the Yellow Ball

```bash
uv run scripts/ball_follow.py            # drag the ball; random tricks
uv run scripts/ball_follow.py --no-tricks
uv run scripts/ball_follow.py --trick sit
```

Drag the yellow ball with the left mouse button and the robot walks after it,
stopping at a set stand-off distance. Now and then it breaks into a random
trick — sits down, rolls over, touches the ground, kicks — then gets up and
carries on following. Both the trick and the gap between tricks are random.

| Input | Action |
|---|---|
| left-drag **on the ball** | move it (stays at its height, on a ground plane) |
| left-drag elsewhere | orbit the camera |
| right-drag / wheel | pan / zoom |
| `R` | reset |
| `Q` / `Esc` | quit |

| Flag | Meaning |
|---|---|
| `--target-distance 0.35` | stand-off distance, metres |
| `--trick-gap 6 14` | seconds of following between tricks |
| `--trick <name>` | restrict to one trick (`sit`, `roulade`, `ground_pick`, `kick_left`, `kick_right`) |
| `--seed N` | fix the random sequence |
| `--overlay` | draw the guide line and target-distance ring (off by default) |

The drivable ball is a **mocap body with collisions off**: it has no DOFs, so it
cannot be kicked away or roll off, and it is a target rather than an obstacle.
The window is a hand-rolled GLFW window rather than `launch_passive`, because
`mujoco.viewer.Handle` exposes no mouse API at all.

This demo is a *scripted* controller on top of the pretrained walking policy —
see the next section for why training the task end to end is the better route.

---

## Training Your Own Model

The full loop, start to finish:

```bash
uv run list-envs                        # is my task registered?
uv run train Mjlab-BallFollow-Flat-MicroDuck --env.scene.num-envs 4096
uv run play  Mjlab-BallFollow-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
uv run scripts/export.py Mjlab-BallFollow-Flat-MicroDuck --wandb-run-path <...>
uv run scripts/deploy_local.py walk out.onnx --task-id Mjlab-BallFollow-Flat-MicroDuck
```

> `scripts/infer_policy.py` is **not** a rehearsal for this task. It fills the
> command block with velocity commands, so a BallFollow policy would read
> `lin_vel_x` as a target offset. Use `play` (which runs the task's own command
> term) or write a small driver that supplies a target position. The same
> caveat applies to any task that redefines those slots.

Smoke-test a new config before spending hours on it — 64 envs and 5 iterations
catch most config errors for cents:

```bash
uv run train Mjlab-BallFollow-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

Do the same before a long run on unfamiliar hardware: the 4096 above assumes a
card with room for it (see [Gotchas](#gotchas)).

### `Mjlab-BallFollow-Flat-MicroDuck`

Walk to a target and hold a commanded stand-off distance, re-acquiring whenever
it moves. It is the training-side counterpart of `scripts/ball_follow.py`: that
script bolts a hand-written controller onto the pretrained walking policy, this
env trains one policy to do the job end to end.

That distinction matters more than it sounds. Everything awkward in the scripted
demo exists only because it is *not* one policy: it must issue bang-bang
commands (see [Gotchas](#gotchas)), add hysteresis, refuse to walk while
turning, stand still for 2 s after every trick, and treat "too close" as
"freeze" because the gait has no reverse. None of that is inherent to following
a target — it is the cost of driving a velocity-tracking policy through a
velocity interface to solve a position-seeking problem. A policy trained on the
task reads the target offset directly and can simply back off.

The unified 61-D layout is preserved exactly; the 3 `twist` slots are
**redefined** as `[target_x_body, target_y_body, hold_distance]`. Redefining
that slot is an established move here — `SitStand` uses `twist[0]` as a posture
flag — and the runtime fills the command block without interpreting it, so a
target position arrives through the same channel a velocity command would.

> **This task needs a target source.** The actor has no target sensor — the
> `BallKick` env is deliberately ball-blind for the same reason. So something
> upstream must supply the target position on the real robot (camera, motion
> capture, a UI picking a point). Decide that before training, or the policy
> will have nothing to follow on hardware.

### Extension points

| I want to change… | Override this |
|---|---|
| where the target comes from | `TargetProvider` (`target_provider_cls` on the cfg) |
| what the command block means | register a `CommandSpec` |
| reward shaping | the `ball_follow_*` weights / stds |
| network, PPO, symmetry | `MicroduckBallFollowRlCfg` |

`TargetProvider` is the seam for *what the robot chases* — swap it, leave the
command term and rewards alone.

| Provider | Behaviour |
|---|---|
| `RandomPolarTargetProvider` | default; static target, re-sampled on an interval |
| `DriftingTargetProvider` | the target keeps moving — harder, and closer to real use |
| `CallbackTargetProvider` | escape hatch: `target_fn=fn(env) -> [N,3]` world positions, every step |
| subclass | scripted paths, curricula, replayed detections |

One rule for providers: **do not read the robot pose in `reset()`.**
`command_manager.reset()` runs between the env writing the new pose and the next
`scene.update()`, so you would get the previous episode's. Sample relative
quantities there and resolve them inside `targets()`.

---

## Deploying a Trained Policy

```bash
uv run scripts/export.py Mjlab-BallFollow-Flat-MicroDuck --wandb-run-path <...> \
    --onnx-file out.onnx
uv run scripts/stamp_contract.py out.onnx Mjlab-BallFollow-Flat-MicroDuck
uv run scripts/check_contract.py --onnx out.onnx
uv run scripts/deploy_local.py walk out.onnx \
    --task-id Mjlab-BallFollow-Flat-MicroDuck \
    --write-config ../Microduck/deploy/robotd.toml
```

| Step | Why |
|---|---|
| `export.py` | bakes the observation normalizer into the graph — always deploy its output, never a hand-converted checkpoint |
| `stamp_contract.py` | writes the command-block spec into the ONNX metadata, so the file documents what must be written into it |
| `check_contract.py` | validates the deployment contracts |
| `deploy_local.py` | checks `obs[1,61] → actions[1,14]` with the same check `robotd` performs at load, installs the policy with a provenance manifest, and prints (or patches) the `[policy]` line |

Trained policies are installed to `../deployed_policies/`, **not** into
`Microduck/policies/`: the runtime's `xtask` test
`every_policy_in_the_repo_is_packaged` requires every `.onnx` in that directory
to appear in all three packaging manifests, so a locally trained file there
breaks CI. `robotd.toml` takes absolute paths by design.

### Command-block contracts

The runtime copies `twist` / `head_pose` / `body_pose` into the observation
without looking at what they mean. That is what makes hot-swapping work — and it
means **nothing checks that the numbers piped in are the numbers the policy was
trained on.** Four tasks in this repo redefine those same 13 slots; all are
61 wide, all export cleanly, and feeding one another's command block produces a
robot that does something confidently wrong with no error anywhere.

`src/mjlab_microduck/command_contract.py` makes the meaning explicit and
checkable. Validate before deploying:

```bash
uv run scripts/check_contract.py Mjlab-BallFollow-Flat-MicroDuck   # build + check a task
uv run scripts/check_contract.py --onnx out.onnx                   # check an export
```

It checks that the actor obs is 61-D, that the task declares a `CommandSpec`, and
that the symmetry table agrees with the declared mirror signs. That last one is
not theoretical: the velocity symmetry table mirrors `twist` as
`[vx, -vy, -vyaw]`, and reusing it for BallFollow negates the stand-off distance
— i.e. it trains the policy that the mirror image of "hold 0.35 m" is
"hold −0.35 m". Widths matched, training ran, nothing errored.

> **Hot-swap only holds between policies that share a command semantics.** A
> velocity policy handed a target position will read `target_x` as a forward
> speed. This is already true of `SitStand` vs `Velocity`.

---

## Tasks

Run `uv run list-envs` to print the registry. Some tasks have both a Flat and a
Rough terrain variant.

| Task ID | Terrain | Description |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | flat/rough | Main task: walk from a velocity command and head pose command |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | flat/rough | Walking and fall recovery in one policy |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | flat/rough | Rise from prone, supine or sitting, then stand with body pose control |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | flat/rough | Controlled sit↔stand in one policy, with head control |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | flat/rough | Crouch and touch the ground with the beak tip, then recover to standing |
| `Mjlab-BallKick-Flat-MicroDuck` | flat | Kick a 70 mm, 15 g ball forward (the actor is ball-blind) |
| `Mjlab-Roulade-Flat-MicroDuck` | flat | Roll forward and land on the feet |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | flat | Roller velocity tracking (passive wheels) |
| `Mjlab-Velocity-Swizzle-MicroDuck` | flat | The classic symmetric roller move |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | flat | Crouch while roller skating |
| `Mjlab-RollerSlope-Flat-MicroDuck` | slope | Roller down a slope |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | flat | Stand up onto the wheels from the ground |
| `Mjlab-Spin-Flat-MicroDuck` | flat | Fast spin in place on rollers |
| `Mjlab-BallFollow-Flat-MicroDuck` | flat | Seek a target and hold a commanded stand-off distance — [see above](#training-your-own-model) |

At deployment, the runtime hot-swaps these policies under one shared 61-D
observation protocol, so any policy can take over at any time.
`scripts/infer_policy.py` rehearses that:

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
  --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

Keyboard-driven (velocity commands, `G` ground touch, `Y` sit/stand, `R`
roulade, `K`/`L` kick), with `--debug`, `--save-csv` and `--record` for sim2real
comparison.

---

## Backlash Variants

Every main task has a Backlash variant, trained with a gear-play model of ±1°
in series with each of the 14 servo joints (2° total). Insert `-Backlash-`
before `MicroDuck` in the task ID — e.g.
`Mjlab-Velocity-Flat-Backlash-MicroDuck`.

The backlash is modeled properly for sim2real: each servo gets an unactuated
`passive_<joint>_backlash` hinge, and because the real encoder sits on the
output side of the play, both the firmware PD emulation
(`BacklashEncoderBamActuator`) and the `joint_pos` / `joint_vel` observations
read *through* the backlash (`qpos[servo] + qpos[backlash]`). Observation and
action dims are unchanged, so ONNX export and the runtime need no changes.
See `src/mjlab_microduck/tasks/backlash.py`.

---

## Actuator Model

All tasks use the [BAM](https://github.com/Rhoban/bam) M6 actuator model for the
Dynamixel XL330 (voltage control law, back-EMF, Coulomb/Stribeck/load-dependent
friction), with per-environment domain randomization of battery voltage, voltage
sag under load, command latency, and friction magnitude
(`FrictionDRBamActuator` in `src/mjlab_microduck/actuator/`).

At this scale — small servos driving an ~800 g biped — actuator fidelity is the
dominant sim2real gap, which is why actuators are modeled down to the voltage
control law rather than as ideal PD.

---

## Robot Model

MJCF models live in `src/mjlab_microduck/robot/microduck/`, exported from
Onshape via `onshape-to-robot`, each with a `config_mjcf_*.json`:

- `robot_walk.xml` — velocity tasks (torso/head contact points removed, cheaper
  fall cost)
- `robot_allcollisions.xml` — VelStand, StandUp, SitStand, GroundPick, BallKick,
  Roulade (the body can genuinely lie on the ground)
- `robot_allcollisions_rollers.xml` — roller tasks (passive wheels)
- `robot_*_backlash.xml` — backlash task variants (generated by `add_backlash.py`)
- `scene_track.xml` — walk model plus a draggable mocap target and a parked
  kickable ball, used by `scripts/ball_follow.py`

`scene*.xml` files wrap the robot with a floor and keyframes
(standing/sitting/folded) for quick viewing and for `infer_policy.py`.

---

## Project Structure

```text
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF exports, export configs, scenes, add_backlash.py
│   └── microduck_constants.py        # robot cfgs, HOME frame, BAM actuator cfg
├── actuator/friction_dr_bam.py       # BAM + friction DR + backlash encoder feedback
├── command_contract.py               # what the 13 command-block slots mean, per task
├── tasks/
│   ├── __init__.py                   # task registration (base + backlash variants)
│   ├── mdp.py                        # rewards, events, observations, custom classes
│   ├── backlash.py                   # make_backlash_variant() env-cfg wrapper
│   ├── symmetry.py                   # bilateral symmetry tables (one per command semantics)
│   └── microduck_*_env_cfg.py        # one cfg module per task family
├── train_cli.py                      # `train` entry point (+ --hf-jobs)
└── hf_jobs.py                        # Hugging Face Jobs submission

scripts/
├── ball_follow.py                    # drag-the-ball demo (own GLFW window)
├── infer_policy.py                   # CPU MuJoCo rehearsal, keyboard-driven
├── export.py                         # checkpoint → ONNX (bakes the normalizer)
├── stamp_contract.py                 # write the command spec into an ONNX
├── check_contract.py                 # validate deployment contracts
└── deploy_local.py                   # install an ONNX where robotd loads it
```

---

## Important Conventions

- The observation layout is shared across every policy (61-D actor obs:
  48 proprioception + commands `[twist(3), head_pose(4), body_pose(6)]`), which
  is what makes runtime policy hot-swapping possible. Envs that don't use a
  command slot zero-pad it rather than dropping it.
- Unactuated joints are all named `passive_*` (roller wheels, backlash hinges);
  actuators, joint observations and pose rewards select servo joints with
  `^(?!passive_).*`.
- Domain-randomization toggles are `ENABLE_*` booleans at the top of each env
  cfg file.
- Joint layout (14 servos): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee,
  ankle), 5–8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll), 9–13
  right leg.
- The exporter bakes the observation normalizer into the ONNX graph — always
  deploy ONNX produced by `scripts/export.py`, never a hand-converted
  checkpoint, or the policy sees unnormalized observations at runtime.

[CLAUDE.md](CLAUDE.md) documents the env-building workflow and the reward-design
rules learned across the project (also aimed at AI coding agents working in this
repo).

---

## Gotchas

Things that cost real time to find, collected so they only cost you a read.

**The walking policy has command dead zones.** Measured steady-state response of
`alpha_walking.onnx` in the rehearsal (5 ms, decimation 4, kp 0.55 position
actuators):

```text
cmd vx    0.15   0.20   0.22   0.25   0.30   →  0.000  0.000  0.090  0.103  0.122 m/s
cmd vyaw  0.3    0.6    1.0    1.5           →  0.001  0.000  0.447  0.746    rad/s
cmd vx   -0.20  -0.25  -0.30                 →  0.000  0.000  0.000  m/s  (no reverse)
```

Commands below ~0.22 m/s and ~0.8 rad/s produce *nothing*, and reverse does not
exist. So a proportional controller (`0.3 × error`) spends its life inside the
dead zone and the robot just stands there — command magnitudes must clear the
dead zone and let the error gate motion (with hysteresis), not scale it. Two
consequences worth internalising: any turn that works is abrupt (~0.45 rad/s),
so walking through one tips the robot over; and "keep your distance" can only
mean "stop", because there is no reverse gait.

**Match the runtime's low-pass if the rehearsal is meant to predict the robot.**
`robotd` low-passes the joint targets (head 0.5, legs 0.7 —
`deploy/robotd.toml`, `duck-control/src/control.rs`), and the alpha policies are
trained with it. `infer_policy.py` did not, so pass
`--head-lowpass 0.5 --legs-lowpass 0.7` when the point is sim2real prediction;
it cuts yaw jitter ~9% and visibly halves the sway. Note the sim runs at
`timestep 0.005` with `decimation 4` (= 50 Hz) — get either wrong in a headless
script and you are measuring a different controller.

**Two ways `infer_policy.py` goes deaf to the keyboard, both silent** — the
viewer opens and walks normally either way, so the symptom is only "the keys do
nothing":

1. **No TTY on stdin.** `TerminalInput` disables keyboard control when
   `sys.stdin.isatty()` is false, i.e. anything backgrounded (`nohup ... &`, a
   tool call, a script). Run it in the foreground of a terminal, and that
   terminal must hold keyboard focus.
2. **The key needs a policy that was not loaded.** `G` / `Y` / `R` / `K` / `L`
   drive `--ground-pick` / `--sitstand` / `--roulade` / `--kick-left` /
   `--kick-right`; without them the handler prints "unavailable" and does
   nothing. Arrow keys always work.

**Tricks each end differently; only `sit` needs handling by hand.**
`roulade` / `kick_*` self-return via a duration countdown and `ground_pick`
hands over at 70% of its period — both restore the walk policy themselves. `sit`
is a posture *flag*: nothing flips it back, and with only `--walking` loaded
`_update_policy_session` returns immediately (it needs walking *and* standing),
so a sit never ends on its own.

**The kick needs a joint literally named `ball_free`.** `_place_ball` looks it up
by name; without one the kick swings at air. `scene_track.xml` keeps a real ball
parked far outside the drag area, and it must be parked again after each kick or
the robot walks into it.

**8 GB of VRAM may not hold 4096 envs.** The headline number assumes a bigger
card; start lower and raise until it fits rather than discovering this an hour
into a run.

**Moving the checkout breaks the venv.** `uv sync` writes an absolute path into
`.venv/lib/python3.12/site-packages/mjlab_microduck.pth` and into the shebang of
every script in `.venv/bin/`, so renaming or moving the repo leaves
`uv run train` pointing at a directory that no longer exists. Re-run
`uv sync --frozen` and check `head -1 .venv/bin/train` if commands suddenly stop
being found.

---

## Testing

```bash
uv run --with pytest pytest tests/
```

CPU-only config-invariant and reward-function regression tests — they lock in
joint-index mappings, reward sign conventions, and NaN guards.

For a task you are about to deploy, also run:

```bash
uv run scripts/check_contract.py <task-id>       # widths, command spec, symmetry
uv run scripts/check_contract.py --onnx out.onnx # exported graph + declared task
```

---

## Contributing

Contributions are welcome. Before opening a pull request:

```bash
uv run --with pytest pytest tests/          # regression tests
uv run ruff format . && uv run ruff check . # formatting and lints
uv run train <task-id> --env.scene.num-envs 64 --agent.max_iterations 5
```

That last one matters: a 5-iteration smoke test at 64 envs catches the large
majority of config errors for cents. Never launch a long run without one.

If you add a task that redefines the command block, register a `CommandSpec` for
it and run `check_contract.py` — see
[Command-block contracts](#command-block-contracts) for why.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Acknowledgments

Built on the work of others:

- **[microduck](https://github.com/pollen-robotics/microduck)** — the robot and
  the runtime that runs these exported policies. This repo is a fork of its
  training environment, [microduck_rl](https://github.com/pollen-robotics/microduck_rl).
- **[mjlab](https://github.com/mujocolab/mjlab)** — the training framework
  (MuJoCo Warp + rsl_rl).
- **[BAM](https://github.com/Rhoban/bam)** — better actuator models, by Rhoban.
- **[MuJoCo](https://github.com/google-deepmind/mujoco)** and
  **[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)**.
