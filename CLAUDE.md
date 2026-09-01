 # CLAUDE.md

RL training environments for Microduck — a ~800 g, ~25 cm tall bipedal
robot with 14 Dynamixel XL330 servos — built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp) with PPO (rsl_rl). Policies are trained here at 50 Hz, exported to
ONNX, and deployed by the runtime in the `pollen-robotics/microduck` repo on
the real robot. Sim2real transfer
is the whole point: every convention below exists because breaking it produced a
policy that worked in the viewer and failed on hardware.

## Commands

```bash
uv run list-envs                                    # live task registry
uv run train <TASK_ID> --env.scene.num-envs 4096    # train (add --hf-jobs for Hugging Face Jobs)
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # SMOKE TEST — always run first
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>
uv run scripts/export.py <TASK_ID> --wandb-run-path <...>   # → ONNX (bakes obs normalizer — mandatory path)
uv run scripts/infer_policy.py --walking out.onnx   # CPU MuJoCo deployment rehearsal
uv run scripts/deploy_local.py walk out.onnx        # install it where the local runtime loads it
uv run scripts/ball_follow.py                       # drag the yellow ball; random tricks
uv run scripts/ball_follow.py --no-tricks           # just follow
uv run scripts/ball_follow.py --overlay             # show the guide line and ring
uv run --with pytest pytest tests/
```

`deploy_local.py <role> <onnx>` checks the graph is `obs[1,61] → actions[1,14]`,
copies it to `../deployed_policies/<role>.onnx` with a provenance manifest, and
prints the `[policy]` line for `robotd.toml` (`--write-config <path>` patches it
in place). Roles: `walk`, `stand`, `sitstand`, `ground_pick`, `kick_left`,
`kick_right`, `roulade`. The install directory is deliberately outside both
repos: the runtime's `xtask` test `every_policy_in_the_repo_is_packaged`
requires every `.onnx` under `Microduck/policies/` to appear in all three
packaging manifests, so a locally trained file does not belong there.

A 5-iteration smoke test at 64 envs catches ~95% of config errors for cents.
Never launch a long run without one.

## Training a model — the entry points

Everything a new task needs is already reachable from the CLI; adding a task
means adding a cfg module and registering it, not adding tooling.

```bash
uv run list-envs                       # is my task registered?
uv run train Mjlab-BallFollow-Flat-MicroDuck --env.scene.num-envs 4096
uv run play  Mjlab-BallFollow-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
uv run scripts/export.py Mjlab-BallFollow-Flat-MicroDuck --wandb-run-path <...>
uv run scripts/infer_policy.py --walking out.onnx          # CPU MuJoCo rehearsal
uv run scripts/deploy_local.py walk out.onnx               # where robotd loads it
```

Resume from a checkpoint:

```bash
uv run train Mjlab-BallFollow-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

`Mjlab-BallFollow-Flat-MicroDuck` is the worked example of a task built on top
of the velocity recipe — read `microduck_ball_follow_env_cfg.py`'s module
docstring before writing your own. It is the training-side counterpart of
`scripts/ball_follow.py`: that script bolts a hand-written controller onto the
pretrained walking policy, this env trains one policy to do the job end to end.

Two things it demonstrates that are easy to get wrong:

- **Redefining the command block.** The 61D obs has no slot for "where is the
  thing I am chasing", so the task puts `[target_x_body, target_y_body,
  hold_distance]` in the 3 twist slots. Width stays 61, so robotd's obs check,
  the ONNX shape and export are untouched — only the meaning changes, and the
  runtime is what fills that block anyway. Hot-swap then only holds between
  policies sharing a command semantics (already true of sitstand vs velocity).
- **A command term that changes every step.** `BallFollowCommand` stores a world
  target and reprojects it into the body frame on each `_update_command`, so
  the command keeps moving as the robot walks. Targets are *placed* on the
  first update, not at resample time: `command_manager.reset()` runs between the
  env writing the new robot pose and the next `scene.update()`, so
  `root_link_pos_w` is still the previous episode's there.
- **Symmetry tables are per-command-semantics.** `symmetry.py` mirrors the
  twist slot as `[vx, -vy, -vyaw]`; reusing it would negate the stand-off
  distance. Use `SYMMETRY_BALL_FOLLOW_CFG` (or write a table) for any task that
  redefines those slots.

Extension points — what to override instead of rewriting
--------------------------------------------------------
The interfaces below exist so a new task changes *one* thing. All are optional;
nothing breaks if you ignore them.

| I want to change... | Override this | Where |
|---|---|---|
| where the target comes from | `TargetProvider` (`target_provider_cls` on the cfg) | `tasks/mdp.py` |
| what the command block means | register a `CommandSpec` | `command_contract.py` |
| reward shaping | the `ball_follow_*` weights / stds | the env cfg module |
| network, PPO, symmetry | `MicroduckBallFollowRlCfg` | the env cfg module |

`TargetProvider` is the seam for *what the robot chases* — swap it, leave the
command term alone:

- `RandomPolarTargetProvider` (default) — static target, re-sampled on an
  interval. The training analogue of the operator dropping the ball somewhere
  new.
- `DriftingTargetProvider` — the target keeps moving; closer to real use and
  harder, because the policy must keep re-acquiring.
- subclass — scripted paths, a curriculum that starts near and recedes, replayed
  detections.
- `CallbackTargetProvider(target_fn=fn)` — escape hatch; `fn(env) -> [N,3]`
  world positions, called every step (keep it batched, it is on the hot path).

One rule for providers: **do not read the robot pose in `reset()`.**
`command_manager.reset()` runs between the env writing the new pose and the next
`scene.update()`, so you would get the previous episode's. Sample relative
quantities there and resolve them against the pose inside `targets()`.

Declare the command block once per task. The runtime copies twist/head/body into
the observation without interpreting them, so the `CommandSpec` is the only
record of what should be written there — and it is what lets a deployed policy
document itself:

```bash
uv run scripts/check_contract.py Mjlab-BallFollow-Flat-MicroDuck  # widths, spec, symmetry
uv run scripts/stamp_contract.py out.onnx Mjlab-BallFollow-Flat-MicroDuck
uv run scripts/check_contract.py --onnx out.onnx                  # reads it back
```

`check_contract.py` exists because every one of these failures is silent at
export time: wrong obs width, an undeclared command block, and a symmetry table
that disagrees with the declared mirror signs (that last one shipped a real bug
— see `check_mirror_signs`).

## Repo map

- `src/mjlab_microduck/tasks/mdp.py` — ALL custom MDP functions (rewards, events,
  observations, commands, curricula). Add new functions here, grouped by task.
  `BallFollowCommand` is the example of a command term (not just a reward).
- `src/mjlab_microduck/tasks/microduck_*_env_cfg.py` — one cfg module per task
  family. `microduck_velocity_env_cfg.py` is the main walking recipe AND the
  shared base (robot, DR, obs, commands) other envs build on or mirror.
- `src/mjlab_microduck/tasks/microduck_ball_follow_env_cfg.py` — the reference
  task for "seek a target and hold a distance". Train it with
  `uv run train Mjlab-BallFollow-Flat-MicroDuck ...`; read the module docstring
  when starting a new task.
- `src/mjlab_microduck/tasks/symmetry.py` — bilateral symmetry tables. There is
  one per command semantics (`SYMMETRY_CFG`, `SYMMETRY_BALL_FOLLOW_CFG`); pick
  the one matching your twist slot, or the mirror loss will negate slots that
  have no left/right sign.
- `src/mjlab_microduck/tasks/__init__.py` — task registration (base + `-Backlash-` variants).
- `src/mjlab_microduck/tasks/backlash.py` — wraps any env cfg into its backlash twin.
- `src/mjlab_microduck/robot/microduck_constants.py` — robot cfgs, HOME frame, BAM actuator cfg.
- `src/mjlab_microduck/robot/microduck/` — MJCF exports from Onshape
  (onshape-to-robot, one `config_mjcf_*.json` per model) + scenes + `add_backlash.py`.
- `src/mjlab_microduck/actuator/friction_dr_bam.py` — BAM actuator + friction DR + backlash encoder.
- `scripts/` — export, infer, sim2real comparison, wandb helpers.
- `src/mjlab_microduck/command_contract.py` — what the 13 command-block slots
  mean, per task. The runtime copies them into the obs without interpreting
  them, so this is the only record of what to write there.
- `scripts/check_contract.py` — validate a task or ONNX against the deployment
  contracts (61-D, spec declared, symmetry signs agree).
- `scripts/stamp_contract.py` — write the spec into an exported ONNX's metadata
  so the file documents itself.
- `scripts/deploy_local.py` — install an exported ONNX where the local `robotd`
  can load it (61-D check, provenance manifest incl. command block, `[policy]`
  patch).
- `scripts/ball_follow.py` + `robot/microduck/scene_track.xml` — drag a yellow
  mocap ball, the walking policy follows and stops at a set distance, breaking
  into a random trick (sit / roll / ground-pick / kick) now and then. Own GLFW
  window (see below). Guide lines and the target ring are hidden by default and
  only drawn with `--overlay`.
- `tests/` — cfg-invariant and mdp-function regression tests (CPU, no GPU needed).

## Invariants — do not break these

- **Obs layout is 61D (actor) and shared across the whole policy family** so
  policies are hot-swappable in the runtime: 48 base proprioception +
  13D command block `[twist(3), head_pose(4), body_pose(6)]`, in that order.
  An env that doesn't use a command slot ZERO-PADS it (keep the obs term,
  sample tiny ranges) — never delete a slot.
- **Joint layout** (14 servos, ctrl idx = joint idx on walk/allcollisions
  models): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5–8
  neck/head (neck_pitch, head_pitch, head_yaw, head_roll), 9–13 right leg.
  On roller/backlash models, passive joints INTERLEAVE — never hardcode joint
  indices in mdp functions; use the `_servo_joint_ids` / `_servo_joint_pos`
  helpers in mdp.py (identity on plain models, correct everywhere else).
- **Unactuated joints are all named `passive_*`** (wheels, backlash hinges).
  Every actuator/obs/reward selector uses `^(?!passive_).*` — keep the prefix
  convention when adding joints, and new `passive_` regexes must not
  accidentally match backlash joints (`^passive_.*wheel`, not `^passive_.*`).
- **Actuators are BAM** (voltage-controlled XL330 model, friction computed by
  the actuator). Two consequences: any STANDALONE env cfg must register the
  `expand_bam_friction_fields` startup event, and joint-friction DR must scale
  the actuator's `friction_scale` — `dof_frictionloss` is zeroed under BAM, so
  randomizing it is a silent no-op.
- **Obs normalization is ON** → the normalizer must be baked into the ONNX.
  `scripts/export.py` does this; in-sim play hides the bug (it applies the
  normalizer anyway), so never hand-convert a checkpoint.
- **Policies are UNFILTERED** (no action low-pass in training). Don't add EMA
  filtering without a matched runtime flag and a transfer test — trained-with /
  deployed-without (either direction) breaks transfer.
- **Domain randomization must not accumulate across resets.** mjlab 1.3.0's
  `dr.*` ops with `operation="add"/"scale"` are natively non-accumulating (they
  re-read compile-time defaults); custom DR functions must restore-then-apply.
  An accumulating CoM randomizer once degraded every long run for months.
- If an obs is remapped to a sensor view (backlash encoder, bias), any tracking
  REWARD on the same quantity must measure the same view — otherwise the policy
  is punished for correcting what it sees.
- `-Backlash-` task variants must mirror their base task's robot model
  (walk / allcollisions / rollers) so backlash A/B comparisons are unconfounded.

## Building a new env — the workflow

1. **Pick the closest template** and build on it, don't start from scratch:
   locomotion → the velocity recipe; episodic trick ending in a pose →
   standup; commanded two-state → sitstand; dynamic maneuver → roulade
   (read its cfg docstring — it encodes a 5-run lesson arc). Building on
   `make_microduck_velocity*_env_cfg` keeps DR / obs / noise / delays in sync
   for free; if you build standalone from mjlab's base template, you must port
   the whole DR + obs-noise + NaN-guard stack yourself (grep for what velocity
   wires: `_safe` critic obs terms, `nan_state` termination with sensor_names,
   `expand_bam_friction_fields`, encoder bias, IMU misalignment).
2. **Verify physics assumptions in sim BEFORE training** — this is the single
   biggest time-saver:
   - A target/rest pose must be a stable equilibrium: hold its ctrl for 3 s
     from noisy inits and check TILT, not just height (a settle test that only
     records z reports fallen states as "resting fine").
   - Measure target heights off the actual robot in sim (e.g. trunk z under a
     standing policy), never carry them across model revisions. A 5 mm-wrong
     STAND_Z once turned the goal into an impossible target for days.
3. **Config conventions**: `ENABLE_*` toggles + tuned constants at the top of
   the cfg file; factory `make_..._env_cfg(play: bool, rough: bool)`; register
   in `tasks/__init__.py` (+ the `_BACKLASH_TASKS` table if applicable); own
   `RslRl...RunnerCfg` with a distinct `experiment_name`. Symmetry mirror-loss
   is available (61D table in `symmetry.py`) — OFF by default, never for
   asymmetric tasks.
4. **Write cfg tests** (see `tests/test_*_cfg.py`): joint indices resolve on
   the actual model, reward weights have the intended sign, gates open/closed
   where expected. These run on CPU and lock in the invariants.
5. **Smoke test** (64 envs, 5 iters): builds, steps NaN-free, obs is 61D,
   every reward term computes, ONNX exports.
6. Train, watch the log (below), and expect 2–5 iterations of reward-hacking
   whack-a-mole — that's normal, the lessons below shortcut most of it.

## Reward design — rules that were each learned the hard way

- **Sign convention (bit four envs):** mdp.py has two penalty styles. mjlab-base
  cost functions return ≥ 0 → negative weight. Self-negating microduck functions
  (`*_penalty`, `*_l1` returning ≤ 0) → POSITIVE weight. A negative weight on a
  self-negating penalty double-negates into a reward for the violation, and the
  policy will farm it (butt-hopping, crash-sits). **The infallible check: on
  every run, every `Episode_Reward/<penalty>` in wandb must be ≤ 0.**
- **RL optimizes the letter of the reward.** Every under-specified degree of
  freedom will be exploited (ballistic whip instead of a roll, shoulder-roll
  instead of sagittal, head-tripod instead of standing). Encode what counts as
  the maneuver in hard state-based gates (support contact, orientation-axis
  checks, latches), not in small penalty nudges.
- **No jackpots:** any "reach X" reward must be rate-limited or slewed.
  Arriving early at a goal state that then pays per-step is a jackpot that
  buys arbitrary violence. For commanded transitions, track a slewed internal
  target (constant-rate blend) — being ahead of the ramp pays zero, so slow IS
  the argmax. Speed-cap penalties alone integrate to a bounded cost and lose.
- **Never gate a positive reward on being in a bad state** (fallen, low) — the
  policy parks in the cheapest qualifying pose and farms it. Use
  potential-based shaping instead (pay Δprogress, e.g. Δcos(tilt): rising pays,
  holding pays zero, unfarmable). For rest tasks, audit each positive term
  against every stable flop (on back / face / side): if flopping keeps most of
  the stack, the policy will flop.
- **Episodic pose-landing tasks:** single fixed target from t=0 (Gaussian + L1
  on joints and height, generous std) + |a_z| impact penalty + two-layer
  upright — NOT keyframe/waypoint trajectories (the policy camps at
  waypoints). The path is what RL is supposed to discover.
- **Regularizers come in two kinds.** Motion-blockers (body_ang_vel,
  angular_momentum, pose std) penalize what a dynamic motion physically
  requires — keep them LOW for dynamic tasks. Smoothness (action_rate,
  joint_torque_rate) damps jitter without blocking slow big motions — safe to
  weight, but introduce it AFTER skill discovery (curriculum from ~0): any
  attempt-tax active while a hard skill is being explored makes "do nothing"
  win. Slow careful tasks (reaching) want heavier smoothness than walking.
- **Compare reward mass, not weights, when copying regularizers between envs.**
  PPO sees relative advantage: the same action_rate weight is 4× weaker under a
  4×-larger positive task stack.
- **Tracking Gaussian std:** ≈ the error you still care about, not the max
  error — too loose has no gradient at small errors. BUT before tightening,
  ask whether the error is escapable by the policy or inherent to the behavior
  you want (a 38%-of-body-mass head MUST oscillate while walking; a tight
  instantaneous head-tracking std taxed walking so hard the policy stood
  still). Price only the escapable part — e.g. L1 on a 1 s EMA charges DC bias
  and lets oscillation cancel.
- **Multiplicative composites beat additive sums at goal states:** when an
  additive stack has a compromise basin (80% of every term via a lean), a
  product of Gaussians collapses on any single deficient factor — but pick stds
  wide enough that the CURRENT policy scores visibly, or the gradient is
  invisible and nothing changes.
- **Joints parking on hard limits:** fix with a qpos-side limit-proximity
  penalty on the offending joints; the stock `dof_pos_limits` only fires in the
  last ~7.5% of range, and command-side penalties don't work (wide ctrlrange is
  intentional — low-kp servos need overshoot).

## Commands, observations, dead weights

- **A command input that is never non-zero has dead weights forever.** Every
  command slot keeps a small non-zero sampling range from step 0 (even at
  reward weight 0) so its input neurons stay alive for later curricula.
- **Zero-command behavior must be explicitly trained** (`zero_command_prob`-style
  exact-zero sampling): uniform sampling essentially never produces the all-zero
  command, which is exactly the deployment idle state.
- Rare-but-important command regions need explicit buckets — e.g. turn-in-place
  (`rel_turn_in_place_envs`): independent uniform sampling made spinning ~2% of
  experience and it never trained.

## Curricula

- Steps are env steps: `iteration × 24` (`NUM_STEPS_PER_ENV = 24`).
- Use the proven split: `microduck_mdp.reward_weight` for weight schedules, a
  dedicated params-curriculum for command/event ranges. `mdp.reward_weight` is
  a step function, not an interpolation — discretize ramps into stages.
- Mutate term cfgs via the managers (`env.event_manager.get_term_cfg(...)`),
  never `env.cfg.events[...]` — managers deepcopy their cfg at init, so writes
  to `env.cfg` are silent no-ops (this also bites eval scripts that force
  spawn states).
- **Phase-align every stage with what the policy has actually learned**: don't
  harden spawn mixes before the current slice consolidates; don't introduce
  taxes before the skill exists. When a wandb metric steps DOWN exactly at
  curriculum stage boundaries, the pacing is wrong — stretch stages or move
  the introduction later, never earlier.
- Reverse-curriculum spawns (starting episodes partway through the maneuver,
  including nearly-done) are the reliable fix for "learns the start, never the
  last mile" — the frontier otherwise gets no on-policy data.

## Training ops & reading a run

- wandb project `mjlab_microduck`; logs in `logs/<experiment_name>/`; resume
  with `--agent.load-checkpoint model_XXXX.pt --agent.resume True`.
- Watch per-iteration: mean reward rising AND episode length behaving as the
  task demands; every penalty term ≤ 0; the MAIN task term actually growing
  (total reward can rise purely on regularizers while the trick never happens).
  `Episode_Reward/<term>` logs the WEIGHTED value — a term at weight 0 reads 0
  regardless of behavior, so interpret against the weight schedule.
- Budgets: simple episodic tricks ≈ 1000 iters at 4096 envs; gaits and
  curriculum-heavy recovery need 4000–6000.
- **Measure before theorizing.** When a run "fails", run a headless eval of the
  actual checkpoint (per-spawn-type batteries, end-state clusters, angular-rate
  profiles) before changing rewards: past "failures" turned out to be early
  checkpoints, a success criterion splitting one behavior cluster in half, and
  a pay cap fighting measured physics. Sim metrics can pass while the video
  fails the human eye — watch the video AND check which geom/axis touches.
- Report what rollouts actually show ("rolls but face-plants 1 in 3"), not
  "it works!". The user decides when it's good enough.

## Sim2real footguns (cost real debugging weeks)

- A fresh `uv sync` is the ground truth (HF Jobs run one): anything that only
  works via manually-installed local packages will die remotely. Keep
  `pyproject.toml` honest.
- Physics-aligned limits: a 25 cm robot tumbles at 3.5–5.5 rad/s NATURALLY —
  don't impose human-scale speed intuitions via caps; put anti-violence
  pressure on impacts and thrash (|a_z|, action_rate, support gates), not on
  rotation speed.
- IMU DR is zero-centered — it trains tolerance to misalignment magnitude, and
  CANNOT compensate a systematic mounting bias (that's a runtime calibration).
- Real deployments hot-swap ONNX policies (walk / stand / trick) with a shared
  obs contract — rehearse in `scripts/infer_policy.py` before touching the
  robot, with the correct command-slot writes (a posture flag lives in the
  twist vx slot; feeding all-zeros means "stand", which looks like "policy
  ignores the button").
- **A rehearsal must run the same controller as the robot, not just the same
  network.** `robotd` low-passes the joint targets (head 0.5, legs 0.7 —
  `deploy/robotd.toml`, `duck-control/src/control.rs`); `infer_policy.py` did
  not, so pass `--head-lowpass 0.5 --legs-lowpass 0.7` when the point is to
  predict real-robot behaviour — measured on `alpha_walking.onnx`, it cuts yaw
  jitter ~9% (std) and the sway visible in the viewer roughly halves. Note the
  sim also runs at `timestep 0.005` with `decimation 4` (= 50 Hz): get either
  wrong and you are running a different controller than the one you measured.
- **The walking policy has dead zones; a P controller will not work.** Measured
  on `alpha_walking.onnx` in the rehearsal (5 ms, decimation 4, kp 0.55
  position actuators), achieved steady-state response:

  ```
  cmd vx    0.15   0.20   0.22   0.25   0.30   →  0.000  0.000  0.090  0.103  0.122 m/s
  cmd vyaw  0.3    0.6    1.0    1.5           →  0.001  0.000  0.447  0.746    rad/s
  cmd vx   -0.20  -0.25  -0.30                 →  0.000  0.000  0.000  m/s  (no reverse)
  ```

  Commands below ~0.22 m/s and ~0.8 rad/s produce *nothing*, so command
  magnitude must be chosen above the dead zone and the error should only gate
  motion (with hysteresis), not scale it. "Keep your distance" can only mean
  "stop" — there is no reverse gait. Re-measure if the policy or actuator model
  changes; these are properties of this model, not of the task.
- **Overlay geoms must be `mjv_initGeom`-ed before `mjv_connector`.**
  `mjv_connector` fills only type/size/pos/mat; without the init, `mjr_render`
  reads uninitialised geoms and the process dies with **SIGSEGV and no Python
  traceback**. Budget a bisect if you hit it.
- **Two deadlock/robustness traps in the ball-follow controller, both found by
  running it for minutes rather than seconds:**
  1. *Scale-with-error commands must be floored at the dead zone.* An unclamped
     `GO + k*(err - ON)` decays below the ~0.22 m/s / ~0.8 rad/s dead zone as
     the error shrinks, while the hysteresis latch is still set (it only clears
     at OFF, below ON). The robot then issues a command too small to move, so
     the error never shrinks and it stays latched forever — observed stuck
     "turning" at 0.08 rad for 40 s. Fix: `max(CMD_*_GO, ...)`.
  2. *Never walk while turning, and stand still after a trick.* The dead zone
     means any effective turn is abrupt (~0.45 rad/s); walking through one, or
     stepping straight off a trick, tips the robot over — it fell every time
     right after standing up. Trick → RECOVER_TIME of standing → then follow.
- **Tricks each end differently; only one of the five needs handling by hand.**
  `roulade`/`kick_*` self-return via `update_behavior` and `ground_pick` via
  `update_ground_pick_phase` (hands over at 70% of its period, 2.8 s of 4 s) —
  both put the walk policy back themselves. **`sit` is a posture flag, not a
  timer**: nothing flips it back, and with only `--walking` loaded
  `_update_policy_session` returns immediately (it needs walking *and* standing),
  so a sit never ends on its own. The scheduler must flip it and then wait
  STAND_UP seconds for the robot to physically rise. Measured durations: a kick's
  motion is over by ~0.3 s, a roll's by ~1.6 s — hence kick 0.8 s and roulade
  2.0 s (infer_policy's own 3.0 s kick default just stands still afterwards).
- **The kick needs a joint literally named `ball_free`.** `_place_ball` looks it
  up by that name; without one the kick swings at air and says so. `scene_track`
  keeps a real ball parked at (20, 20), far outside the 3 m drag area, and it
  must be parked again after each kick or the robot walks into it.
- **`mujoco.viewer.launch_passive` exposes no mouse at all** — `Handle` has
  cam/opt/perturb/user_scn/sync and that is it. Anything pointer-driven (the
  ball demo) needs its own GLFW window; the fovy for projection is
  `model.vis.global_.fovy`, not on `MjvCamera`.
- **Two ways `infer_policy.py` goes deaf to the keyboard, both silent in the
  viewer** (it opens and walks normally either way, so the symptom is "the keys
  do nothing"):
  1. **No TTY on stdin.** `TerminalInput` disables keyboard control when
     `sys.stdin.isatty()` is false — i.e. anything backgrounded (`nohup ... &`,
     a tool call, a script). It does print `WARNING: stdin is not a TTY`, but
     into the same stream as fifty lines of startup banner. Run it in the
     foreground of a terminal, and that terminal must hold keyboard focus.
  2. **Key needs a policy that was not loaded.** `G`/`Y`/`R`/`K`/`L` drive
     `--ground-pick` / `--sitstand` / `--roulade` / `--kick-left` /
     `--kick-right`; without them the handler prints "unavailable: no
     --<policy> policy loaded" and does nothing. Arrow keys always work.
  Check both before concluding a key is broken.
