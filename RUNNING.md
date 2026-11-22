# Running the Multi-Agent Coordination Project

This guide covers setup, tests, training, evaluation, and result files for the
repository as it currently exists.

## Repository map

```text
agents/
  __init__.py
  communication.py   Learned, random, and helper communication components
  dqn.py              Shared multi-agent DQN and action-selection logic
  memory.py           GRU memory used by recurrent conditions
  replay_buffer.py    Transition and episode-sequence replay
environment/
  __init__.py
  grid_world.py       Grid environment, actions, rewards, and episode endings
  observations.py     Local/full observation windows and position features
evaluation/
  __init__.py
  baselines.py        Decentralized greedy heuristic baseline
  evaluate.py         Checkpoint and baseline evaluation entry points
  metrics.py          Success, return, completion-time, and task metrics
  mutual_information.py Message/direction metrics and bootstrap comparisons
  plots.py            Plotting helpers for result tables and training logs
  statistics.py       Aggregation and confidence-interval helpers
training/
  __init__.py
  config.py           Defaults, condition presets, and YAML overrides
  train.py            Training loop and command-line interface
tests/
  __init__.py
  conftest.py         Adds the repository root to the test import path
  test_baselines.py
  test_communication.py
  test_environment.py
  test_evaluation_engine.py
  test_evaluation_metrics.py
  test_mutual_information.py
.gitignore            Ignores generated checkpoints, results, and local files
requirements.txt      Python dependencies
README.md             Project overview and quick start
RUNNING.md            This runbook
```

`checkpoints/` and `results/` are runtime output directories, not source
directories. Training and evaluation create them as needed. Their generated
files are ignored by Git, so copy any artifacts you need to keep elsewhere.
There is no checked-in `configs/`, `scripts/`, or top-level analysis runner.
Training accepts a YAML file at any path through `--config`.

## 1. Set up Python

Run these commands from the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.12 was used to run the test suite for this repository. The project does
not declare a minimum Python version. If PowerShell blocks virtual-environment
activation, either allow it for the current terminal:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

or use the environment's interpreter directly, for example:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

`requirements.txt` installs NumPy, PyTorch, Matplotlib, pandas, pytest, and
PyYAML. PyTorch installation can be platform- and accelerator-dependent; if
the default pip install does not suit your machine, install the appropriate
PyTorch build before installing or rerunning the remaining requirements.

## 2. Run tests

Run the complete test suite:

```powershell
python -m pytest
```

Run one area while iterating:

```powershell
python -m pytest tests/test_environment.py
python -m pytest tests/test_communication.py
python -m pytest tests/test_evaluation_engine.py
```

Some end-to-end checkpoint tests are skipped when their optional local
checkpoints are absent. The suite does not require training a model first.

## 3. Train an agent

Start the default C1 training run:

```powershell
python -m training.train
```

The base run uses a 10 x 10 grid, four agents, four distinct targets, a
50-step episode limit, 6,000 training episodes, and seed 0. Training selects
CUDA when PyTorch reports it available; otherwise it runs on CPU. The current
default training hyperparameters are hidden size 128, learning rate 0.001,
discount 0.99, batch size 64, replay capacity 50,000, a 1,000-transition
warm-up for non-recurrent conditions, and an optimizer step every four
environment steps. The target network is synchronized every 250 steps.
Exploration starts at 1.0 and is reduced linearly to 0.05 over the first 70%
of training episodes. Communication uses four symbols with a one-step delay.
These are repository defaults, not numerical hyperparameters specified by
the proposal.

### Conditions and individual runs

The presets are:

| Condition | Preset radius | Observation | Memory | Message input |
|---|---:|---|---|---|
| C1 | `full` | Full grid | None | None |
| C2 | `2` | Local | None | None |
| C3 | `2` | Local | GRU | None |
| C4 | `2` | Local | None | Learned, one-step delayed |
| C5 | `2` | Local | GRU | Learned, one-step delayed |
| C6 | `2` | Local | None | Uniform-random, one-step delayed |

Run the six presets individually (these examples use seed 0):

```powershell
python -m training.train --condition C1 --radius full --seed 0
python -m training.train --condition C2 --radius 2 --seed 0
python -m training.train --condition C3 --radius 2 --seed 0
python -m training.train --condition C4 --radius 2 --seed 0
python -m training.train --condition C5 --radius 2 --seed 0
python -m training.train --condition C6 --radius 2 --seed 0
```

Valid radii in the project configuration are `1`, `2`, `3`, `5`, and `full`.
For C2-C6, omitting `--radius` selects the preset radius 2. An explicit
`--radius` overrides it, for example:

```powershell
python -m training.train --condition C4 --radius 3 --seed 42
```

Use `full` literally for a full-grid observation. C1 is the full-observation
reference; for the proposal's radius sweep, C2-C6 are trained at the requested
finite radii while C1 remains full.

### Proposal main-grid sweep

The proposal's main grid is N=4 with C1 as the full-observation reference and
C2-C6 at radii 1, 2, 3, and 5. It calls for five matched seeds and evaluation
on 200 fresh episodes per trained checkpoint. The following PowerShell loop
runs that grid serially, using seeds 0 through 4. It launches 105 training
runs in total, so expect a long job with the current 6,000-episode default.

First create `proposal-main.yaml` in the repository root to keep these
checkpoints and logs separate from other experiments:

```yaml
checkpoint_dir: checkpoints/proposal-main
results_dir: results/proposal-main
```

Then run:

```powershell
$conditions = @('C2', 'C3', 'C4', 'C5', 'C6')
$radii = @('1', '2', '3', '5')
$seeds = 0..4

foreach ($seed in $seeds) {
    python -m training.train --condition C1 --radius full --seed $seed --config proposal-main.yaml
    if ($LASTEXITCODE -ne 0) { throw "C1 seed $seed failed" }

    foreach ($condition in $conditions) {
        foreach ($radius in $radii) {
            python -m training.train --condition $condition --radius $radius --seed $seed --config proposal-main.yaml
            if ($LASTEXITCODE -ne 0) { throw "$condition radius $radius seed $seed failed" }
        }
    }
}
```

Evaluate every checkpoint from that sweep on 200 fresh episodes:

```powershell
python -m evaluation.evaluate `
  --checkpoints_dir checkpoints/proposal-main `
  --episodes 200 `
  --eval_seed 10000 `
  --results_dir results/proposal-main/evaluation
```

The proposal's stated time-saving order is to drop the bandwidth ablation,
then the team-size sweep, then radius 5, while retaining the C2-C5 memory by
communication 2 x 2 design at radii 1, 2, 3, and `full` with N=4. Choose this
reduced schedule explicitly rather than assuming the loop above applies it.

The proposal also includes a greedy heuristic comparison. This is an
evaluation baseline, not a trained condition, and currently has a Python API
but no CLI option or automatic inclusion in the sweep. Team-size and message
bandwidth ablations can be configured through YAML overrides such as
`num_agents`, `num_targets`, and `num_messages`; they are not generated by the
loop above. Keep `num_targets` equal to `num_agents` as required by the
environment. Since the run filename does not include `num_messages`, use
separate output directories for different bandwidth settings.

### Short smoke run

To try the training pipeline without running the 6,000-episode default, save
the following as `smoke.yaml` in the repository root:

```yaml
num_episodes: 2
batch_size: 4
min_replay_size: 8
replay_capacity: 500
log_interval: 1
train_frequency: 1
target_update_frequency: 10
checkpoint_dir: checkpoints/smoke
results_dir: results/smoke
```

Then run:

```powershell
python -m training.train --condition C2 --seed 7 --config smoke.yaml
```

The YAML values override preset values. The command-line `--radius` and
`--seed` are applied after YAML loading; avoid setting `condition` in YAML
unless intentionally replacing the selected preset. The training CLI accepts
`--condition`, `--radius`, `--seed`, and `--config`; it does not expose every
training hyperparameter as a command-line flag.

For recurrent C3/C5, the buffer trains on completed episode sequences; its
warm-up is based on the number of stored episodes rather than the
1,000-transition threshold used by non-recurrent conditions.

### Training outputs

For example, the default run writes:

```text
checkpoints/C1_rfull_N4_seed0.pt
results/C1_rfull_N4_seed0_train.csv
```

The CSV columns are `episode`, `return`, `success`, `steps`, `epsilon`, and
`loss`. The checkpoint contains the network weights, configuration, and
observation dimension needed for evaluation.

The run name uses condition, radius, agent count, and seed. Repeating the same
run with the same output directories overwrites its CSV and checkpoint; use
different seeds or output directories to preserve separate runs. For
reproducible comparisons, use the same seed across conditions and different
seeds for independent repetitions.

The logged episode return is the environment's shared reward. The learner's
replay transitions use per-agent shaped rewards in addition to that shared
reward, so the CSV return is not the per-agent training reward.

**Proposal-versus-implementation note:** the environment's shared reward
matches the proposal's coverage-change reward, but the current training loop
stores a separate shaped per-agent reward in replay (target arrival/exit,
holding, distance progress, step cost, and a terminal bonus). Therefore the
policy is not currently optimized solely on the proposal's shared reward.
Also, the proposal requires cross-seed confidence intervals and paired
comparisons; evaluating a directory writes per-checkpoint results and a
master list, but does not automatically aggregate those statistics or run
those hypothesis tests.

## 4. Evaluate checkpoints

Evaluate one checkpoint:

```powershell
python -m evaluation.evaluate `
  --checkpoint checkpoints/C2_r2_N4_seed42.pt `
  --episodes 200 `
  --eval_seed 10000 `
  --results_dir results/evaluation
```

Evaluate every `.pt` checkpoint in a directory:

```powershell
python -m evaluation.evaluate `
  --checkpoints_dir checkpoints `
  --episodes 200 `
  --eval_seed 10000 `
  --results_dir results/evaluation
```

Evaluation uses greedy action selection and creates fresh episodes with seeds
`eval_seed`, `eval_seed + 1`, and so on. It reports success rate, mean
completion time over successful episodes, mean return, and (for communication
conditions) message/direction mutual information. The default CLI evaluation
device is CPU. The evaluation CLI options are `--checkpoint`,
`--checkpoints_dir`, `--results_dir`, `--episodes`, and `--eval_seed`.

For each checkpoint, evaluation writes a summary JSON and episode CSV. It also
writes a message CSV when message data is available. Evaluating a directory
additionally writes `master_evaluation_summary.json`. Specify a separate
`--results_dir` to avoid mixing evaluation outputs with training logs.

The greedy heuristic baseline is available as a Python API; there is no
dedicated baseline command-line option:

```powershell
python -c "from environment.grid_world import GridWorld; from evaluation.evaluate import evaluate_heuristic_baseline; print(evaluate_heuristic_baseline(GridWorld(radius=2), num_episodes=200, seed=10000))"
```

`evaluation.plots` and `evaluation.statistics` provide callable analysis
helpers; they are not assembled into a single command-line analysis pipeline.
For example, plot the rolling curves in one training log with:

```powershell
python -c "from evaluation.plots import plot_learning_curves_from_csv; plot_learning_curves_from_csv(['results/C1_rfull_N4_seed0_train.csv'], output_path='results/plots/learning_curves.png')"
```

## 5. Environment and modeling notes

- Actions are `0=up`, `1=down`, `2=left`, `3=right`, and `4=stay`.
- `GridWorld.reset()` returns `(windows, pbar)`. The windows contain three
  channels and are local `(N, 3, 2r+1, 2r+1)` or full-grid
  `(N, 3, grid_size, grid_size)` observations; `pbar` contains each agent's
  normalized position. The third window channel is walls for local views and a
  self-position marker for full views.
- Targets are distinct; agents are allowed to share a cell. A step returns
  observations, reward, `done`, and an info dictionary. Check `info["success"]`
  to distinguish task completion from a time-limit truncation.
- The environment reward reflects the change in the number of covered targets,
  includes a `-0.01` per-step cost, penalizes losing coverage, and gives a
  completion bonus. Reaching the episode limit is not itself a successful
  completion.
- C3 and C5 use recurrent sequence replay and can require more memory and
  training time than the non-recurrent conditions. C4 and C5 use learned
  communication; C6 is the random-communication control.
- Run modules from the repository root with `python -m ...` so package imports
  resolve consistently. Avoid running `training/train.py` or
  `evaluation/evaluate.py` as direct file paths.
- `python -m evaluation.evaluate --help` may display a `runpy` warning because
  `evaluation/__init__.py` imports the evaluation module. The CLI options are
  still printed; this warning is separate from model evaluation results.
