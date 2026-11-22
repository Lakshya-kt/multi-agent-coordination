# Seeing vs. Saying: Multi-Agent Coordination

For the current repository map, setup instructions, and detailed training,
evaluation, and testing commands, see [RUNNING.md](RUNNING.md).

## Project Overview

This project studies how observation range,
memory, and communication jointly affect
multi-agent coordination.

## Environment

- 10 x 10 grid
- Multiple agents
- Distinct static targets
- Five possible actions
- Maximum episode length: 50 steps

## Communication

- Four discrete message symbols: 0, 1, 2, 3
- Broadcast communication
- One-step message delay
- Learned communication using Gumbel-Softmax

## Experimental Conditions

- C1: Full observation, no memory, no communication
- C2: Local observation, no memory, no communication
- C3: Local observation with GRU memory
- C4: Local observation with learned communication
- C5: Local observation with memory and communication
- C6: Local observation with random communication

## Project Structure

- environment/: Grid world and observations
- agents/: DQN, memory, and communication
- training/: Training loop and configurations
- evaluation/: Metrics and experiments
- tests/: Unit tests

## Setup

```bash
pip install -r requirements.txt

```
## Running
```bash
python -m training.train
```

## Simplified folder structure

``` text
multi-agent-coordination/
│
├── environment/
│   ├── __init__.py
│   ├── grid_world.py
│   └── observations.py
│
├── agents/
│   ├── __init__.py
│   ├── dqn.py
│   ├── replay_buffer.py
│   ├── communication.py
│   └── memory.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   └── config.py
│
├── evaluation/
│   ├── __init__.py
│   ├── baselines.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── mutual_information.py
│   ├── plots.py
│   └── statistics.py
│
├── tests/
│   ├── test_environment.py
│   └── test_communication.py
│
├── checkpoints/
├── results/
├── requirements.txt
├── .gitignore
├── README.md
└── RUNNING.md
