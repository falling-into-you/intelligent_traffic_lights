# Repository Guidelines

## Project Structure & Module Organization

This repository contains reinforcement-learning agents for the `intelligent_traffic_lights` project. Each agent variant is isolated in its own package:

- `agent_target_dqn/`, `agent_dqn/`, `agent_ppo/`, `agent_diy/`: algorithm implementations.
- Inside each agent package: `agent.py` defines the runtime agent, `algorithm/` contains learning logic, `model/` contains PyTorch models, `feature/` contains observation/action preprocessing, `workflow/` contains training loops, and `conf/` contains agent-specific settings.
- `conf/`: shared application and algorithm TOML configuration.
- `train_test.py`: local smoke-test entry point. Set `algorithm_name` to one of `target_dqn`, `dqn`, `ppo`, or `diy`.
- `kaiwu.json`: project metadata for the Kaiwu/Tencent AI Arena runtime.

## Build, Test, and Development Commands

No package manifest is included. Runtime dependencies such as `torch`, `numpy`, `kaiwudrl`, `common_python`, and `tools` are expected from the target environment.

```bash
python train_test.py
```

Runs the training test harness with the selected `algorithm_name`.

```bash
python -m py_compile train_test.py agent_target_dqn/agent.py
```

Checks syntax for edited Python files without launching training.

## Coding Style & Naming Conventions

Use Python 3, 4-space indentation, and UTF-8 files. Keep existing bilingual comments when editing nearby code. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase names for configuration constants such as `Config.LR`. Follow the existing package pattern when adding a new agent: `agent_<name>/{algorithm,conf,feature,model,workflow}`.

## Testing Guidelines

There is no dedicated test suite in this checkout. Use `train_test.py` as the main smoke test after changing agent behavior. For focused checks, compile changed modules with `python -m py_compile`. When changing preprocessing, validate both `observation_process` and `action_process`; when changing learning code, verify tensor shapes for all action heads.

## Commit & Pull Request Guidelines

Local Git history is unavailable in this checkout, so no project-specific commit pattern can be inferred. Use concise imperative commits, for example `Fix target DQN target network sync` or `Tune PPO reward shaping`. Pull requests should state the affected agent, summarize config changes, include the command used for validation, and mention any expected reward or training-impact observations. Do not include generated checkpoints, logs, or temporary experiment outputs unless explicitly required.

## Security & Configuration Tips

Keep environment-specific values in TOML files under `conf/` or each agent’s `conf/`. Do not hardcode secrets, absolute local paths, or machine-specific runtime settings in agent code.
