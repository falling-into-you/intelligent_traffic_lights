# 智能交通信号灯环境说明

本文整理自腾讯开悟智能交通信号灯调度开发指南，用于说明环境配置、观测数据、动作空间和训练时需要关注的指标。

## 环境配置

训练开始时，工作流会读取对应智能体目录下的 `train_env_conf.toml`，并传入 `env.reset()`：

```python
usr_conf = read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", logger)
env_obs = env.reset(usr_conf=usr_conf)
```

各算法目录都有自己的环境配置文件，例如：

```text
agent_target_dqn/conf/train_env_conf.toml
agent_dqn/conf/train_env_conf.toml
agent_ppo/conf/train_env_conf.toml
agent_diy/conf/train_env_conf.toml
```

常用配置项如下：

| 配置 | 字段 | 说明 |
| --- | --- | --- |
| 天气 | `weather` | `-1` 随机，`0` 晴天，`1` 雨天，`2` 雪天，`3` 雾霾 |
| 高峰期 | `rush_hour` | `-1` 随机，`0` 正常，`1` 高峰 |
| 速度限制 | `speed_limit` | 车道最大限速，范围 `[5, 17]`，单位 m/s |
| 超速车辆 | `speeding_cars_rate` | 无规则超速车辆概率，范围 `[0, 4]` |
| 等待车辆上限 | `max_waiting_cars` | 等待车辆数阈值，范围 `[150, 450]` |
| 等待持续时长 | `max_waiting_cars_duration` | 连续等待超时阈值，范围 `[80, 200]` 步 |
| 最大步数 | `max_step` | 单局最大步数，范围 `[100, 2000]` |
| 交通事故 | `traffic_accidents` | 可随机生成或指定事故车道与时间 |
| 交通管制 | `traffic_control` | 格式同交通事故 |

这些配置只影响训练环境。评估任务的环境配置需要在平台评估任务中设置。

## 环境交互流程

环境主要通过两个接口交互：

```python
env_obs = env.reset(usr_conf=usr_conf)
score, env_obs = env.step(act)
```

`reset()` 用于初始化环境，返回初始 `observation` 和 `extra_info`。`step(act)` 执行动作，返回当前得分和新的环境状态。

一次典型训练循环是：

```text
读取配置 -> reset 环境 -> 处理 observation -> 预测动作 -> step 执行动作
-> 计算 reward -> 组装样本 -> learner 更新模型
```

## Observation 数据

环境返回的 `observation` 主要包含：

| 字段 | 说明 |
| --- | --- |
| `frame_state` | 当前帧的原始交通状态 |
| `legal_action` | 当前是否允许智能体决策 |

`frame_state` 中包含：

| 字段 | 说明 |
| --- | --- |
| `frame_no` | 当前帧数 |
| `frame_time` | 当前仿真时间 |
| `vehicles` | 当前场景内车辆列表 |
| `phases` | 信号灯信息 |
| `lanes` | 车道信息 |

当前代码主要在 `agent_target_dqn/agent.py` 的 `observation_process()` 中读取 `vehicles`，并把车辆位置和速度编码成 560 维状态特征。

## Vehicle 数据

`vehicles` 只包含当前仍在场景内的车辆。驶离路网的车辆不会继续出现在列表里。

常用车辆字段：

| 字段 | 说明 |
| --- | --- |
| `v_id` | 车辆唯一编号 |
| `v_config_id` | 车辆属性配置 ID |
| `lane` | 当前车道 ID |
| `junction` | 当前所在交叉口 ID |
| `target_junction` | 目标交叉口 ID |
| `position_in_lane` | 车辆在车道内的位置，包含 `x` 和 `y` |
| `speed` | 速度，单位 m/s |
| `accel` | 加速度 |
| `v_status` | 车辆状态，`0` 正常，`1` 事故，`2` 无规则车辆 |
| `delay` | 延误时间，单位 s |
| `waiting_time` | 等待时间，单位 s |

奖励函数可直接使用 `speed`、`waiting_time`、`lane`、`target_junction` 等字段。当前 `target_dqn` 的基础 reward 使用进口车道车辆的等待时间变化和停车比例。

## 信号灯与车道数据

信号灯 `phases` 中常用字段：

| 字段 | 说明 |
| --- | --- |
| `s_id` | 信号灯编号 |
| `phase_id` | 当前信号灯相位 |
| `duration` | 当前相位持续时间，单位 s |
| `remaining_duration` | 当前相位剩余时间，单位 s |
| `lights` | 灯色 mask 信息 |

车道 `lanes` 中常用字段：

| 字段 | 说明 |
| --- | --- |
| `lane_id` | 车道 ID |
| `v_count` | 当前车道车流量 |
| `congestion` | 当前车道拥堵程度 |
| `queue_length` | 当前车道排队长度 |

这些字段适合后续增强特征和 reward，例如加入车道排队长度或拥堵程度。

## 动作空间

传入 `env.step(act)` 的动作格式是：

```text
[junction_id, phase_idx, duration]
```

本环境中 `junction_id` 固定为 `0`。

`phase_idx` 有 4 个取值：

| phase_idx | 含义 |
| --- | --- |
| `0` | 南北直行 |
| `1` | 南北左转 |
| `2` | 东西直行 |
| `3` | 东西左转 |

`duration` 表示本次相位持续时间。指南说明环境动作里的 duration 单位是毫秒，而代码包中智能体默认输出 `0-19` 的整数，语义是秒。因此在传入 `env.step()` 前需要确认是否已完成秒到毫秒的换算。

## 时间单位

环境中 step、frame、仿真时间存在映射关系：

```text
1 frame 约等于 66-68 ms
1 step = 10 frame
```

强化学习中的一步 step 表示智能体执行一次动作并接收环境反馈。本环境中，一个动作会在连续 10 个 frame 内保持生效。

## Score 与 Reward

`env.step(act)` 返回的 `score` 是环境评分，用于衡量模型表现。它不等同于强化学习训练使用的 `reward`。

训练中的 `reward` 由代码自定义，例如：

```text
agent_target_dqn/feature/definition.py::reward_shaping()
```

区别如下：

| 项目 | 来源 | 用途 |
| --- | --- | --- |
| `score` | 环境返回 | 评估模型表现 |
| `reward` | 代码自定义 | 指导强化学习更新 |

## 监控指标

平台会提供环境监控指标：

| 指标 | 说明 |
| --- | --- |
| `score` | 总得分 |
| `avg_delay` | 平均延误 |
| `avg_waiting_time` | 平均等待时间 |
| `avg_queue_len` | 平均排队长度 |
| `avg_phase_change_punish_cnt` | 平均信号变化频率 |

调试 reward 时，应重点观察 `avg_waiting_time`、`avg_queue_len` 和 `avg_phase_change_punish_cnt`。如果等待时间下降但相位切换频率明显升高，需要在 reward 或动作策略中加入切换成本。

## 当前代码对应关系

| 任务 | 文件 |
| --- | --- |
| 读取环境配置 | `agent_target_dqn/workflow/train_workflow.py` |
| 原始状态转模型输入 | `agent_target_dqn/agent.py` |
| 车辆与车道辅助判断 | `agent_target_dqn/feature/traffic_utils.py` |
| 动态交通信息缓存 | `agent_target_dqn/feature/preprocessor.py` |
| reward 计算 | `agent_target_dqn/feature/definition.py` |
| 动作输出 | `agent_target_dqn/agent.py::action_process()` |
