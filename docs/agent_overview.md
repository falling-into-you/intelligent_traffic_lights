# 智能体概述

本文整理自腾讯开悟智能交通信号灯调度开发指南，用于说明代码包中智能体的观测处理、奖励处理、算法选择和监控指标。

## 智能体职责

智能体负责把环境返回的原始观测转成模型输入，并输出信号灯动作：

```text
observation -> feature -> model -> action
```

在本项目中，动作包含：

```text
[junction_id, phase_idx, duration]
```

其中 `junction_id` 在单路口场景中固定为 `0`，`phase_idx` 表示信号灯相位，`duration` 表示该相位持续时间。

## 观测处理

环境返回的 `observation` 包含原始交通状态。智能体在 `observation_process()` 中处理这些信息。

当前代码会先调用预处理器：

```python
def observation_process(self, raw_obs, extra_info):
    self.preprocess.update_traffic_info(raw_obs, extra_info)
```

预处理器负责更新车辆历史信息、等待时间、行驶距离、车道车辆数量等动态交通状态。相关代码位于：

```text
agent_target_dqn/feature/preprocessor.py
```

## 特征组成

代码包默认使用两类特征：

| 特征 | 类型 | 说明 |
| --- | --- | --- |
| `position` | `list` | 车辆位置信息 |
| `speed` | `list` | 车辆速度信息 |

最终输入模型的特征为：

```python
feature = position + speed
return ObsData(feature=feature)
```

当前配置中：

```python
GRID_WIDTH = 14
GRID_NUM = 20
DIM_OF_OBSERVATION = 560
```

因此：

```text
position: 14 * 20 = 280
speed:    14 * 20 = 280
feature:  280 + 280 = 560
```

## 位置信息处理

地图中的进口车道会被离散成二维栅格：

```text
(Config.GRID_WIDTH, Config.GRID_NUM)
```

含义：

| 维度 | 说明 |
| --- | --- |
| `GRID_WIDTH` | 进口车道数量，当前为 14 |
| `GRID_NUM` | 每条车道沿行驶方向划分的栅格数，当前为 20 |
| `GRID_LENGTH` | 单个栅格长度 |

处理逻辑：

```text
遍历车辆
只保留进口车道车辆
根据 lane 映射到 x_pos
根据 position_in_lane["y"] 映射到 y_pos
对应栅格置为 1
```

核心辅助函数：

```text
on_enter_lane(vehicle)
get_lane_code(vehicle)
```

位置信息是一种 one-hot 编码。某个栅格有车时，该位置为 `1`，否则为 `0`。

## 速度信息处理

速度特征和位置特征使用相同的二维栅格。

处理逻辑：

```text
遍历车辆
只保留进口车道车辆
定位到对应车道和栅格
写入归一化速度
```

归一化公式：

```python
vehicle["speed"] / self.preprocess.vehicle_configs_dict[vehicle["v_config_id"]]["max_speed"]
```

这样可以把不同车辆配置下的速度映射到相近尺度，便于模型学习。

## 奖励处理

代码包只提供 reward 函数结构，具体奖励需要根据环境评价指标自行设计。

默认结构：

```python
def reward_shaping(_obs, act, agent):
    frame_state = _obs["frame_state"]
    vehicles = frame_state["vehicles"]
    return 0
```

当前 `target_dqn` 已实现基础奖励：

```text
等待时间下降 -> 奖励增加
停车比例升高 -> 奖励降低
```

相关文件：

```text
agent_target_dqn/feature/definition.py
```

后续可继续加入：

```text
车辆延误 delay
车道排队长度 queue_length
相位切换惩罚
duration 过短惩罚
```

## 算法选择

代码包提供 4 套算法目录：

| 目录 | 说明 |
| --- | --- |
| `agent_dqn/` | DQN |
| `agent_target_dqn/` | Target-DQN |
| `agent_ppo/` | PPO |
| `agent_diy/` | 自定义算法模板 |

在 `train_test.py` 中切换算法：

```python
algorithm_name = "target_dqn"
```

可选值：

```text
target_dqn
dqn
ppo
diy
```

## DQN

DQN 将 Q-Learning 和深度神经网络结合，用神经网络拟合 Q 值函数：

```text
Q(s, a)
```

含义是在状态 `s` 下执行动作 `a` 的长期预期回报。

DQN 可以处理高维状态输入，但训练稳定性较弱，需要经验回放和调参支持。

## Target-DQN

Target-DQN 是 DQN 的稳定性改进版本。

它使用两个网络：

| 网络 | 作用 |
| --- | --- |
| online model | 当前正在训练的策略网络 |
| target model | 计算目标 Q 值的目标网络 |

目标网络不会每一步都更新，而是隔一段时间从 online model 同步参数。这样可以降低目标值震荡，提高训练稳定性。

当前优先学习和修改的是：

```text
agent_target_dqn/
```

## PPO

PPO 是策略梯度类算法。核心思想是限制每次策略更新幅度：

```text
小步多次更新，避免策略崩溃
```

PPO 通常包含：

```text
policy loss
value loss
entropy loss
```

当前课程代码中，PPO 的模型结构、clip loss、entropy loss 和总 loss 仍留有 TODO。

## 训练与评估决策

训练模式一般调用：

```python
agent.predict(...)
```

评估模式一般调用：

```python
agent.exploit(...)
```

两者可以使用不同策略。常见做法是：

```text
predict: 保留探索，例如 epsilon-greedy
exploit: 使用确定性动作，例如 argmax
```

当前 `target_dqn` 训练时使用 epsilon-greedy，评估时通过 `exploit_flag=True` 关闭随机探索。

## 算法监控指标

DQN 和 Target-DQN 常用指标：

| 指标 | 说明 |
| --- | --- |
| `reward` | 累积回报，正常训练时应整体震荡向上 |
| `q_value` | Q 值均值，可观察 Q 估计是否稳定 |
| `value_loss` | Q 目标和当前 Q 预测之间的误差，通常应逐步下降或趋稳 |

PPO 常用指标：

| 指标 | 说明 |
| --- | --- |
| `reward` | 累积回报，正常训练时应整体震荡向上 |
| `value_loss` | 价值函数预测误差 |
| `policy_loss` | 策略优化目标，通常在 0 附近震荡 |
| `entropy_loss` | 策略探索性和多样性 |

## 模型保存限制

平台对模型保存频率有限制：

| 限制 | 数值 |
| --- | --- |
| 保存频率 | 2 次/分钟 |
| DQN / Target-DQN / PPO 单任务保存次数 | 200 次 |
| DIY 单任务保存次数 | 200 次 |

训练代码中不要过于频繁调用 `agent.save_model()`。

## 学习顺序建议

当前默认运行：

```python
algorithm_name = "target_dqn"
```

建议学习顺序：

```text
1. 先理解 observation_process 如何构造 560 维特征
2. 再理解 reward_shaping 如何影响学习方向
3. 再理解 Target-DQN 的 Q 值更新
4. 最后再看 PPO
```

不要先跳到 PPO。当前重点是让 `agent_target_dqn/` 跑通并具备基本学习信号。
