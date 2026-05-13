# 数据协议

本文整理自腾讯开悟智能交通信号灯调度开发指南，用于说明环境返回数据、动作数据和初始化静态信息的结构。

## 总览

环境交互中最常用的数据有三类：

| 类型 | 说明 |
| --- | --- |
| `Observation` | 智能体每步收到的观测 |
| `ExtraInfo` | 初始化信息、得分信息和错误信息 |
| `Action` / `Command` | 智能体发送给环境的控制指令 |

当前代码中主要使用：

```text
observation["frame_state"]
observation["legal_action"]
extra_info["init_state"]
```

## Observation

```protobuf
message Observation {
  FrameState frame_state = 1;
  repeated int32 legal_action = 2;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `frame_state` | 当前帧交通状态 |
| `legal_action` | 当前是否允许智能体执行动作 |

代码中的使用位置：

```text
agent_target_dqn/workflow/train_workflow.py
agent_target_dqn/agent.py
```

`workflow` 会先判断：

```python
need_to_predict = obs["legal_action"][0] != 0
```

只有允许决策时，才调用 `agent.predict()`。

## ExtraInfo

```protobuf
message ExtraInfo {
  InitState init_state = 1;
  FrameState frame_state = 2;
  ScoreInfo score_info = 3;
  optional int32 result_code = 4;
  optional string result_message = 5;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `init_state` | 环境初始化静态信息 |
| `frame_state` | 当前帧数据 |
| `score_info` | 环境得分信息 |
| `result_code` | 错误码 |
| `result_message` | 错误信息 |

当前预处理器会在初始帧读取：

```python
game_info = extra_info.get("init_state", {})
self.init_road_info(game_info)
```

## ScoreInfo

```protobuf
message ScoreInfo {
  float score = 1;
  float avg_junction_delay = 2;
  float avg_junction_queue_length = 3;
  float avg_junction_waiting_time = 4;
  float avg_phase_change_punish_cnt = 5;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `score` | 总得分 |
| `avg_junction_delay` | 交叉路口车辆平均延误 |
| `avg_junction_queue_length` | 交叉路口车辆平均排队长度 |
| `avg_junction_waiting_time` | 交叉路口车辆平均等待时间 |
| `avg_phase_change_punish_cnt` | 信号灯切换惩罚次数 |

`ScoreInfo` 是环境评分，不是训练 reward。训练 reward 由 `reward_shaping()` 自定义。

## Action 与 Command

```protobuf
message Action {
  repeated AICommandInfo cmd_list = 1;
}

message Command {
  repeated AICommandInfo cmd_list = 1;
}

message AICommandInfo {
  uint32 s_id = 1;
  uint32 next_phase_idx = 2;
  uint32 duration = 3;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `s_id` | 信号灯编号 |
| `next_phase_idx` | 下一个相位在相位序列中的索引 |
| `duration` | 下一个相位持续时间，单位 ms |

代码中传入 `env.step()` 的动作是列表：

```text
[junction_id, phase_idx, duration]
```

相关位置：

```text
agent_target_dqn/agent.py::action_process()
```

注意：协议里的 `duration` 单位是毫秒，代码中模型输出的 duration 默认是 `0-19` 的整数。后续需要确认框架是否在底层完成单位转换。

## FrameState

```protobuf
message FrameState {
  uint32 frame_no = 1;
  uint32 frame_time = 2;
  repeated Vehicle vehicles = 3;
  repeated Phase phases = 4;
  repeated Lane lanes = 5;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `frame_no` | 帧号 |
| `frame_time` | 时间，单位 ms |
| `vehicles` | 当前场景内车辆列表 |
| `phases` | 当前信号灯相位信息 |
| `lanes` | 当前车道信息 |

当前代码主要使用 `vehicles` 构造特征和 reward。

## Vehicle

```protobuf
message Vehicle {
  uint32 v_id = 1;
  uint32 v_config_id = 2;
  int32 edge = 3;
  int32 lane = 4;
  int32 junction = 5;
  int32 target_junction = 6;
  PositionInLane position_in_lane = 7;
  int32 speed = 8;
  int32 accel = 9;
  int32 angle = 10;
  Position position = 11;
  int32 absolute_angle = 12;
  int32 v_status = 13;
  float delay = 14;
  float waiting_time = 15;
}
```

常用字段：

| 字段 | 说明 | 当前用途 |
| --- | --- | --- |
| `v_id` | 车辆编号 | 预处理缓存 |
| `v_config_id` | 车辆配置 ID | 查询最大速度 |
| `lane` | 当前车道 | 判断车道和栅格 |
| `junction` | 当前交叉口 | 判断车辆位置 |
| `target_junction` | 目标交叉口 | 筛选当前路口车辆 |
| `position_in_lane` | 车道内位置 | 计算栅格位置 |
| `speed` | 速度，单位 m/s | 速度特征、停车判断 |
| `v_status` | 车辆状态 | 可区分正常、事故、无规则车辆 |
| `delay` | 车辆延误，单位 s | 后续 reward 可用 |
| `waiting_time` | 等待时间，单位 s | 当前 reward 使用 |

`position_in_lane` 包含：

```protobuf
message PositionInLane {
  int32 x = 1;
  int32 y = 2;
}
```

其中 `y` 表示到达停止线的距离，当前代码用它计算车辆所在栅格：

```python
y_pos = int((vehicle["position_in_lane"]["y"] / 1) // Config.GRID_LENGTH)
```

## Phase

```protobuf
message Phase {
  uint32 s_id = 1;
  uint32 phase_id = 2;
  uint32 duration = 3;
  uint32 remaining_duration = 4;
  repeated Light lights = 5;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `s_id` | 当前信号灯编号 |
| `phase_id` | 当前信号灯相位 |
| `duration` | 信号灯持续时间，单位 s |
| `remaining_duration` | 信号灯剩余时间，单位 s |
| `lights` | 信号灯灯色信息 |

后续可用 `phase_id`、`duration` 和 `remaining_duration` 判断相位切换是否过频。

## Light

```protobuf
message Light {
  uint32 green_mask = 1;
  uint32 yellow_mask = 2;
  uint32 red_mask = 3;
  NaturePosition nature_position = 4;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `green_mask` | 绿色灯方向 mask |
| `yellow_mask` | 黄色灯方向 mask |
| `red_mask` | 红色灯方向 mask |
| `nature_position` | 信号灯位置 |

方向 mask 使用 `DirectionMask` 组合。

## Lane

```protobuf
message Lane {
  uint32 lane_id = 1;
  uint32 v_count = 2;
  float congestion = 3;
  float queue_length = 4;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `lane_id` | 当前车道 ID |
| `v_count` | 当前车道车流量 |
| `congestion` | 当前车道拥堵情况 |
| `queue_length` | 当前车道排队长度 |

这些字段适合用于增强 reward 或新增特征。目前基础代码主要使用 `vehicles`，没有直接使用 `lanes`。

## InitState

```protobuf
message InitState {
  repeated Junction junctions = 1;
  repeated Signal signals = 2;
  repeated Edge edges = 3;
  repeated LaneConfig lane_configs = 4;
  repeated VehicleConfig vehicle_configs = 5;
}
```

`InitState` 是环境静态信息，通常在第一帧读取一次。

当前预处理器会保存：

```text
junction_dict
edge_dict
lane_dict
l_id_to_index
vehicle_configs_dict
```

这些字典用于后续特征处理和归一化。

## Junction

```protobuf
message Junction {
  uint32 j_id = 1;
  uint32 signal = 2;
  repeated LanesOnDirection enter_lanes_on_directions = 3;
  repeated LanesOnDirection exit_lanes_on_directions = 4;
  repeated uint32 neighbor_junctions = 5;
  string j_name = 6;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `j_id` | 交叉口 ID |
| `signal` | 关联信号灯 |
| `enter_lanes_on_directions` | 进口车道 |
| `exit_lanes_on_directions` | 出口车道 |
| `neighbor_junctions` | 相邻交叉口 |
| `j_name` | 交叉口名称 |

当前任务是单交叉口，代码中默认：

```python
junction_id = 0
```

## Signal

```protobuf
message Signal {
  uint32 s_id = 1;
  repeated LightPhaseConfig phases = 2;
  uint32 phase_idx = 3;
  uint32 duration = 4;
  uint32 start_time = 5;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `s_id` | 信号灯编号 |
| `phases` | 相位顺序 |
| `phase_idx` | 当前相位在序列中的索引 |
| `duration` | 当前相位设定持续时间，单位 s |
| `start_time` | 当前相位开始时间，单位 s |

## Edge 与 LaneConfig

```protobuf
message Edge {
  uint32 e_id = 1;
  repeated uint32 lanes = 2;
}

message LaneConfig {
  uint32 l_id = 1;
  uint32 edge_id = 2;
  uint32 dir_mask = 3;
  uint32 length = 4;
  uint32 width = 5;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `Edge.e_id` | 道路编号 |
| `Edge.lanes` | 道路包含的车道 ID |
| `LaneConfig.l_id` | 车道 ID |
| `LaneConfig.edge_id` | 所属 edge |
| `LaneConfig.dir_mask` | 允许转向 |
| `LaneConfig.length` | 车道长度，单位 m |
| `LaneConfig.width` | 车道宽度，单位 m |

## VehicleConfig

```protobuf
message VehicleConfig {
  uint32 v_config_id = 1;
  VehicleType v_type = 2;
  uint32 length = 3;
  uint32 max_speed = 4;
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `v_config_id` | 车辆配置 ID |
| `v_type` | 车辆类型 |
| `length` | 车辆长度，单位 m |
| `max_speed` | 车辆最大速度，单位 m/s |

当前代码使用 `max_speed` 对速度特征归一化。

## 枚举

车辆类型：

```protobuf
enum VehicleType {
  Unknown = 0;
  CAR = 1;
  BUS = 2;
  TRUCK = 3;
  MOTORCYCLE = 4;
  BICYCLE = 5;
}
```

方向 mask：

```protobuf
enum DirectionMask {
  None = 0;
  Straight = 1;
  Left = 2;
  Right = 4;
  UTurn = 8;
}
```

## 当前代码使用建议

优先关注这些字段：

| 目标 | 字段 |
| --- | --- |
| 构造位置特征 | `vehicle["lane"]`、`vehicle["position_in_lane"]["y"]` |
| 构造速度特征 | `vehicle["speed"]`、`vehicle["v_config_id"]`、`VehicleConfig.max_speed` |
| 设计等待奖励 | `vehicle["waiting_time"]` |
| 设计排队惩罚 | `vehicle["speed"] <= 0.1`、`Lane.queue_length` |
| 设计延误惩罚 | `vehicle["delay"]` |
| 控制切换频率 | `Phase.phase_id`、`Phase.duration`、`Phase.remaining_duration`、`act` |

第一阶段先使用 `vehicles` 即可。`lanes`、`phases` 和 `score_info` 可以在基础版本跑通后再接入。
