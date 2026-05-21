#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.traffic_utils import *


ObsData = create_cls("ObsData", feature=None, legal_action=None, sub_action_mask=None)

ActData = create_cls("ActData", junction_id=None, action=None, d_action=None, prob=None, value=None)

# SampleData with dimensions: define dimensions directly, no need for SampleData2NumpyData/NumpyData2SampleData
# SampleData with dimensions: 直接定义维度，不需要 SampleData2NumpyData/NumpyData2SampleData
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,  # 560
    legal_action=Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1,  # 8
    act=Config.NUMB_HEAD,  # 2
    reward=1,
    reward_sum=1,
    done=1,
    value=1,
    next_value=1,
    advantage=1,
    prob=Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1,  # 8
    sub_action=Config.NUMB_HEAD,  # 2
    is_train=1,
)


def sample_process(list_sample_data):
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_reward(list_sample_data)

    return list_sample_data


def reward_shaping(_obs, act, agent):
    """
    This function is an important function for reward processing, mainly responsible for:
        - Unpacking data, obtaining the data required for reward calculation from _obs
        - Reward calculation, calculating rewards based on the unpacked data
        - Reward concatenation, concatenating all rewards into a list

    Parameters:
        - _obs: The original feature data sent by battlesrv
        - act: The previous act predicted and executed
        - agent: real agent perform action

    Returns:
        - phase reward: The reward corresponding to the action of the phase number
        - duration reward: The reward corresponding to the action of the phase duration
    """
    """
    该函数是奖励处理的重要函数, 主要负责：
        - 数据解包, 从 _obs 获取计算奖励所需要的数据
        - 奖励计算, 根据解包的数据计算奖励
        - 奖励拼接, 将所有的奖励拼接成一个list

    参数：
        - _obs: battlesrv 发送的原始特征数据
        - act: 前一次预测并执行动作
        - agent: 实际执行动作智能体

    返回：
        - phase reward: 对应相位编号动作的奖励
        - duration reward: 对应相位持续时间动作的奖励
    """
    junction_id = 0  # 单路口场景里，路口 id 固定为 0

    frame_state = _obs["frame_state"]
    vehicles = frame_state["vehicles"]

    # ========== TODO 15 ==========
    # Improve the reward function design.
    # Hint: Build the reward with waiting-time change, phase matching, traffic efficiency, and switching penalties.
    # 完善奖励函数设计。
    # 提示：可结合等待时间变化、相位匹配、通行效率和切换惩罚构造奖励。

    enter_vehicle_count = 0  # 当前进口车道上的车辆数量
    stopped_vehicle_count = 0  # 当前进口车道上处于等待状态的车辆数量
    total_delay = 0.0  # 当前所有进口车辆延误时间之和
    total_waiting_time = 0.0  # 当前所有进口车辆等待时间之和

    for vehicle in vehicles:
        if not on_enter_lane(vehicle):  # 如果车辆不在进口车道上，则不参与奖励计算
            continue
        if vehicle["target_junction"] != junction_id:
            continue

        enter_vehicle_count += 1

        if vehicle["speed"] <= 0.1:
            stopped_vehicle_count += 1

        # delay 是环境给出的车辆延误时间，单位是秒。
        total_delay += vehicle.get("delay", 0.0)

        # waiting_time 是环境给出的车辆等待时间
        total_waiting_time += vehicle.get("waiting_time", 0.0)

    # 没有进口车辆时，不给奖励也不给惩罚
    if enter_vehicle_count == 0:
        return 0.0

    avg_delay = total_delay / enter_vehicle_count
    avg_waiting_time = total_waiting_time / enter_vehicle_count
    queue_length = stopped_vehicle_count

    # 对齐环境最终得分的三个正向指标：
    # 1. 平均延误越低，delay_score 越接近 1。
    # 2. 排队车辆越少，queue_score 越接近 1。
    # 3. 平均等待时间越低，waiting_score 越接近 1。
    delay_score = 1.0 / (1.0 + avg_delay / 9.0)
    queue_score = 1.0 / (1.0 + queue_length / 10.0)
    waiting_score = 1.0 / (1.0 + avg_waiting_time / 8.0)

    # 环境会惩罚过于频繁的信号切换。
    # 评估规则中，连续两个绿灯相位间隔小于 8 秒会增加切换惩罚。
    # PPO 的 act 是 ActData，d_action = [phase_idx, duration]。
    short_duration_penalty = 0.0
    if act is not None and act.d_action is not None:
        duration = int(act.d_action[1])
        if duration < 8:
            short_duration_penalty = 1.5 * (8 - duration) / 8.0

    reward = delay_score + queue_score + waiting_score - short_duration_penalty

    return reward


def _calc_reward(list_sample_data):
    """
    Calculate cumulated reward and advantage with GAE.
    reward_sum: used for value loss
    advantage: used for policy loss
    V(s) here is a approximation of target network

    使用 GAE 计算累积奖励和优势函数。
    reward_sum: 用于价值损失
    advantage: 用于策略损失
    V(s) 这里是目标网络的近似值
    """

    gae, last_gae = 0.0, 0.0
    gamma, lamda = Config.GAMMA, Config.LAMDA
    for rl_info in reversed(list_sample_data):
        delta = -rl_info.value + rl_info.reward + gamma * rl_info.next_value
        gae = gae * gamma * lamda + delta
        rl_info.advantage = gae
        rl_info.reward_sum = gae + rl_info.value
