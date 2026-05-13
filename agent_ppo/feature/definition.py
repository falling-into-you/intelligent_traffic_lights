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
    junction_id = 0
    phase_reward, duration_reward = 0, 0

    frame_state = _obs["frame_state"]
    vehicles = frame_state["vehicles"]

    # ========== TODO 15 ==========
    # Improve the reward function design.
    # Hint: Build the reward with waiting-time change, phase matching, traffic efficiency, and switching penalties.
    # 完善奖励函数设计。
    # 提示：可结合等待时间变化、相位匹配、通行效率和切换惩罚构造奖励。

    return 0


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
