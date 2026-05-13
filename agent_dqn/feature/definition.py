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
from agent_dqn.conf.conf import Config
from agent_dqn.feature.traffic_utils import *


# SampleData with dimensions: define dimensions directly, no need for SampleData2NumpyData/NumpyData2SampleData
# SampleData with dimensions: 直接定义维度，不需要 SampleData2NumpyData/NumpyData2SampleData
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,  # 560
    _obs=Config.DIM_OF_OBSERVATION,  # 560
    act=4,
    # [phase(4 choices)]
    # [相位(4个选择)]
    rew=2,
    # [phase_reward, duration_reward]
    # [相位奖励, 持续时间奖励]
    done=1,
    legal_action=4,
    # phase legal actions
    # 相位合法动作
)

ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls("ActData", junction_id=None, phase_index=None, duration=None)


def sample_process(list_game_data):
    r_data = np.array(list_game_data).squeeze()

    sample_datas = []
    for data in r_data:
        legal_action = [data.legal_action[0], data.legal_action[0], data.legal_action[0], data.legal_action[0]]
        sample_data = SampleData(
            obs=data.obs,
            _obs=None,
            act=data.act,
            rew=data.rew,
            done=1 if data.done == 0 else 0,
            legal_action=legal_action,
        )
        sample_datas.append(sample_data)

    for i in range(len(sample_datas) - 1):
        sample_datas[i]._obs = sample_datas[i + 1].obs
    sample_datas[-1]._obs = sample_datas[-1].obs

    if sample_datas[-1].done:
        del sample_datas[-1]

    return sample_datas


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

    # ========== TODO 9 ==========
    # Improve the reward function design.
    # Hint: Design phase_reward and duration_reward with waiting-time change, best phase matching, and switching penalties.
    # 完善奖励函数设计。
    # 提示：可结合等待时间变化、最佳相位匹配和切换惩罚设计 phase_reward 与 duration_reward。
    
    return 0, 0
