"""
This is the definition and configurations of the traffic lights structure in the simulator
"""

import os
import sys

import torch

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
from gym import spaces


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    Default observation space is a vector R^(#greenPhases + 2 * #lanes)
    s = [current phase one-hot encoded, density for each lane, queue for each lane]
    You can change this by modifing self.observation_space and the method _compute_observations()

    Action space is which green phase is going to be open for the next delta_time seconds
    """

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green, begin_time, sumo=traci):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.sumo = sumo

        self.build_phases()


    def build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if 'y' not in state and (state.count('r') + state.count('s') != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()
        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j: continue
                yellow_state = ''
                for s in range(len(p1.state)):
                    if (p1.state[s] == 'G' or p1.state[s] == 'g') and (p2.state[s] == 'r' or p2.state[s] == 's'):
                        yellow_state += 'y'
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)  # 绿灯相位i,变换到相位j的过程中对应的黄灯相位在all_phases中的索引
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        # print(self.green_phases)
        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0  # 将程序控制器设置为固定时长控制
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    def get_phase_onehot(self):  # 得到当前相位的onehot编码
        return [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]

    def set_next_phase(self, new_phase, step_length, env):
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.min_green:  # 假设相位并没有变化，或者上次等改变的时间<黄灯时间+最小绿灯时间， 保持相位不变
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state)
            self.green_phase = new_phase
            self.is_yellow = True
            self.time_since_last_phase_change = 0


    def update(self, step_length):
        self.time_since_last_phase_change += step_length
        # 保留两位小数
        self.time_since_last_phase_change = round(self.time_since_last_phase_change, 2)
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            #self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False
            self.time_since_last_phase_change = 0



