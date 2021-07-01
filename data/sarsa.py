import numpy as np
import gym
from gridworld import CliffWakingWapper
from sample_read_data import *


class SarsaAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n  # 动作维度 有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值 采样输出的动作值
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作  uniform(0, 1)=>[0, 1]的随机数
            action = self.predict(obs)  # 拿到最优动作
        else:
            action = np.random.choice(self.act_n)  # 其他动作也有一定概率随机探索到
        return action

    # 根据输入观察值 预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]  # Q_list中 行代表状态 列代表动作
        maxQ = np.max(Q_list)  # 贪心策略
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)  # 从这多个action中随机选取一个
        return action

    def learn(self, obs, action, reward, next_obs, next_action, done):
        """
        obs: 交互前的obs s_t
        action: 本次交互选择的action a_t
        reward: 本次动作获得的奖励 r
        next_obs: 本次交互后的obs s_t+1
        next_action: 根据当前Q表格 针对next_obs会选择的动作 即：a_t+1
        done: episode是否介绍
        """
        predict_Q = self.Q[obs, action]  # 预测到的下一个Q值
        if done:  # 最后一轮迭代  没有下一个状态
            target_Q = reward
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]  # Sarsa
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)

