import gym
from agent import SarsaAgent
import time
# from sample_read_data import *  # load data

def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0
    obs = env.reset()  # 重置环境 重新开始一个新的episode
    action = agent.sample(obs)  # agent根据算法选择一个动作输出
    while True:
        next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
        next_action = agent.sample(next_obs)  # 根据算法选择一个动作
        agent.learn(obs, action, reward, next_obs, next_action, done)  # 训练Sarsa

        action = next_action
        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        if render:
            env.render()  # 渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps

# 查看最后效果
def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print("test reward = %.lf" % (total_reward))
            break

def main():
    # gym创建迷宫环境 is_slipper=False 降低环境难度
    env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 up, 1 right, 2 down, 3 left
    # env = gym.make("CliffWalking-v0", is_slippery=False)  # 0 up, 1 right, 2 down, 3 left
    # 创建agent实例
    agent = SarsaAgent(obs_n=env.observation_space.n,
                       act_n=env.action_space.n,
                       learning_rate=0.01,
                       gamma=0.9,
                       e_greed=0.1)
    is_render = False
    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print("Epsiode %s: steps = %s , reward = %.lf " % (episode, ep_steps, ep_reward))
        if episode%20 == 0:  # 每隔20个episode 渲染 看目前效果
            is_render = True
        else:
            is_render = False
    # 训练结束 查看算法效果
    test_episode(env, agent)

if __name__ == "__main__":
    main()
