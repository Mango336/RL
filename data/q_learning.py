import gym
from agent import Q_learningAgent
# from gridworld import CliffWalkingWapper
import time

def run_episode(env, agent, render=False):
    total_steps = 0
    total_reward = 0
    obs = env.reset()  # 环境重置

    while True:
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        # 训练Q-learning算法
        agent.learn(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
        total_steps += 1
        if render:
            env.render()
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
    env = gym.make("CliffWalking-v0")
    # env = CliffWalkingWapper(env)
    agent = Q_learningAgent(obs_n=env.observation_space.n,
                            act_n=env.action_space.n,
                            learning_rate=0.1,
                            gamma=0.9,
                            e_greed=0.1)
    is_render = False
    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.lf' % (episode, ep_steps, ep_reward))
        if episode%20 == 0:
            is_render = True
        else:
            is_render = False
    test_episode(env, agent)
    
if __name__ == "__main__":
    main()
