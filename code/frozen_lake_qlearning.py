import gymnasium as gym
import random 
import pickle as pkl 
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np 


# Action: 0=left, 1=down, 2=right, 3=up

def run(episodes, is_training=True, render=False, is_slippery=False):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=is_slippery, render_mode='human' if render else None)

    if is_training: 
        q=np.zeros((env.observation_space.n, env.action_space.n)) #This will init a 64x4 array
    else: 
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pkl.load(f)
        f.close()

    learning_rate_a = 0.9           # alpha or learning rate 
    discount_factor_g = 0.9         # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state
    epsilon = 1                      # 1 = 100% random action
    epsilon_decay_rate = 0.0001     # epsilon decay rate. 1/0.0001 = 10,000. i:e. it will take 10,000 steps to reduce epsilon to 0
    rng = random.random()

    rewards_per_episode = np.zeros(episodes) # We generate an array of 0's of size episodes

    for i in range(episodes): 
        state = env.reset()[0]  #resets the env, and puts the agent on position 0.(top left corner)
        terminated = False      #True when the agent falls in hole or reaches the goal 
        truncated = False       #True when actions > 200 
        print(f"{i} epsisodes out {episodes} done")
        while(not terminated and not truncated): 
            if is_training and rng < epsilon: 
                action = env.action_space.sample() #randomly selects an action from random space
            else: 
                action = np.argmax(q[state,:])
            
            new_state, reward, terminated, truncated, _ = env.step(action=action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state,action]
                )

            state = new_state
        epsilon = max(epsilon - epsilon_decay_rate, 0) 

        if(epsilon == 0): 
            learning_rate_a = 0.0001    # We do this to minimize the effect of further training after 
                                        #the epsilon has reached 0 and, to prevent overfitting
        

        if(reward == 1): 
            rewards_per_episode[i] = 1 
    
    env.close()

    sum_rewards = np.zeros(episodes)
    for timestamp in range(episodes): 
        sum_rewards[timestamp] = np.sum(rewards_per_episode[max(0, timestamp-100):(timestamp+1)]) #We do rolling avg, to smoothen out the results, and have better idea about change in rewards
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if is_training: 
        f = open("frozen_lake8x8.pkl", "wb")
        pkl.dump(q,f)
        f.close()

if __name__ == "__main__": 
    run(episodes=1000, is_training=True, render=True, is_slippery=False)