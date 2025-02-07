import gymnasium as gym
import pickle as pkl 
import matplotlib.pyplot as plt
import torch  

#CHECKINNG FOR GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_info= torch.cuda.get_device_name(0)
    print(f"GPU: {device_info}")
    
    # Get CUDA compute capability (affects CUDA core count)
    capability = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {capability}")
else:
    print("CUDA not available")

# Action: 0=left, 1=down, 2=right, 3=up
# Reward: 1 if reaches the gift or step*0.001 for anything else

def run(episodes, is_training=True, render=False, is_slippery=False):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=is_slippery, render_mode='human' if render else None)

    # Initialize Q-table on GPU

    if is_training: 
        q=torch.zeros((env.observation_space.n, env.action_space.n), device=device) #This will init a 64x4 array
    else: 
        with open('frozen_lake8x8.pkl', 'rb') as f:
            q = torch.tensor(pkl.load(f), device=device) #Loading Q-Table to GPU
            print(q)
        

    learning_rate_a = 0.9           # alpha or learning rate 
    discount_factor_g = 0.9         # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state
    epsilon = 1                      # 1 = 100% random action
    epsilon_decay_rate = 0.0001     # epsilon decay rate. 1/0.0001 = 10,000. i:e. it will take 10,000 steps to reduce epsilon to 0

    rewards_per_episode = torch.zeros((episodes), device=device) # We generate an array of 0's of size episodes

    for i in range(episodes): 
        state = env.reset()[0]  #resets the env, and puts the agent on position 0.(top left corner)
        terminated = False      #True when the agent falls in hole or reaches the goal 
        truncated = False       #True when actions > 200 
        if is_training: 
            print(f"{i} epsisodes out {episodes} done")

        while(not terminated and not truncated): 
            # Implementing - Choose action (vectorized epsilon-greedy)
            if is_training and torch.rand(1, device=device) < epsilon: 
                action = torch.tensor(env.action_space.sample(), device=device).item() #randomly selects an action from random space
            else: 
                action = torch.argmax(q[state,:]).item()
            
            new_state, reward, terminated, truncated, _ = env.step(action=action)

            if is_training:

                #Below is bellman's equation, used to update values in Q table
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * torch.max(q[new_state, :]).item() - q[state,action]
                )

            state = new_state
        epsilon = max(epsilon - epsilon_decay_rate, 0) 

        if(epsilon == 0): 
            learning_rate_a = 0.0001    # We do this to minimize the effect of further training after 
                                        #the epsilon has reached 0 and, to prevent overfitting
        
        rewards_per_episode[i] = reward if reward == 1 else i*0.001
    
    env.close()

    sum_rewards = torch.zeros((episodes), device=device)
    for timestamp in range(episodes): 
        sum_rewards[timestamp] = torch.sum(rewards_per_episode[max(0, timestamp-100):(timestamp+1)]) #We do rolling avg, to smoothen out the results, and have better idea about change in rewards
    plt.plot(sum_rewards.cpu().numpy()) #Converting tensor's dtype to Numpy Array
    plt.savefig('frozen_lake8x8.png')

    if is_training: 
        f = open("frozen_lake8x8.pkl", "wb")
        pkl.dump(q,f)
        f.close()

if __name__ == "__main__": 
    run(episodes=1500, is_training=True, render=False, is_slippery=True)