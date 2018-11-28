import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import gym 
import numpy as np 
from copy import deepcopy
import pdb 
from matplotlib import pyplot as plt 

#random seed
np.random.seed(1)

#environment name 
env_name = 'MountainCar-v0'

#hyperparameters
num_episodes = 1000
num_steps = 500
eps_init = 0.5
T = 10000 #temperature for epsilon-decay 
train_batch_size = 50
gamma = 0.9
target_update_freq = 10
    
class ReplayBuffer:
    def __init__(self,buffer_size=10000):
        self.buffer_size = buffer_size
        self.buffer = []

    def push(self,transition_tuple):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop()
        self.buffer.insert(0,transition_tuple)

    def sample(self,n):
        indices = np.random.choice(len(self.buffer),n)
        return np.array(self.buffer)[indices]

class DQN(nn.Module):
    def __init__(self,input_size, output_size):
        super(DQN,self).__init__()
        
        self.lin1 = nn.Linear(input_size,256)
        self.lin2 = nn.Linear(256,output_size)
        
    def forward(self,x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x

def eps_greedy(greedy_action,num_actions,t):
    rnd_num = np.random.rand()
    current_eps = eps_init*np.exp(-t/T) 
    if t % num_steps == 0:
        print("Current Epsilon: ", current_eps)
    if rnd_num > current_eps:
        return greedy_action
    else: 
        return np.random.choice(num_actions)
    
def plotter(r):
    plt.figure(0)
    plt.clf()
    plt.plot(r)
    plt.show()

def train_step(policy_net,target_net,memory,optimizer):
    if train_batch_size > len(memory.buffer):
        return
    batch = memory.sample(train_batch_size)
    states = torch.tensor([batch[i][0] for i in range(len(batch))]).float()
    actions = torch.tensor([batch[i][1] for i in range(len(batch))]).view(-1,1)
    rewards = torch.tensor([batch[i][2] for i in range(len(batch))])
    next_states = torch.tensor([batch[i][3] for i in range(len(batch))]).float()
    dones = torch.tensor([batch[i][4] for i in range(len(batch))]).float()
        
    state_values = policy_net(states).gather(1,actions).view(-1)
    max_next_state_values = target_net(next_states).max(1)[0].detach()
    expected_state_values = rewards + gamma*max_next_state_values*(1-dones)
    
    #loss
    loss = F.smooth_l1_loss(state_values,expected_state_values)
    
    optimizer.zero_grad()
    loss.backward()
    
    #clipping gradients
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
if __name__ == '__main__':
    #use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
 
    #environment
    env = gym.make(env_name)
    env.reset()
    total_timesteps = 0 
    episode_rewards = []
    
    #initializing useful class instances 
    memory = ReplayBuffer()
    policy_net = DQN(env.observation_space.shape[0],env.action_space.n).to(device)
    target_net = deepcopy(policy_net)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters())
    
    for ep in range(num_episodes):
        
        state = env.reset()
        episode_rewards.append(0)
        for st in range(num_steps):
            greedy_action = policy_net(torch.tensor(state).float()).max(0)[1].item()
            action = eps_greedy(greedy_action,env.action_space.n,total_timesteps)
            next_state,reward,done,_ = env.step(action)
            transition_tuple = (state,action,reward,next_state,done)
            memory.push(transition_tuple)
            train_step(policy_net,target_net,memory,optimizer)
            state = next_state
            episode_rewards[ep] += reward
            total_timesteps += 1
            if total_timesteps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if done:
                print("Steps in episode {}: {}".format(ep,st))
                break
#            env.render()
