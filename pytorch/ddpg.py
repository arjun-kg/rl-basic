import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np 
from matplotlib import pyplot as plt 
import gym 
from copy import deepcopy
import pdb 
import time

#environment name 
env_name = 'Pendulum-v0'

#random seed
np.random.seed(1)

#train or test
train_flag = True
models_path = './models/ddpg_actor.pkl'

#hyperparam search mode
hyperparam_search_mode = True

#hyperparameters
max_timesteps = 10000
num_steps = 1000
train_batch_size = 64
noise_factor = 0.01 #gaussian noise added to output of actor network 
eps_start = 0.5
T = 100000 #Temperature for epsilon decay
gamma = 0.99 
tau = 0.001 #target nets are updated by: theta_target <- tau*theta_current + (1-tau)*theta_target
hidden_layer_size = 256 #all hidden layer sizes in all networks are the same
lr_actor = 0.0001 #actor optimizer's learning rate
lr_critic = 0.001 #critic optimizer's learning rate

class ReplayBuffer:
    def __init__(self,buffer_size=1000000):
        self.buffer_size = buffer_size
        self.buffer = []
    
    def push(self,transition_tuple):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop()
        self.buffer.insert(0,transition_tuple)
    
    def sample(self,n):
        indices = np.random.choice(len(self.buffer),n)
        return np.array(self.buffer)[indices]

class DDPG_Actor(nn.Module):
    def __init__(self,input_size,output_size):
        super(DDPG_Actor,self).__init__()
        self.lin1 = nn.Linear(input_size,hidden_layer_size)
        self.lin2 = nn.Linear(hidden_layer_size,output_size)
        
    def forward(self,x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

class DDPG_Critic(nn.Module):
    def __init__(self,input_size, output_size):
        super(DDPG_Critic,self).__init__()
        self.lin1 = nn.Linear(input_size,hidden_layer_size)
        self.lin2 = nn.Linear(hidden_layer_size,output_size)
        
    def forward(self,x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    
def train_step(actor_net,critic_net,actor_target_net,critic_target_net,
                       memory,optimizers,tau):
    batch = memory.sample(train_batch_size)
    states = torch.tensor([batch[i][0] for i in range(len(batch))]).float()
    actions = torch.tensor([batch[i][1] for i in range(len(batch))]).float()
    rewards = torch.tensor([batch[i][2] for i in range(len(batch))])
    next_states = torch.tensor([batch[i][3] for i in range(len(batch))]).float()
    dones = torch.tensor([batch[i][4] for i in range(len(batch))]).float()

    #Updating critic net    
    next_state_actions = actor_target_net(next_states)
    critic_target_net_input = torch.cat([next_states,next_state_actions],dim=-1)
    critic_target = rewards.view(-1,1) + gamma*critic_target_net(critic_target_net_input)
    
    critic_net_inputs = torch.cat([states,actions],dim=-1)
    critic_values = critic_net(critic_net_inputs)
    
    critic_loss = F.mse_loss(critic_values,critic_target.detach())
    optimizers[1].zero_grad()
    critic_loss.backward()
    
    #clipping gradients     
    for param in critic_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizers[1].step()

    #Updating actor net
    actions_from_actor = actor_net(states)
    critic_net_inputs2 = torch.cat([states,actions_from_actor],dim=-1)
    actor_loss = - critic_net(critic_net_inputs2).mean()
    optimizers[0].zero_grad()
    actor_loss.backward()
    
    #clipping gradients     
    for param in actor_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizers[0].step()
    
    #updating target nets
    for par,t_par in zip(actor_net.parameters(),actor_target_net.parameters()):
        t_par.data.copy_(tau*par.data+ (1-tau)*t_par.data)

    for par,t_par in zip(critic_net.parameters(),critic_target_net.parameters()):
        t_par.data.copy_(tau*par.data+ (1-tau)*t_par.data)


def add_noise(greedy_action,t,action_max,T,noise_factor,eps_start):
    #Adding Gaussian noise
    noise = noise_factor*action_max*np.random.rand(greedy_action.shape[-1])
    action = greedy_action + noise

    if hyperparam_search_mode:
        if t > 8000:
            return action 
    
    #Epsilon-Greedy
    rnd_num = np.random.rand()
    eps_now = eps_start*np.exp(-t/T)
    
    if eps_now < rnd_num:
        return action
    else:
        return np.random.uniform(low=-action_max,high=action_max
                                 ,size=action.shape)

def train_ddpg(train_batch_size,noise_factor, eps_start,T,gamma,tau,hidden_layer_size,lr_actor,lr_critic):

    start_time = time.time()
    #environment
    env = gym.make(env_name)
    
    #initializing class instances
    actor_net = DDPG_Actor(env.observation_space.shape[0],env.action_space.shape[0])
    critic_net = DDPG_Critic(env.observation_space.shape[0]+env.action_space.shape[0],1)
    
    actor_target_net = deepcopy(actor_net)
    actor_target_net.load_state_dict(actor_net.state_dict())
    actor_target_net.eval()
    
    critic_target_net = deepcopy(critic_net)
    critic_target_net.load_state_dict(critic_net.state_dict())
    critic_target_net.eval()
    
    memory = ReplayBuffer()
    optimizers = [optim.Adam(actor_net.parameters(),lr=lr_actor),
                  optim.Adam(critic_net.parameters(),lr=lr_critic)]
    
    total_timesteps = 0
    rew_list = []

    while total_timesteps < max_timesteps:    
        state = env.reset()
        ep_reward = 0
        for st in range(num_steps):
            greedy_action = actor_net(torch.tensor(state).float()).detach().numpy()
            action = add_noise(greedy_action,total_timesteps,
                               env.action_space.high,T,noise_factor,eps_start)
            
            next_state,reward,done,_ = env.step(action)
            transition_tuple = (state,action,reward,next_state,done)
            memory.push(transition_tuple)
            if train_flag:
                train_step(actor_net,critic_net,actor_target_net,critic_target_net,
                           memory,optimizers,tau)
            total_timesteps += 1
            ep_reward += reward
            # env.render()
            if done or st == num_steps-1:
                rew_list.append(ep_reward)
                break
    end_time = time.time()

    if hyperparam_search_mode:
        last_ten_ep_avg = np.mean(rew_list[-10:])

        filename = 'hyperparam_search/{}'.format(last_ten_ep_avg)
        file = open(filename,'w')
        file.write("Script ran for {} seconds\n\n".format(end_time-start_time))
        file.write("Train Batch Size: {}\n".format(train_batch_size))
        file.write("Noise Factor: {}\n".format(noise_factor))
        file.write("Eps Start: {}\n".format(eps_start))
        file.write("T: {}\n".format(T))
        file.write("Gamma: {}\n".format(gamma))
        file.write("Tau: {}\n".format(tau))
        file.write("Hidden Layer Size: {}\n".format(hidden_layer_size))
        file.write("Learning Rate - Actor: {}\n".format(lr_actor))
        file.write("Learning Rate -  Critic: {}\n".format(lr_critic))
        file.write("\n\nRewards:\n\n")
        for r in rew_list:
            file.write('{}\n'.format(r))
        file.close()

        print("DONE: {} {} {} {} {} {} {} {} {}".format(train_batch_size,noise_factor, eps_start,T,gamma,tau,hidden_layer_size,lr_actor,lr_critic))

if __name__ == "__main__":
    train_ddpg(train_batch_size,noise_factor, eps_start,T,gamma,tau,hidden_layer_size,lr_actor,lr_critic)