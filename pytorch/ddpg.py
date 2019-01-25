import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np 
from matplotlib import pyplot as plt 
from collections import namedtuple
import gym 
import random
from copy import deepcopy
import pdb 
import time
import tensorflow as tf 

from parallelize_env_wrapper import ParallelizeEnv

torch.set_default_tensor_type(torch.DoubleTensor)

#environment name 
env_name = 'Pendulum-v0'

#random seed
np.random.seed(1)

#named tuple for experience replay
Transition = namedtuple('Transition',
                            ('state', 'action', 'reward','next_state','done'))

#train or test
train_flag = True
models_path = './models/ddpg_actor.pkl'

#hyperparam search mode
hyperparam_search_mode = False

#hyperparameters
max_timesteps = 1000000
num_steps = 1000
train_batch_size = 128
noise_factor = 0.001 #gaussian noise added to output of actor network 
eps_start = 0.8
T = 200000 #Temperature for epsilon decay
gamma = 0.99 
tau = 0.001 #target nets are updated by: theta_target <- tau*theta_current + (1-tau)*theta_target
hidden_layer_size = 256 #all hidden layer sizes in all networks are the same
lr_actor = 1e-4 #actor optimizer's learning rate
lr_critic = 1e-3 #critic optimizer's learning rate

class ReplayBuffer:
    def __init__(self,buffer_size=1000000):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0
    
    def push(self,s,a,r,ns,d):

        transitions_list = [Transition(a,b,c,d,e) for a,b,c,d,e in zip(s,a,r,ns,d)]
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        list_len = len(transitions_list)

        if list_len + self.position > self.buffer_size:
            self.buffer[self.position:self.buffer_size] = transitions_list[:self.buffer_size-self.position]
            self.buffer[:list_len-self.buffer_size+self.position] = transitions_list[list_len-self.buffer_size+self.position:]
        else:
            self.buffer[self.position:self.position+list_len] = transitions_list
        self.position = (self.position + list_len)%self.buffer_size
    
    def sample(self,n):
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)

class DDPG_Actor(nn.Module):
    def __init__(self,input_size,output_size,action_space_high):
        super(DDPG_Actor,self).__init__()

        self.action_space_high = action_space_high
        self.lin1 = nn.Linear(input_size,hidden_layer_size)
        self.lin2 = nn.Linear(hidden_layer_size,hidden_layer_size)
        self.lin3 = nn.Linear(hidden_layer_size,hidden_layer_size)
        self.lin4 = nn.Linear(hidden_layer_size,output_size)
        self.lin4.weight.data.uniform_(-1e-2,1e-2)

        self.ln1 = nn.LayerNorm(hidden_layer_size)
        self.ln2 = nn.LayerNorm(hidden_layer_size)
        self.ln3 = nn.LayerNorm(hidden_layer_size)
        
    def forward(self,x):
        x = F.relu(self.ln1(self.lin1(x)))
        x = F.relu(self.ln2(self.lin2(x)))
        x = F.relu(self.ln3(self.lin3(x)))
        x = torch.tanh(self.lin4(x))*torch.tensor(self.action_space_high).double()
        return x

class DDPG_Critic(nn.Module):
    def __init__(self,state_size,action_size, output_size):
        super(DDPG_Critic,self).__init__()
        self.lin1 = nn.Linear(state_size,hidden_layer_size)
        self.lin2 = nn.Linear(hidden_layer_size + action_size,hidden_layer_size)
        self.lin3 = nn.Linear(hidden_layer_size,hidden_layer_size)
        self.lin4 = nn.Linear(hidden_layer_size,output_size)
        self.lin4.weight.data.uniform_(-1e-2,1e-2)

        self.ln1 = nn.LayerNorm(hidden_layer_size)
        self.ln2 = nn.LayerNorm(hidden_layer_size)
        self.ln3 = nn.LayerNorm(hidden_layer_size)
        
    def forward(self,state,action):
        x = F.relu(self.ln1(self.lin1(state)))
        x = torch.cat((x,action),dim=-1)
        x = F.relu(self.ln2(self.lin2(x)))
        x = F.relu(self.ln3(self.lin3(x)))
        x = self.lin4(x)
        return x

class StateNormalizer: 
    def __init__(self,size):
        self.mean = np.zeros(size)
        self.M2 = np.zeros(size)
        self.count = 0
        self.std = np.ones(size)

    def normalize(self,x,skip=False):
        if skip:
            return x
        return (x - self.mean)/(self.std+1e-4) #1e-4 is to avoid blow up

    def update(self,x): #uses Welford's online algorithm to calculate variance
        self.count += 1
        delta = x - self.mean
        self.mean += delta/self.count
        delta2 = x - self.mean
        self.M2 += delta*delta2
        self.std = np.sqrt(self.M2/np.maximum((self.count-1),1e-3))


class ParameterSpaceNoise:
    def __init__(self,init_std=0.1, des_action_dev = 0.1, adapt_coeff = 1.01):
        self.current_std = init_std
        self.des_action_dev = des_action_dev
        self.adapt_coeff = adapt_coeff
    
    def adapt(self,action,noisy_action):
        # pdb.set_trace()
        dist = np.linalg.norm(action-noisy_action,axis=-1) #axis = -1 in case there are batched actions in the future

        if dist < self.des_action_dev:
            self.current_std *= self.adapt_coeff
        else:
            self.current_std /= self.adapt_coeff

    def apply(self,source_policy,noisy_policy):
        keys = source_policy.state_dict().keys()
        for key in keys:
            # pdb.set_trace()   
            if 'ln' in key: #ignoring parameters of layer normalization
                continue
            par = source_policy.state_dict()[key]
            n_par = noisy_policy.state_dict()[key]

            new_n_par = par+torch.randn(par.shape) * self.current_std
            n_par.data.copy_(new_n_par)


def train_step(actor_net,critic_net,actor_target_net,critic_target_net,
                       memory,optimizers,tau,train_batch_size,total_timesteps,logger):

    transitions = memory.sample(train_batch_size)
    batch = Transition(*zip(*transitions))
    states = torch.tensor(batch.state)
    actions = torch.tensor(batch.action)
    rewards = torch.tensor(batch.reward)
    next_states = torch.tensor(batch.next_state)
    dones = torch.tensor(batch.done).double()

    #Updating critic net    
    next_state_actions = actor_target_net(next_states)
    critic_target = rewards.unsqueeze(1) + gamma*(1-dones).unsqueeze(1)*critic_target_net(next_states,next_state_actions)
    
    critic_values = critic_net(states,actions)
    
    critic_loss = F.smooth_l1_loss(critic_values,critic_target.detach())
    optimizers[1].zero_grad()
    critic_loss.backward()
    
    #clipping gradients     
    for param in critic_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizers[1].step()

    #Updating actor net
    actions_from_actor = actor_net(states)
    actor_loss = - critic_net(states,actions_from_actor).mean()
    logger.record('Actor Loss',actor_loss,total_timesteps)
    optimizers[0].zero_grad()
    actor_loss.backward()
    
    #clipping gradients     
    for param in actor_net.parameters():
        param.grad.data.clamp_(-1, 1)
        logger.record('Actor Net Gradient {}'.format(param.name),param.grad.data)
    optimizers[0].step()
    
    #updating target nets
    for par,t_par in zip(actor_net.parameters(),actor_target_net.parameters()):
        t_par.data.copy_(tau*par.data+ (1-tau)*t_par.data)

    for par,t_par in zip(critic_net.parameters(),critic_target_net.parameters()):
        t_par.data.copy_(tau*par.data+ (1-tau)*t_par.data)


def add_noise(greedy_action,t,action_max,T,noise_factor,eps_start):
    #Adding Gaussian noise
    # noise = noise_factor*action_max*np.random.rand(greedy_action.shape[-1])
    noise = 0
    action = greedy_action + noise
   
    #Epsilon-Greedy
    rnd_num = np.random.rand()
    eps_now = eps_start*np.exp(-t/T)

    if t % 200 == 0:
        print("Epsilon Now: ",eps_now)
    
    if eps_now < rnd_num:
        return action
    else:
        return np.random.uniform(low=-action_max,high=action_max
                                 ,size=action.shape)
class EpsilonGreedy:
    def __init__(self,eps_start,T,timesteps=0):
        self.eps_start = eps_start
        self.T = T
        self.eps_now = self.eps_start
    def get_action(self,action,t,action_max):
        rnd_num = np.random.rand()
        self.eps_now = self.eps_start*np.exp(-t/self.T)
        if self.eps_now < rnd_num:
            return action
        else:
            return np.random.uniform(low=-action_max,high=action_max
                                 ,size=action.shape)

def evaluate_policy(actor_net,normalizer,env):
    for _ in range(3):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            env.render()
            norm_state = normalizer.normalize(state)
            action = actor_net(torch.tensor(norm_state)).detach().numpy()
            # print(action)
            
            next_state,r,done,_ = env.step(action)
            print("state: ", state,  "Action: ",action, "Reward: ", r)
            ep_reward += r
            state = next_state
        print("Ep. rew: {}".format(ep_reward))

class Logger:
    def __init__(self):
        self.writer = tf.summary.FileWriter('./tb_logdir')
    def record(self,tag,value,step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,simple_value=value)])
        self.writer.add_summary(summary,step)
        self.writer.flush()


def train_ddpg(train_batch_size,noise_factor, eps_start,T,gamma,tau,hidden_layer_size,lr_actor,lr_critic):

    start_time = time.time()

    #logger
    logger = Logger()

    #environment
    env = gym.make(env_name)
    action_space_high = env.action_space.high
    
    #initializing class instances
    actor_net = DDPG_Actor(env.observation_space.shape[0],env.action_space.shape[0],action_space_high)
    noisy_actor_net = deepcopy(actor_net)
    critic_net = DDPG_Critic(env.observation_space.shape[0],env.action_space.shape[0],1)
    
    actor_target_net = deepcopy(actor_net)
    actor_target_net.load_state_dict(actor_net.state_dict())
    actor_target_net.eval()
    
    critic_target_net = deepcopy(critic_net)
    critic_target_net.load_state_dict(critic_net.state_dict())
    critic_target_net.eval()
    
    memory = ReplayBuffer()
    optimizers = [optim.Adam(actor_net.parameters(),lr=lr_actor),
                  optim.Adam(critic_net.parameters(),lr=lr_critic)]
    
    normalizer = StateNormalizer(env.observation_space.shape[0])
    param_space_noise = ParameterSpaceNoise()
    eps_greedy = EpsilonGreedy(eps_start,T)
    total_timesteps = 0
    total_episodes = 0
    rew_list = []

    while total_timesteps < max_timesteps:    

        if total_episodes % 25 == 0 and total_episodes != 0:
            print("Episode: {}, Timesteps: {}".format(total_episodes,total_timesteps))
            evaluate_policy(actor_net,normalizer,env)


        state = env.reset()
        ep_reward = 0

        for st in range(num_steps):
            
            normalizer.update(state)
            norm_state = normalizer.normalize(state)
            
            greedy_action = actor_net(torch.tensor(norm_state)).detach().numpy() 
            noisy_greedy_action = noisy_actor_net(torch.tensor(norm_state)).detach().numpy() 
            param_space_noise.adapt(greedy_action , noisy_greedy_action)
            param_space_noise.apply(actor_net,noisy_actor_net)

            action = noisy_greedy_action
            # action = add_noise(noisy_greedy_action,total_timesteps,
            #                    env.action_space.high,T,noise_factor,eps_start)
            action = eps_greedy.get_action(action,total_timesteps,env.action_space.high)
            next_state,reward,done,_ = env.step(action)
            next_state = next_state
            
            norm_next_state = normalizer.normalize(next_state)
            memory.push(norm_state,action,reward,norm_next_state,done)
            if train_flag and len(memory)>= train_batch_size:
                train_step(actor_net,critic_net,actor_target_net,critic_target_net,
                           memory,optimizers,tau,train_batch_size,total_timesteps,logger)
            total_timesteps += 1
            ep_reward += reward

            '''
            logging
            '''
            logger.record('Reward',reward,total_timesteps)
            logger.record('Param Noise Std',param_space_noise.current_std,total_timesteps)
            logger.record('Param Noise Std',param_space_noise.current_std,total_timesteps)
            logger.record('State Normalizer Mean',normalizer.mean[0],total_timesteps)
            logger.record('State Normalizer Std',normalizer.std[0],total_timesteps)
            logger.record('Current Epsilon',eps_greedy.eps_now,total_timesteps)

            '''
            end logging
            '''

            if done or st == num_steps-1:
                rew_list.append(ep_reward)
                print("Timesteps: {}, Reward per step: {}, Steps: {}".format(total_timesteps,ep_reward/(st+1),st+1))
                total_episodes += 1
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