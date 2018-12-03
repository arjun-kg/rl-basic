from multiprocessing import Pool
from itertools import product
from ddpg import train_ddpg

if __name__ == '__main__':

    train_batch_size = [8,32,128]
    noise_factor = [0.01,0.1]
    eps_start = [0.8,0.5]
    T = [10000,50000]
    gamma = [0.9,0.99]
    tau = [0.01,0.1]
    hidden_layer_size = [16,64,256,512]
    lr_actor = [1e-4,1e-3,1e-2,1e-1]
    lr_critic = [1e-4,1e-3,1e-2,1e-1]

    pool = Pool(processes=4)
    pool.starmap(train_ddpg,product(train_batch_size,noise_factor,eps_start,T,gamma,tau,hidden_layer_size,lr_actor,lr_critic))