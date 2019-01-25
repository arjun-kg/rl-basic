from multiprocessing import Pipe,Process

nenvs = 5

def worker(parent,child,env):
    parent.close() #Why?
    while True:
        instruction,data = child.recv()
        if instruction == 'step':
            packet = env.step(data)
            child.send(packet)

        elif instruction == 'reset':
            packet = env.reset()
            child.send(packet)
        elif instruction == 'render':
            child.send(env.render())
        elif instruction == 'spaces':
            child.send((env.observation_space,env.action_space))

class ParallelizeEnv:
    def __init__(self,env):
        self.parents, self.children = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker,args = (self.parents[i],self.children[i],env)) for i in range(nenvs)]

        for i in range(nenvs):
            self.ps[i].daemon = True
            self.ps[i].start()

        for i in range(nenvs):
            self.children[i].close()   #Why?

        self.parents[0].send(('spaces',None))
        self.observation_space,self.action_space =  self.parents[0].recv()

    def step(self,actions):
        for action,parent in zip(actions,self.parents):
            parent.send(('step',action))

        results = [parent.recv() for parent in self.parents]
        obs,rews,dones,infos = zip(*results)

        return obs,rews,dones,infos

    def reset(self):
        for parent in self.parents:
            parent.send(('reset',None)) 
        obs = [parent.recv() for parent in self.parents]
        return obs

    def render(self):
        self.parents[0].send(('render',None))
        self.parents[0].recv()