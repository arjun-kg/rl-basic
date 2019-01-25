from collections import namedtuple
import numpy as np 
Transition = namedtuple('Transition',
                            ('state', 'action', 'reward','next_state','done'))

a1 = np.zeros((10,2,3))
a2 = np.zeros((10,4))
a3 = np.zeros((10,1))
a4 = np.zeros((10,7,7))
a5 = np.zeros((10,1))



t = [Transition(a,b,c,d,e) for a,b,c,d,e in zip(a1,a2,a3,a4,a5)]