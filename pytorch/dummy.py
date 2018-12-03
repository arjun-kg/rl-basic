from multiprocessing import Process,Value,Array, Pool
from itertools import product

def name(first,last,middle):
    print(first,middle, last)

if __name__ == '__main__':
    first_names = ['Michelle','Barack','Siesta']
    last_names = ['Obama','Jones']
    middle_names = ['Cougar','Kronecker']

    pool = Pool(processes=4)
    pool.starmap(name,product(first_names,last_names,middle_names))
