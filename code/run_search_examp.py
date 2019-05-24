import numpy as np
import scipy
import os
import time
import multiprocessing
from scipy.sparse import csgraph

import dijkstra_mp64

def dijkstra_MP_test():
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir( scriptdir )
    #load data file
    #move relative to github directory
    os.chdir( '..' )
    os.chdir( 'data_examples' )
    dataset = scipy.sparse.load_npz('csr_sparse_9kx9k.npz')

    start_time = time.time()
    #set number of processors to use
    nprocs = 10
    DST,PR = dijkstra_mp64.multiSearch(dataset,nprocs)
    print("MultiProc ",nprocs," Time:",time.time()-start_time)

    #run a single core version to compare
    start_time = time.time()
    
    start = None
    distances, predecessors = csgraph.shortest_path(dataset,method='D',indices=start,directed=True, return_predecessors = True) #indices for specific
    print("D Time Single Core:",time.time()-start_time)   
    
    print('\n')
    #check if the multi core code is the same result as calling dijkstra directly 
    print(np.array_equal(PR,predecessors),np.array_equal(DST,distances))

    return
    
if __name__ == '__main__':
    multiprocessing.freeze_support()    
    dijkstra_MP_test()
