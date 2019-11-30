# In the first Python interactive shell
import numpy as np
import multiprocessing
from multiprocessing import shared_memory
import scipy
from scipy.sparse import csgraph,csr_matrix

def sharedSearch_D(shmNames,shapes,idx,indexArray):

    #create array of index list to be calculated
    indexList = range(indexArray[0],indexArray[1])
    start_idx = indexArray[0]
    end_idx = indexArray[1]

    # Attach to the existing shared memory block
    data_shm = shared_memory.SharedMemory(name=shmNames[0])
    data_array = np.ndarray(shapes[1], dtype=np.float64, buffer=data_shm.buf)
    
    # Attach to the existing shared memory block
    indices_shm = shared_memory.SharedMemory(name=shmNames[1])
    indices_array = np.ndarray(shapes[2], dtype=np.int32, buffer=indices_shm.buf)
    
    # Attach to the existing shared memory block
    indptr_shm = shared_memory.SharedMemory(name=shmNames[2])
    indptr_array = np.ndarray(shapes[3], dtype=np.int32, buffer=indptr_shm.buf)
    
    reconCSR = csr_matrix((data_array, indices_array, indptr_array), shape=(shapes[0]))

    # Attach to the existing shared memory block
    DST_shm = shared_memory.SharedMemory(name=shmNames[3])
    DST_array = np.ndarray(shapes[0], dtype=np.float64, buffer=DST_shm.buf)
    
    # Attach to the existing shared memory block
    PR_shm = shared_memory.SharedMemory(name=shmNames[4])
    PR_array = np.ndarray(shapes[0], dtype=np.int32, buffer=PR_shm.buf)
    
    distances, predecessors = csgraph.shortest_path(reconCSR,method='D',indices=indexList,directed=True, return_predecessors = True)  

    DST_array[start_idx:end_idx,:] = distances#[:]
    PR_array[start_idx:end_idx,:] = predecessors#[:]

    # Clean up
    data_shm.close()
    indices_shm.close()
    indptr_shm.close()
    DST_shm.close()
    PR_shm.close()

def multiSearch(dataset,nprocs):
    
    datashape = np.shape(dataset)
    data_array = dataset.data #dtype = float64
    indices_array = dataset.indices #dtype = int32
    indptr_array = dataset.indptr #dtype = int32
    
    shapes = [datashape,data_array.shape,indices_array.shape,indptr_array.shape ]

    shmData = shared_memory.SharedMemory(create=True, size=data_array.nbytes)
    shmIndices = shared_memory.SharedMemory(create=True, size=indices_array.nbytes)
    shmIndptr = shared_memory.SharedMemory(create=True, size=indptr_array.nbytes)
    
    DST_array = np.empty(datashape,dtype=np.float64)
    shmDST = shared_memory.SharedMemory(create=True, size=DST_array.nbytes)
    buf_DST = np.ndarray(datashape,dtype=np.float64, buffer = shmDST.buf)  
    shmDST_name = shmDST.name
    
    PR_array = np.empty(datashape,dtype=np.int32)
    shmPR = shared_memory.SharedMemory(create=True, size=PR_array.nbytes)
    buf_PR = np.ndarray(datashape,dtype=np.int32, buffer = shmPR.buf)
    shmPR_name = shmPR.name
    
    buf_data = np.ndarray(data_array.shape,dtype=np.float64, buffer = shmData.buf)
    buf_data[:] = data_array[:]
        
    buf_indices = np.ndarray(indices_array.shape,dtype=np.int32, buffer = shmIndices.buf)
    buf_indices[:] = indices_array[:]
    
    buf_indptr = np.ndarray(indptr_array.shape,dtype=np.int32, buffer = shmIndptr.buf)
    buf_indptr[:] = indptr_array[:]
    
    shmData_name = shmData.name
    shmIndices_name = shmIndices.name
    shmIndptr_name = shmIndptr.name
    
    shmNames = [shmData_name,shmIndices_name,shmIndptr_name,shmDST_name,shmPR_name]
       
    #divide work 
    chunk_calc = int( datashape[0] / float(nprocs) )
    procs = []
    for i in range(nprocs):
        
        #define the section to work on
        startIDX=i*chunk_calc
        endIDX=(i+1)*chunk_calc
        
        if i==nprocs-1:
            endIDX = datashape[0]
        indexList = [startIDX,endIDX]
        #Start the process
        ptemp=multiprocessing.Process(target=sharedSearch_D, args=(shmNames,shapes,i,indexList) )
        ptemp.daemon=True
        ptemp.start()
        procs.append(ptemp)
    
    #Join the processes back together
    for ptemp in procs:
        ptemp.join()
           
    DST_array[:] = buf_DST[:]        
    PR_array[:] = buf_PR[:]
    
    shmData.close()
    shmData.unlink()  # Free and release the shared memory block at the very end
    
    shmIndices.close()
    shmIndices.unlink()  # Free and release the shared memory block at the very end
    
    shmIndptr.close()
    shmIndptr.unlink()  # Free and release the shared memory block at the very end   
    
    shmDST.close()
    shmDST.unlink()  # Free and release the shared memory block at the very end    
    
    shmPR.close()
    shmPR.unlink()  # Free and release the shared memory block at the very end
    
    # return dst_copy, pr_copy
    return DST_array, PR_array
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    