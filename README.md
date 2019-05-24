# Multiprocessing Dijkstra Scipy

This is not fully testing, and uses shared memory, so please be careful.  
With that said, contributions, ideas, or advice is welcome.  

## Overview
Scipy is a commonly used package and library with some good optimizations.  However, the shortest_path methods are not too fast.  Since code I have been working on interfaces with some Windows only applications, many of the optimized graph toolkits are not available, or are prohibitively difficult to install for the average user.  This code is meant to try and improve some performance by using the python multiprocessing module.  In the few cases we are using it, it seems to provide some tangible benefit, with the single core method running in 30 seconds, and the multiprocessing in 3 seconds.  

### Requirements
  Python 64 bit (tested on 2.7 or 3.7)  
  Numpy  
  Scipy  
  Windows 10  

### Example
Run the example code provided as run_search_examp.py in the code folder. 

### Credits
Mathew Schwartz, CoAD, NJIT

## Concept
In the case that a user needs all nodes to all nodes, floyd-warshall is sometimes slower (depending on the number of nodes), so dijkstra can be used. As dijkstra can be split to calculate specific indices, this approach divides the number of indices by the processors to be used and calculates each group separately.  

The issue in scalability here is the data transfer between processes as Dijkstra needs the entire matrix to calculate even a short list of indices.  As the CSR matrix is comprised of 3 numpy arrays, the CSR can be split into the arrays and stored into shared memory (RawArray in this case) using the python multiprocessing module.  This is significantly faster than passing the data and much more scalable.  

The other issue with data transfer is returning the results back to the main process.  Since each process calculates a specific index list, it does not share results with the other processes.  The results per process can be written to a ctypes shared array and then loaded in the main process, bypassing the need to copy the data back to the main process.  

