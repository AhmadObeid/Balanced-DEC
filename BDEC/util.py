import numpy as np
import pdb
import sys

def ensure(file, mode, typ):
    if file != 'S' and file != 'SA' and file != 'PC' and file != 'PU' and file != 'pavia_small':
        print("Please indicate the correct wanted file\nOptions are: S, SA, PU, PC, pavia_small")
        sys.exit()
    if mode != 'BDEC' and mode != 'DEC':
        print("Please choose the correct mode\nOptions are: DEC, BDEC")
        sys.exit()
    if typ != 'predict' and typ != 'train':
        print("Plase choose the correct excution type\nOptions are: predict, train")
        sys.exit()
    
def get_indices(dataset):
    File = open('saved/indices/'+dataset+'_indices.txt',"r")
    file = File.readlines()    
    data_n = len(file)
    idx = []
    for i in range(data_n):
        idx.append(int(file[i].split()[0]))
    idx = np.asarray(idx)
    return idx    

def fix_label(y):
    unique = np.unique(y)
    actual = range(len(np.unique(y)))    
    for count, individual in enumerate(unique):
        idx = np.arange(len(y))[y==individual]
        y[idx] = actual[count]        
    return y

def getStats(gt,verbose=False):
    uniq = np.unique(gt)
    stats = np.zeros(max(uniq).astype(int)+1)
    for label in uniq:  
        stats[label] = sum(gt==label)        
        if verbose:
            print('Label ', label, ' has ', stats[label], ' samples')   
    return stats

def matchup(y,ind):
    y_new = 10*np.ones(np.asarray(y.shape))
    n_clusters = len(np.unique(y))
    for row in range(n_clusters): 
        idx = np.arange(len(y))[y==ind[row,0]]
        y_new[idx] = ind[row ,1]   
    return y_new

