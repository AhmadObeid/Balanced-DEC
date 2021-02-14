import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from util import matchup
import pdb

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred, returns=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    # from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    
    if returns:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind 
    else: 
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    
    
def avgacc(y_true, y_pred, matchings): 
    cls_num = np.unique(y_true)[-1]
    idx, assignment = {}, {}
    sz, num_crct, indv_acc = np.zeros([cls_num + 1,]), np.zeros([cls_num + 1,]), np.zeros([cls_num + 1,])

    
    y_pred = matchup(y_pred,matchings)
    for i in range(cls_num + 1):        
        idx[i] = np.arange(len(y_true))[y_true==i] 
        sz[i] = len(idx[i]) # sz = size
        assignment[i] = y_pred[idx[i]]
        num_crct[i] = sum(assignment[i]==i)
        indv_acc[i] = np.double(num_crct[i])/sz[i]
    avg_acc = np.mean(indv_acc)
    M = confusion_matrix(y_true, y_pred)
    p = np.dot(sum(M,1),sum(M,0))/(sum(sum(M)))**2
    pdb.set_trace()
    return avg_acc, p
        
        
        
        
        
        
        
        
        
        
        