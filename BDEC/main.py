from backbone import *

################################## MAIN #################################
#########################################################################
if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--iter', default=51, type=int)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--init', default='glorot_uniform',type=str)
    parser.add_argument('--update_interval', default=10,type=int)
    parser.add_argument('--neighborhood_size', default=950,type=int)
    parser.add_argument('--lr', default=0.01,type=int)
    parser.add_argument('--file', default=None,type=str)
    parser.add_argument('--mode', default=None,type=str)
    parser.add_argument('--typ', default=None,type=str)

    args = parser.parse_args()
    import os
    ensure(args.file, args.mode, args.typ)
    res_dir = 'results/'+args.mode+'/'+args.file  
    if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
    
    if args.file == 'SA':
        args.neighborhood_size = 500
        args.lr = 0.005
        args.iter = 501
    elif args.file == 'S':
        args.iter = 11
    elif args.file == 'pavia_small':
        args.neighborhood_size = 100
        args.iter = 501
              
    ####### Get and prepare the data #######
    loaded_data = sio.loadmat('datasets/'+args.file+'.mat')
    loaded_gt = sio.loadmat('datasets/'+args.file+'_gt.mat')
    X, Y = loaded_data['data'], loaded_gt['gt']
    a,b,c = np.asarray(X.shape)
    X = np.reshape(X,[a*b,c])
    Y = Y.flatten()
    if args.file == 'PC': #PC has 49 Nan pixels that we'll remove
        valid_idx = np.prod(~np.isnan(X),axis=1).astype(bool)   
        X, Y = X[valid_idx], Y[valid_idx]
    ## removing unlabled pixels ##                
    X, Y = X[Y!=0], Y[Y!=0]    
    ## for standard labeleing (0-->k-1) ##
    Y = fix_label(Y)         
    n_clusters = len(np.unique(Y))           
    
    if args.mode == 'BDEC': #Get the indices + grow neighborhoods around them            
        indices = get_indices(args.file)  
        nbrs = NearestNeighbors(n_neighbors=args.neighborhood_size, algorithm='ball_tree').fit(X) 
        _, neighbor_ind = nbrs.kneighbors(X[indices])
        neighbor_ind = neighbor_ind.flatten()                
        _ = getStats(Y[neighbor_ind],True)
        print('********************************')
        x, y = X[neighbor_ind], Y[neighbor_ind]
    elif args.mode == 'DEC': #No need to get indices
        x,y = X,Y
    else: 
        raise TypeError("Only BDEC/DEC is supported...Please ensure correct spelling")
    

    ####### Prepare the model and use it #######            
    dec = DEC(dims=[X.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=args.init)            
            
    if args.typ == 'predict': 
        dec.model.load_weights('saved/weights/'+args.mode+'/'+args.file+'/DEC_model_final.h5')
        
    elif args.typ =='train':
        dec.autoencoder.load_weights('saved/weights/'+args.mode+'/'+args.file+'/ae_weights.h5')
        dec.compile(optimizer=SGD(args.lr, 0.9), loss='kld')
        dec.fit(x, y=y, x_needed=X, y_needed=Y, iteration=args.iter, batch_size=args.batch_size,
                             update_interval=args.update_interval, save_dir=res_dir)
    
    
    q = dec.model.predict(X)        
    y_pred = q.argmax(1)
    
    print('nmi:', metrics.nmi(Y, y_pred))
    print('ari:', metrics.ari(Y, y_pred))
    
    acc, matchings = metrics.acc(y_pred,Y, True)
    avg_acc, p = metrics.avgacc(y_pred,Y,matchings)
    kappa = np.double(acc-p)/(1-p)
    
    print('OA:', acc)
    print('AA:', avg_acc)
    print('kappa:', kappa)
    
