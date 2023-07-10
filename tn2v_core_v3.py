import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import random
#import ot
#import ot.plot
#import itertools

import os
from os.path import expanduser
home = expanduser("~")

#import scipy.stats as sps
#import sklearn
import matplotlib
#from scipy.stats import multivariate_normal
import scipy
import time

from sklearn.metrics.pairwise import euclidean_distances as skpw

import utils


def tn2v(
    main_directory,
    project_name,
    data,
    mode,
    embedding_dimension,
    param_array,
    l0_array,
    l1_array,
    l2_array,
    eta_array,
    LEN,
    cpu_gpu='cpu',
    mbs_array=None,
    lift_array=None,
    gpd=None,
    W1_data=None,
    grad_old=False,
    nbhd_regen=None,
    pointcloud_data_save=None,
    pointcloud_vf_save=None,
    alpha=1.0
):
    
    
    
    if not project_name.endswith('/'):
        project_name += '/'
    
    path = main_directory + project_name
    if not os.path.isdir(path):
        os.mkdir(path)
   
    data.to_csv(path+'_original_data.csv')
    
    P = pd.DataFrame()
    P['eta'] = eta_array

    P['L0'] = l0_array
    P['L1'] = l1_array
    P['L2'] = l2_array

    P['l'] = [param_array[i]['l'] for i in range(len(param_array))]
    P['r'] = [param_array[i]['r'] for i in range(len(param_array))]
    P['p'] = [param_array[i]['p'] for i in range(len(param_array))]
    P['q'] = [param_array[i]['q'] for i in range(len(param_array))]

    P['lift'] = lift_array

    P['pdata'] = [int(i in pointcloud_data_save) for i in range(LEN+1)]
    P['pvf'] = [int(i in pointcloud_vf_save) for i in range(LEN+1)]
    P['mbs'] = [int(i) for i in mbs_array]
    P.to_csv(path+'data_array.csv')
    
    Q = {
        'mode':mode,
        'embedding_dimension':embedding_dimension,
        'LEN':LEN,
        'nbhd_regen':nbhd_regen,
        'alpha':alpha
    }
    
    pd.Series(Q).to_csv(path+'data_scalar.csv')
    
    X = TN2V(name=project_name,
             mode=mode,
             data_in=data,
             embed_dim=embedding_dimension,
             given_pd=gpd,
             W1_data=W1_data,
             name_regen=False,
             grad_old=grad_old,
             alpha=alpha,
             cpu_gpu=cpu_gpu)
    
    #mega_out = {}
    #timer_data = {}

    X.graph_cycles = False
    

        
    L = X.train(
        epochs=LEN,
        eta_array=eta_array,
        nbhd_params=param_array,
        project_name=project_name,
        pointcloud_data_save=pointcloud_data_save,
        pointcloud_vf_save=pointcloud_vf_save,
        sg_mult_array=l0_array,
        wa1_mult_array=l1_array,
        wa2_mult_array=l2_array,
        lift_array=lift_array,
        minibatch_sizes=mbs_array,
        nbhd_regen=nbhd_regen
    )
    

    pd.DataFrame(L[0]).to_csv(path+'mega_out.csv')
    pd.DataFrame(L[1]).to_csv(path+'timer_data.csv')
    
    return X #,mega_out



def lite(project):
    
    out = {}
    
    def PWDM(data):

        if isinstance(data,pd.DataFrame):
            data = data.values

        try:
            data.shape[0]
        except:
            data = np.array(data)

        return skpw(data)
    
    F = pd.read_csv(home+'/tn2v_output/'+project+'/mega_out.csv',index_col=0).transpose()
    G = pd.read_csv(home+'/tn2v_output/'+project+'/_original_data.csv',index_col=0)
    P = pd.read_csv(home+'/tn2v_output/'+project+'/data_array.csv',index_col=0)
    
    print_data_i = [i for i in range(len(P['pdata'].values)) if P['pdata'].values[i] != 0]

    T = PWDM(G)

    #for i in print_data_i:
    i_index = -1
    IN = None
    while IN is None:
        i = print_data_i[i_index]
        try:
            IN = pd.read_csv(home+'/tn2v_output/'+project+'/w1_at_'+str(i).zfill(12)+'.csv',index_col=0)
        except:
            i_index -= 1
            continue
    S = PWDM(IN)

    out['l2_norm'] = np.linalg.norm(S/np.mean(S)-T/np.mean(T))
    
    out['sg_loss'] = F['sg_loss'].values[-1]
    
    if 'wa_loss_1' in F.columns:
        
        out['wa_loss_1'] = np.mean([x for x in F['wa_loss_1'].values[int(.90*F.shape[0]):-1] if x != 0])
    
    if 'wa_loss_2' in F.columns:
        
        out['wa_loss_2'] = np.mean([x for x in F['wa_loss_2'].values[int(.90*F.shape[0]):-1] if x != 0])

    
    return out

def analyze(project,save=None):

    def PWDM(data):

        if isinstance(data,pd.DataFrame):
            data = data.values

        try:
            data.shape[0]
        except:
            data = np.array(data)

        return skpw(data)
    
    F = pd.read_csv(home+'/tn2v_output/'+project+'/mega_out.csv',index_col=0).transpose()
    G = pd.read_csv(home+'/tn2v_output/'+project+'/_original_data.csv',index_col=0)
    P = pd.read_csv(home+'/tn2v_output/'+project+'/data_array.csv',index_col=0)
    
    Q = pd.read_csv(home+'/tn2v_output/'+project+'/data_scalar.csv')
    Q.rename(columns={'Unnamed: 0':'data','0':'value'},inplace=True)

    Q['value'][3] = '%0.2f'%(float(Q['value'][3])/float(G.shape[0]))

    Q.loc[Q.shape[0]] = ['eta start/end','%0.6f'%(P['eta'].iloc[100])+' , '+str(P['eta'].iloc[P.shape[0]-1])]
    Q.loc[Q.shape[0]] = ['lambda0 start/end',str(P['L0'].iloc[0])+' , '+str(P['L0'].iloc[P.shape[0]-1])]
    Q.loc[Q.shape[0]] = ['lambda1 start/end',str(P['L1'].iloc[0])+' , '+str(P['L1'].iloc[P.shape[0]-1])]
    Q.loc[Q.shape[0]] = ['lambda2 start/end',str(P['L2'].iloc[0])+' , '+str(P['L2'].iloc[P.shape[0]-1])]

    Q.loc[Q.shape[0]] = ['l (walk length) start/end',str(P['l'].iloc[0])+' , '+str(P['l'].iloc[P.shape[0]-1])]
    Q.loc[Q.shape[0]] = ['r (walk count) start/end',str(P['r'].iloc[0])+' , '+str(P['r'].iloc[P.shape[0]-1])]
    
    Q.loc[Q.shape[0]] = ['mbs ','%0.5f'%(float(P['mbs'].iloc[P.shape[0]-1])/float(P.shape[0]))]
    Q.loc[Q.shape[0]] = ['lift ','%0.2f'%float(P['lift'].iloc[P.shape[0]-1])]
    
    print_data_i = [i for i in range(len(P['pdata'].values)) if P['pdata'].values[i] != 0]

    X = []
    Y = []

    T = PWDM(G)

    for i in print_data_i:
        try:
            IN = pd.read_csv(home+'/tn2v_output/'+project+'/w1_at_'+str(i).zfill(12)+'.csv',index_col=0)
        except:
            continue
        S = PWDM(IN)

        X.append(i)
        Y.append(np.linalg.norm(S/np.mean(S)-T/np.mean(T))) # (np.mean(np.abs(S/np.mean(S)-T/np.mean(T))))





    dfin = pd.read_csv(home+'/tn2v_output/'+project+'/w1_at_'+str(print_data_i[-1]).zfill(12)+'.csv',index_col=0)    
    dfin['color'] = np.arange(dfin.shape[0])
    dfin['size'] = [10 for i in range(dfin.shape[0])]




    m = 2.5
    h = 7+int('wa_loss_1' in F.columns)+int('wa_loss_2' in F.columns)
    hr = [1,2,2,1,1]
    if 'wa_loss_1' in F.columns:
        hr.append(1)
    if 'wa_loss_2' in F.columns:
        hr.append(1)
    fig, axs = plt.subplots(nrows=len(hr),ncols=1,
                            figsize=(m*2,m*(h-1-0.5*(int('wa_loss_1' in F.columns))-0.5*(int('wa_loss_2' in F.columns)))),
                            gridspec_kw={'height_ratios':hr})
    fig.suptitle('experiment report: '+project) # +'\n\n'

    table = axs[0].table(cellText=Q.values, colLabels=Q.columns,loc='center')
    table.set_fontsize(14)
    table.scale(1,1)
    axs[0].patch.set_visible(False)
    axs[0].axis('off')

    
    axs[1].scatter(G['0'],G['1'],c=dfin['color'],cmap='viridis')
    axs[1].set_title('target data')

    axs[2].scatter(dfin['0'],dfin['1'],c=dfin['color'],cmap='viridis')
    axs[2].set_title('final embedding state')
    
    axs[3].scatter(X,Y,s=20,alpha=1,c='k')
    axs[3].plot(X,Y,c='k')
    axs[3].set_title('l2-norm of difference of normalized matrices\ntarget data vs. embedding at epoch i')

    axs[4].scatter(list(range(F.shape[0])),F['sg_loss'].values,s=4,alpha=1)
    axs[4].plot(list(range(F.shape[0])),F['sg_loss'].values)
    axs[4].set_title('node2vec loss')
    
    cc = 5
    
    if 'wa_loss_1' in F.columns:
        axs[cc].scatter(list(range(F.shape[0])),F['wa_loss_1'],s=4,alpha=1)
        axs[cc].plot(list(range(F.shape[0])),F['wa_loss_1'].values)
        axs[cc].set_title('homology deg = 1 wasserstein loss')
        axs[cc].set_xlim(0,F.shape[0])
        
        cc += 1
    
    if 'wa_loss_2' in F.columns:
        axs[cc].scatter(list(range(F.shape[0])),F['wa_loss_2'].values,s=4,alpha=1)
        axs[cc].plot(list(range(F.shape[0])),F['wa_loss_2'].values)
        axs[cc].set_title('homology deg = 2 wasserstein loss')
        axs[cc].set_xlim(0,F.shape[0])

    fig.tight_layout()

    if save is not None:
        plt.savefig(save,dpi=150)
    else:
        plt.show()





class TN2V:
    
    # this is the meat of the soup, so to speak
    
    def __init__(self,
                 name,
                 mode='pointcloud',
                 data_in=None,
                 data_extra=None,
                 given_pd=None,
                 embed_dim=2,
                 homology_dimensions=[],
                 verbose=1,
                 W1_data=None,
                 name_regen=False,
                 grad_old=False,
                 alpha=1.0,
                 cpu_gpu='cpu'):
        
        
        '''
        mode: 'pointcloud' or 'distancematrix' or 'correlationmatrix'
        '''
        
        if cpu_gpu == 'cpu':
            import gudhi
        elif cpu_gpu == 'gpu':
            import ripserplusplus as rpp
        
        self.grad_old = grad_old
        
        self.name = name
        self.name_regen = name_regen
        
        self.embed_dim = embed_dim
        
        
        self.alpha = alpha
        
        self.mode = mode
        if mode == 'pointcloud':
            self.target_pointcloud = data_in
            self.target_distance_matrix = self.PWDM(self.target_pointcloud)
        elif mode == 'distancematrix':
            self.target_distance_matrix = data_in
        elif mode == 'correlationmatrix':
            self.target_correlation_matrix = data_in
            if data_extra is not None:
                self.target_distance_matrix = data_extra
            else:
                self.target_distance_matrix = 1/(self.target_correlation_matrix+.0001)
                if isinstance(self.target_distance_matrix,pd.DataFrame):
                    self.target_distance_matrix = self.target_distance_matrix.values
        else:
            print('invalid mode specified. Please choose one of [pointcloud, distancematrix, correlationmatrix]')
            
        '''
        *** NOTE: check for uniqueness in distance matrix (i.e., vietoris rips general position is satisfied; we've seen the experiments blow up when this is not the case)
        '''
        
        self.n = data_in.shape[0]

        self.target_cycles_nontrivial = {}
        
        if given_pd is not None:
            self.target_cycles_nontrivial = {i:[{'birth_time':p[0],'death_time':p[1]} for p in given_pd[i]] for i in given_pd.keys()}
            self.given_pd = given_pd
        else:
            self.given_pd = None
        
        
        '''
        Look, we cannot UPSIZE W1, as it seems the node2vec code always wants to put the data set
        in a [-1,1]^m hypercube. So the other solution is to DOWNSCALE the target distance matrix
        '''
        self.target_autoscaling = True
        if self.target_autoscaling:
            #W1_mult = np.max(self.target_distance_matrix)/2
            self.target_distance_matrix /= (np.max(self.target_distance_matrix)/2)
        
        '''
        instantiate parameter matrices.
        W1 can be set manually to resume from some previously achieved embedding or whatever other reason.
        '''
        if W1_data is None:
            self.W1 = np.random.uniform(-1.0,1.0,(self.n,embed_dim))
        else:
            self.W1 = W1_data
        
        self.W2 = np.random.uniform(-1.0,1.0,(embed_dim,self.n))
        
        '''
        using self.target_distance_matrix, (which is guaranteed to have been generated by now,
        no matter the mode), precompute all the objects necessary to sample neighborhood information
        from the input graph data.
        '''
        self.generate_n2v_info()
        
        '''
        sanity flags, to be trimmed when I can
        '''
        
        self.NBHDs = None
        
        self.complete_counter = None
        self.epoch_counter = None
        
        self.HEIGHT = None
        
        self.graph_cycles = False
        
        self.data_color = None
        
        self.hic_flag = False
        
        self.timer = time.time()
        
        self.homology_mode = cpu_gpu
        self.timer_dict = {}
        
        
    def softmax(self, x):
        '''
        returns the softmax of a vector
        '''
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)   
    
    
    def PWDM(self,data):

        if isinstance(data,pd.DataFrame):
            data = data.values
            
        try:
            data.shape[0]
        except:
            data = np.array(data)
            
        return skpw(data)
            
        
        
    def generate_n2v_info(self):
        
        if self.mode == 'correlationmatrix':
            '''
            basically the correlation matrix can itself immediately serve as the traversal likelihood vectors
            '''
            
            self.nodewise_departure_dict = {}
            for i in range(self.target_correlation_matrix.shape[0]):
                self.nodewise_departure_dict[i] = self.softmax(np.delete(self.target_correlation_matrix.values[i],i))
            
        else:

            distance_matrix = self.target_distance_matrix.copy()

            
            self.edgedict = {}
            for i in range(distance_matrix.shape[0]):
                for j in range(i+1,distance_matrix.shape[0]):
                    self.edgedict[str(i)+'-'+str(j)] = 1/((distance_matrix[i][j]+.0001)**self.alpha)
                    
            
            self.nodewise_departure_dict = {}
            for i in range(distance_matrix.shape[0]):
                X = []
                for j in range(i):
                    X.append(self.edgedict[str(j)+'-'+str(i)])

                for j in range(i+1,self.W1.shape[0]):
                    X.append(self.edgedict[str(i)+'-'+str(j)])

                self.nodewise_departure_dict[i] = list(X/np.sum(X))
            
        
    
    def generate_grad_W_B(self,source,target,h,epsilon=0.01,itermax=100):
        
        '''
        Generate gradient of Wasserstein w.r.t. the Barcode.
        This will call the sinkhorn distance/gradient function.
        '''
        
        # source and target are both arrays of pairs of the forms [birth_time,death_time]
        
        # in the odd event that there is no homology whatsoever...
        if len(source) == 0 and len(target) == 0:
            print('NO MATCHING AT ALL')
            return [],0
        
        # see paper draft for "lift" explanation
        lift = self.lift_array[self.epoch_counter]*(np.max([p['death_time'] for p in self.target_cycles_nontrivial[h]]) - np.min([p['birth_time'] for p in self.target_cycles_nontrivial[h]]))
        
        source_lift = [[-lift*int(i<len(source)),lift*int(i<len(source))] for i in range(len(source))]
        target_lift = [[-lift*int(i<len(target)),lift*int(i<len(target))] for i in range(len(target))]
        
        
        
        
        X = np.array(source)+np.array(source_lift)
        Y = np.array(target)+np.array(target_lift)
        
        # Sinkhorn divergence gradient flow
        
        for i in range(itermax):
            P1, dist1 = utils.hurot_tda(X, Y, eps=epsilon, verbose=0)
            P2, dist2 = utils.hurot_tda(X, X, eps=epsilon, verbose=0)
            P3, dist3 = utils.hurot_tda(Y, Y, eps=epsilon, verbose=0)
            grad1 = X - utils.barycentric_map_tda(P1, X, Y)
            grad2 = X - utils.barycentric_map_tda(P2, X, X)
            grad3 = Y - utils.barycentric_map_tda(P3, Y, Y)
            S = dist1-1/2*dist2-1/2*dist3

            gradS = grad1-1/2*grad2
        
        
        
        
        # here is where the gradient is turned negative
        return -gradS,S

    def generate_grad_B_P(self,i):
        '''
        Generate gradient of Barcode w.r.t. the Pointcloud.
        '''
        
        M=self.n
        L=self.embed_dim

        # every birth death coordinate of nontrivial cycles (by) every coordinate of every point of the pointcloud
        gradient = np.zeros((2*len(self.source_cycles_nontrivial[i]),M*L))

        # for every nontrivial cycle
        for s in range(len(self.source_cycles_nontrivial[i])):

            # p has all the info
            # BUT HEY, THIS INFO IS IN SEQUENTIAL MINIBATCHED INDICES
            p = self.source_cycles_nontrivial[i][s]
            
            print('elucidation p')
            print(p)

            
            Btemp = sorted(p['birth_simplex'])
            if len(Btemp) == 2:
                B = Btemp
            else:
                B = sorted([[[Btemp[k],Btemp[l]],self.source_distance_matrix[Btemp[k]][Btemp[l]]]
                            for k in range(len(Btemp)) for l in range(k+1,len(Btemp))],
                           key = lambda x:x[1])[0][0]
            
            print('elucidation B')
            print(B)


            Dtemp = sorted(p['death_simplex'])
            if len(Dtemp) == 2:
                D = Dtemp
            else:
                D = sorted([[[Dtemp[k],Dtemp[l]],self.source_distance_matrix[Dtemp[k]][Dtemp[l]]]
                            for k in range(len(Dtemp)) for l in range(k+1,len(Dtemp))],
                           key = lambda x:x[1])[0][0]
            
            print('elucidation D')
            print(D)
            
            # B = birth edge
            # D = death edge
            # relative to the global point-naming system? or local w/minibatches?
            
            B = [self.batch_to_original[B[0]],self.batch_to_original[B[1]]]
            D = [self.batch_to_original[D[0]],self.batch_to_original[D[1]]]
            
            [ua,ub] = B
            [uc,ud] = D

            # (a - b)/||a - b||
            ab_diff = np.subtract(self.W1[ua],self.W1[ub])
            ab_diff = ab_diff/np.linalg.norm(ab_diff)
            
            # (c - d)/||c - d||
            cd_diff = np.subtract(self.W1[uc],self.W1[ud])
            cd_diff = cd_diff/np.linalg.norm(cd_diff)
            
            
            print('elucidation ab_diff')
            print(ab_diff)
            
            print('elucidation cd_diff')
            print(cd_diff)
            
            # L is the embedding dimension (m)
            # the gradient matrix is of
            #  rows = 2*number of nontrivial cycles
            #  columns = every coordinate of every point (points first, then coords, obviously)
            for coordinate in range(L):
                # s is the sequential nontrivial cycle index
                # split into birth/death, we are indeed interested in 2s and 2s+1 as rows
                
                # ua*L+coordinate selects the ua-th section and then, indeed, the appropriate coordinate
                # this represents that the partial of the s-th generator w.r.t. the ua-th point is indeed the vector ua-ub, normalized
                # the same thing for the ub-th point is the same vector with the opposite sign
                gradient[2*s][ua*L+coordinate] = ab_diff[coordinate]
                gradient[2*s][ub*L+coordinate] =-ab_diff[coordinate]
                gradient[2*s+1][uc*L+coordinate] = cd_diff[coordinate]
                gradient[2*s+1][ud*L+coordinate] =-cd_diff[coordinate]

        return gradient

    
    def one_step_vectors(self,x_in,x_E):
        
        n,m = self.W1.shape
        
        # (m x 1): this is x_in's row of W1
        x_in = np.array(x_in).reshape(n,1)

        middle_product = np.dot(x_in.T,self.W1)
        middle_product = middle_product.reshape(m,1)
        
        # (n x 1)
        ux = np.dot(middle_product.T,self.W2).T
        ux = ux.reshape(n,1)
        ux_bar = self.softmax(ux).reshape(n,1)
        
        # x_E - expected probabilities of neighbors for x_in
        x_E = np.array(x_E).reshape(n,1)
        
        self.sg_loss = -np.sum([x_E[i]*np.log(ux_bar[i]) for i in range(n)])
        
        error_vector = np.subtract(ux_bar,x_E)
        
        error_scalar = np.sum(error_vector,axis=0)
        #if verbose > 0:
        #    print('pair error is '+str(error_scalar))
        error_vector = error_vector.reshape(n,1)
        
        # (m x 1) outer (n x 1)
        dL_dW2 = np.outer(middle_product,error_vector)
        
        # (n x 1) outer [(m x n) dot (n x 1)]
        dL_dW1 = np.outer(x_in, np.dot(self.W2,error_vector))
        
        return dL_dW1,dL_dW2
        
            
    def one_step(self,x_in,x_E,i,verbose=0):
        '''
        performs one pass of the network with a single in-out pair
        
        notation:
        ux_bar         - softmaxed output vector
        ux             - pre-softmaxed output vector
        middle_product - x_in * W1
        '''
        
        n,m = self.W1.shape
        
        # (m x 1): this is x_in's row of W1
        x_in = np.array(x_in).reshape(n,1)
        if self.W1.shape != (self.n,self.embed_dim):
            print('W1 shape mismatch?')
        middle_product = np.dot(x_in.T,self.W1)
        middle_product = middle_product.reshape(m,1)
        
        # (n x 1)
        if self.W2.shape != (self.embed_dim,self.n):
            print('W2 shape mismatch?')
        ux = np.dot(middle_product.T,self.W2).T
        ux = ux.reshape(n,1)
        ux_bar = self.softmax(ux).reshape(n,1)
        
        # x_E - expected probabilities of neighbors for x_in
        x_E = np.array(x_E).reshape(n,1)
        
        #self.sg_loss = -np.sum([x_E[i]*np.log(ux_bar[i]) for i in range(n)])
        self.sg_loss = -1*np.log(ux_bar[i])
        
        error_vector = np.subtract(ux_bar,x_E)
        
        error_scalar = np.sum(error_vector,axis=0)
        if verbose > 0:
            print('pair error is '+str(error_scalar))
        error_vector = error_vector.reshape(n,1)
        
        # (m x 1) outer (n x 1)
        dL_dW2_single = np.outer(middle_product,error_vector)
        
        # (n x 1) outer [(m x n) dot (n x 1)]
        dL_dW1_single = np.outer(x_in, np.dot(self.W2,error_vector))
        
        return dL_dW1_single,dL_dW2_single
    
        
    def print_vf_bc(self,i,save=True,wma=0):
        '''
        A function for printing out the vector-field of barcode gradient movements.
        '''
        
        if len(self.P0[i]) == 0 or len(self.P1[i]) == 0:
            return
            
        fig1 = plt.figure(figsize=(10,8))

        print('-- printing barcodes')
        
        X0 = np.array([[p['birth_time'],p['death_time']] for p in self.P0[i]])
        X1 = np.array([[p['birth_time'],p['death_time']] for p in self.P1[i]])
        
        # self.grad_W_B should be the |sampled| x 2 matrix which lines up exactly
        # with the size of X0, X1.
        
        plt.scatter(X0[:, 0], X0[:, 1], s=20.0, alpha=.5, label='BC from target')
        plt.scatter(X1[:, 0], X1[:, 1], s=20.0, alpha=.5, label='BC from source')
        
        for j in range(X1.shape[0]):
            plt.plot([X1[j][0],X1[j][0]+self.eta*wma*self.grad_W_B[i][j][0]],[X1[j][1],X1[j][1]+self.eta*wma*self.grad_W_B[i][j][1]],color='k')
        
        if len(self.target_cycles_nontrivial[i]) > 0:
            DIAM = 2*(np.max([p['death_time'] for p in self.target_cycles_nontrivial[i]]))
        else:
            DIAM = 2
        plt.xlim(0,DIAM)
        plt.ylim(0,DIAM)
        
        plt.plot([0,DIAM],[0,DIAM],color='gray')
        
        print('bc save statement')
        
        plt.legend(loc=0)
        #plt.title('former barcode stage, lines are along negative of W_B_grad (towards Wass = 0)')
        if save:
            plt.savefig(home+'/tn2v_output/'+self.name+'/print_vf_bc_'+str(self.complete_counter).zfill(12)+'_deg'+str(i)+'.png')
            plt.close()
        else:
            plt.show()
        
    def print_vf_pc(self):
        '''
        A function for printing out the vector-field of pointcloud gradient movements.
        '''
        
        fig1 = plt.figure(figsize=(10,8))

        print('-- printing pointcloud')
        
        X1 = self.previous_W1
        
        if self.data_color is not None:
            color = self.data_color
        else:
            color = np.arange(X1.shape[0])
        plt.scatter(X1[:,0], X1[:,1], s=40.0, alpha=.5,label='W1',c=color)
        
        # Look, if you want to plot ALL gradient movements separately, just copy/save the matrices of W1
        # at various steps and reference them here rather than recomputing all this...
        
        for i in range(X1.shape[0]):

            plt.plot([X1[i][0],self.W1[i][0]],[X1[i][1],self.W1[i][1]],label='movement gradient',color='black')
                
        if self.graph_cycles:
            # ******** this is un-checked and un-updated code that I may still want to implement later
            
            # presently hard-coded to degree 1
            birth_simplices = [p['birth_simplex'] for p in self.source_cycles_nontrivial[1]]
            death_simplices = [p['death_simplex'] for p in self.source_cycles_nontrivial[1]]
            
            for D in death_simplices:
                indices = [int(k) for k in D]
                x = [X1[i][0] for i in indices]
                y = [X1[i][1] for i in indices]

                if D[1] == 1:
                    plt.fill(x,y,'k',alpha=0.05)
                else:
                    plt.fill(x,y,'darkviolet',alpha=0.05)
                
            for B in birth_simplices:
                indices = [int(k) for k in B]
                x = [X1[i][0] for i in indices]
                y = [X1[i][1] for i in indices]
                
                if B[1] == 1:
                    plt.plot(x,y,c='k',alpha=0.2)
                else:
                    plt.plot(x,y,c='darkviolet',alpha=0.2)
            
        
        handles, labels = plt.gca().get_legend_handles_labels()

        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        #plt.legend(handles, labels, loc='best')
        #plt.show()

        print('pc save statement')
        
        plt.legend(handles, labels,loc=1)
        plt.title('former pointcloud plot, lines trace to updated positions for current W1')
        plt.savefig(home+'/tn2v_output/'+self.name+'/print_vf_pc_'+str(self.epoch_counter).zfill(12)+'.png')
        plt.close()
        
    
    def train(self,
              epochs,
              eta_array,
              nbhd_params,
              project_name,
              subsample_points=None,
              subsample_nbhds=None,
              pointcloud_data_save=None,
              pointcloud_vf_save=None,
              sg_mult_array=None,
              wa1_mult_array=None,
              wa2_mult_array=None,
              lift_array=None,
              minibatch_sizes=None, # minibatch_sizes
              hausdorff_subsample_threshold=0.5,
              almm={'mode':False,
                    'b_initial':0.05,
                    'b_growth':1.5,
                    'b_max':1,
                    'l_initial':0},
              PD_window=None,
              nbhd_regen=None):
        '''
        e = number of epochs
        eta = step size / learning parameter (constant multiple of gradient for updating network parameters)
        p = return parameter
        q = advance parameter
        l = length of each walk
        r = number of walks
        subsample_points = sample size of original data for each epoch (if None, xs = cardinality of input data)
        subsample_nbhds = sample size of neighborhood for each data point (if None, ns = l*r)
        
        * relative weights on the gradients of skip-gram vs wasserstein
          and whether or not to run said gradient at any given step
          are entirely governed by sg_mult_array and wa_mult_array
        
        '''
        
        self.hausdorff_subsample_threshold = hausdorff_subsample_threshold
        
        # this makes a triangle in the upper half plane and only considers generators in that triangle — not sure why I included this or wanted to use it, but it's here and implemented...
        if PD_window is not None:
            self.PD_window = PD_window
        else:
            self.PD_window = None
            
        if lift_array is None:
            self.lift_array = [0 for i in range(epochs)]
        else:
            self.lift_array = lift_array
        
        '''
        minibatch_size of None (or size of the whole data set) skips mini-batching code altogether and just goes ahead and generates the PD of the target data ONE TIME.
        '''
        #if minibatch_size is None or minibatch_size == self.W1.shape[0]:
        #    self.batches = False
        #else:
        #    self.minibatch_size = minibatch_size
        #    self.batches = True
            

        
        if subsample_points is None or subsample_points > self.n:
            subsample_points = self.n
            
        #if subsample_nbhds is None or subsample_nbhds > l*r:
        #    subsample_nbhds = l*r
           
        # currently sloppy ALMM (augmented lagrange multiplier method) code, but it is functional
        if almm['mode']:
            try:
                self.almm
            except:
                self.almm = {}
                self.almm['b_mult'] = almm['b_initial']
                self.almm['b_max'] = almm['b_max']
                self.almm['l_mult'] = {i:almm['l_initial'] for i in self.HD}
                self.almm['b_growth'] = almm['b_growth']
        else:
            self.almm = {}

        mega_out = {}
        
        if self.complete_counter is None:
            self.complete_counter = 1
        if self.epoch_counter is None:
            self.epoch_counter = 1
        
        
        self.nbhd_info_generated = False
        
        
        
        
        for unrelated_counter in range(epochs):
            
            i = self.epoch_counter
            
            if minibatch_sizes == None:
                self.batches = False
            else:
                if minibatch_sizes[i] >= self.W1.shape[0]:
                    self.batches = False
                else:
                    self.minibatch_size = minibatch_sizes[i]
                    self.batches = True
            
            self.eta = eta_array[i]
            
            p = nbhd_params[i]['p']
            q = nbhd_params[i]['q']
            l = nbhd_params[i]['l']
            r = nbhd_params[i]['r']
            
            self.timer_dict[i] = {}
            
            if not self.nbhd_info_generated:
                print('gen nbhds INITIAL: '+str(round(time.time()%1000,3)))
                self.timer_dict[i]['get nbhs start'] = (round(time.time()%1000,3))
                self.gen_nbhds(p,q,l,r)
                
                self.nbhd_info_generated = True
            
            else:
                if not nbhd_regen is None:
                    if i%nbhd_regen == 0:
                    # generate a new training set.

                        print('gen nbhds epoch '+str(i)+': '+str(round(time.time()%1000,3)))
                        self.timer_dict[i]['get nbhs start'] = (round(time.time()%1000,3))
                        self.gen_nbhds(p,q,l,r)

                        # **** need to reimplement this for nbhd subsampling
                        #inputs,outputs = self.generate_training_set(subsample_points,subsample_nbhds)
                        #training_data = [[inputs[j],outputs[j]] for j in range(len(inputs))]
                        #print('gen nbhds end: '+str(round(time.time()%1000,3)))
                        #self.timer_dict[i]['gen nbhds end'] = (round(time.time()%1000,3))


            
            print('EPOCH #'+str(i))
            self.timer_dict[i] = {}
            print('EPOCH start: '+str(round(time.time()%1000,3)))
            self.timer_dict[i]['EPOCH start'] = (round(time.time()%1000,3))
            
                
            '''
            instantiate gradient matrices here, because the loss function is computed POINT by POINT in the training data, so we aggregate all the changes into these matrices.
            '''
            
            self.previous_W1 = self.W1.copy()
            
            
            self.HD = []
            if wa1_mult_array[i] != 0:
                self.HD.append(1)
            if wa2_mult_array[i] != 0:
                self.HD.append(2)
            
            if len(self.HD) > 0:
                
                #print('- wa1_mult_array is not none, epoch is '+str(self.epoch_counter))
                #print('- wa1_mult_array[epoch] = '+str(wa1_mult_array[self.epoch_counter]))

                self.wasserstein_full()
                
                if almm['mode']:
                    
                    self.gradient_product = {}
                    for h in self.HD:
                        self.gradient_product[h] = -self.almm['l_mult'][h]+self.almm['b_mult']*self.WD[h]
                        self.W1 += self.eta*min(self.gradient_product[h],wa1_mult_array[i])*self.final_wass_gradient[h]
                        
                        self.almm['l_mult'][h] = self.almm['l_mult'][h] - self.almm['b_mult']*self.WD[h]
                    
                    self.almm['b_mult'] *= 1+(self.almm['b_growth']-1)*int(self.almm['b_mult'] < self.almm['b_max'])
                    
                else:
                    
                    print('wa_out1: '+str(round(time.time()%1000,3)))
                    self.timer_dict[i]['wa_out1'] = (round(time.time()%1000,3))
                    
                    self.gradient_product = {}
                    for h in self.HD:
                        self.gradient_product[h] = wa1_mult_array[i]*(int(h==1))+wa2_mult_array[i]*(int(h==2))
                        self.W1 += self.eta*self.gradient_product[h]*self.final_wass_gradient[h]
                    print('wa_out2: '+str(round(time.time()%1000,3)))
                    self.timer_dict[i]['wa_out2'] = (round(time.time()%1000,3))
            else:
                self.gradient_product = {i:0 for i in self.HD}
            
            
            
            
            
            
            
            
            
            
            # node2vec portion
            
            sg_loss_list = []
            
            if sg_mult_array[i] != 0:
            
                print('sg_start: '+str(round(time.time()%1000,3)))
                self.timer_dict[i]['sg_start'] = (round(time.time()%1000,3))

                # instantiate gradient matrices outside of the next loop, in which we compute gradient for every vertex's nbhd individually, compile them in these matrices, then enact the summed gradients all AT ONCE. Otherwise there is this weird imbalance with the numerically 'later' vertices getting a more nuanced update than the early ones.
                self.SG_W1_grad = np.zeros(self.W1.shape)
                self.SG_W2_grad = np.zeros(self.W2.shape)

                # go through all vertices and their respective training nbhds
                for x in self.NBHDs.keys():

                    # x is a vertex index
                    x_in = np.zeros(self.W1.shape[0])
                    x_in[x] = 1

                    # y is also a vertex index
                    x_E = self.NBHDs[x]


                    dL_dW1,dL_dW2 = self.one_step_vectors(x_in,x_E)

                    self.SG_W1_grad += dL_dW1
                    self.SG_W2_grad += dL_dW2

                    sg_loss_list.append(self.sg_loss)

                    self.complete_counter += 1
                    
                self.W2 -= self.eta*sg_mult_array[i]*self.SG_W2_grad
                self.W1 -= self.eta*sg_mult_array[i]*self.SG_W1_grad

                
            print('sg_end: '+str(round(time.time()%1000,3)))
            self.timer_dict[i]['sg_end'] = (round(time.time()%1000,3))
            
            
            
            
            
            
            
            # the rest is all data collection for saving to the "mega_out" file/object for review
                
            mega_out[i] = {}
                        
            mega_out[i]['sg_w1_grad_MEAN'] = np.sum(np.abs(self.SG_W1_grad))/len(np.nonzero(self.SG_W1_grad)[0])
            mega_out[i]['sg_w2_grad_MEAN'] = np.sum(np.abs(self.SG_W2_grad))/len(np.nonzero(self.SG_W2_grad)[0])
            
            mega_out[i]['sg_w1_grad_MAX'] = np.max(np.abs(self.SG_W1_grad))
            mega_out[i]['sg_w2_grad_MAX'] = np.max(np.abs(self.SG_W2_grad))
            
            if almm['mode']:
                mega_out[i]['b_mult'] = self.almm['b_mult']
                mega_out[i]['l_mult'] = {}
                for h in self.HD:
                    mega_out[i]['l_mult'][h] = self.almm['l_mult'][h]

            try:
                for h in self.HD:
                    mega_out[i]['wass_grad_'+str(h)+'_MEAN'] = np.sum(np.abs(self.final_wass_gradient[h]))/len(np.nonzero(self.final_wass_gradient[h])[0])
                    mega_out[i]['wass_grad_'+str(h)+'_MAX'] = np.max(np.abs(self.final_wass_gradient[h]))

                    mega_out[i]['wa_loss_'+str(h)] = self.WD[h]
                    print('wd'+str(h)+': '+str(self.WD[h]))
            except:
                for h in self.HD:
                    mega_out[i]['wass_grad_'+str(h)+'_MEAN'] = -1
                    mega_out[i]['wass_grad_'+str(h)+'_MAX'] = -1

                    mega_out[i]['wa_loss_'+str(h)] = -1
                
            mega_out[i]['sg_loss'] = np.mean(sg_loss_list)

            
            
            if not project_name.endswith('/'):
                project_name += '/'
            
            if i in pointcloud_data_save:
                pd.DataFrame(self.W1).to_csv(home+'/tn2v_output/'+project_name+'w1_at_'+str(i).zfill(12)+'.csv')
            
            
            if i in pointcloud_vf_save:

                self.print_vf_pc()

                if wa1_mult_array[i] != 0:

                    if i > 0:

                        for j in self.HD:

                            self.print_vf_bc(j,wma=self.gradient_product[j])
            
            
            print('EPOCH end: '+str(round(time.time()%1000,3)))
            self.timer_dict[i]['EPOCH end'] = (round(time.time()%1000,3))
            print('\n')
            self.epoch_counter += 1
            
        return mega_out,self.timer_dict
    
    def gen_PD(self,D):
        
        if self.homology_mode == 'gpu':

            def radius_of_simplex(s):
                return np.max([D[s[i]][s[j]] for i in range(len(s)) for j in range(i+1,len(s))])

            X = rpp.run('--format distance --dim '+str(max(self.HD)),D)

            rpp_hom_dict = {}
            for dim in self.HD:
                if dim == 0:
                    continue
                    # our edited ripser-plusplus code does NOT accomodate dim0 at the moment
                rpp_hom_dict[dim] = []
                for l in X[dim]:
                    rpp_hom_dict[dim].append([[l[i] for i in range(dim+1)],[l[5+i] for i in range(dim+2)]])

            

            PD_di = {}

            for i in self.HD:
                PD_di[i] = {'simplices':rpp_hom_dict[i],'times':[[radius_of_simplex(pair[0]),radius_of_simplex(pair[1])]
                                                                 for pair in rpp_hom_dict[i]]}


            return PD_di
        
        elif self.homology_mode == 'cpu':
            
            rips_complex = gudhi.RipsComplex(distance_matrix=D)
            rips_simplex_tree = rips_complex.create_simplex_tree(max_dimension = max(self.HD)+1)
            rips_simplex_tree.compute_persistence()
            
            PD_di = {}
        
            for i in self.HD:
                M = [p for p in rips_simplex_tree.persistence_pairs() if len(p[1]) == i+2]
                R = rips_simplex_tree.persistence_intervals_in_dimension(i)

                PD_di[i] = {'simplices':M,'times':R}

            return PD_di
            
    
    def acquire_target_cycles(self,h,batch=False):
        
        '''
        '''
        
        if batch:
            PD = self.gen_PD(self.target_distance_matrix_BATCH)# **
        else:
            PD = self.gen_PD(self.target_distance_matrix)# **

        for h in self.HD:
            nti = [j for j in range(len(PD[h]['simplices']))]

            tcn = [{'birth_simplex':PD[h]['simplices'][j][0],
                    'death_simplex':PD[h]['simplices'][j][1],
                    'birth_time':PD[h]['times'][j][0],
                    'death_time':PD[h]['times'][j][1],
                    'original_sequential_index':j} for j in nti]

            self.target_cycles_nontrivial[h] = tcn
            

    
    
    
    def gen_nbhds(self,p=1,q=1,l=5,r=10):
        '''
        generate neighborhoods with the parameters
        p, q, l = walk length, r = number of walks
        '''
        if l*r < self.n:
            NBHDs = {}
            for i in range(self.n):
                WALKS = self.generate_walks(node=i,p=p,q=q,l=l,r=r)
                nbhd = np.concatenate([WALKS[key][1:] for key in WALKS.keys()])
                
                prob_vector = [0 for i in range(self.n)]
                for x in list(nbhd):
                    prob_vector[x] += 1
                
                S = np.sum(prob_vector)
                prob_vector = [y/S for y in prob_vector]
                
                NBHDs[i] = prob_vector
        else:
            NBHDs = {i:self.vect_with_zero(i) for i in range(self.n)}
            
        self.NBHDs = NBHDs
        
        
    def generate_walks(self,node,p,q,l,r):
        '''
        l = length of each walk
        r = number of walks
        '''
        walks = {}
        for w in range(r):
            walks[w] = [node]
            for y in range(l):
                if y == 0:
                    walks[w].append(self.get_next_random_node(p,q,walks[w][-1]))
                else:
                    walks[w].append(self.get_next_random_node(p,q,walks[w][-1],walks[w][-2]))

        return walks
            
            
    def pq_func(self,p,q,i,prev):
        '''
        p: return parameter
        q: advance parameter
        i: index of current node in path
        prev: index of previous node in path
        
        returns: modified probability vector leaving from 'i' given that the previous node in the path was 'prev'.
                 note that unlike 'nodewise_departure[i]', which does not include an index for i itself,
                 this returned vector has the full length of the dataset, where of course the ith index is
                 set to zero.
        '''
    
        ndiz = self.vect_with_zero(i)
        ndjz = self.vect_with_zero(prev)

        adjustment_vector = []
        for k in range(len(ndjz)):
            if k == prev:
                adjustment_vector.append(p)
            elif k == i:
                adjustment_vector.append(0)
            elif ndjz[k] == 0:
                adjustment_vector.append(q)
            else:
                adjustment_vector.append(1)

        modified_transition_vector = np.multiply(ndiz,adjustment_vector)
        S = np.sum(modified_transition_vector)
        modified_transition_vector /= S
        return modified_transition_vector
        
    
    def vect_with_zero(self,i):
        '''
        receives an index and returns the nodewise departure vector but with with a zero in the ith coordinate
        (ordinary departure vectors are one short, which is probably kind of silly, but so far it's working fine)
        '''
        return [x for Y in [self.nodewise_departure_dict[i][:i],[0],self.nodewise_departure_dict[i][i:]] for x in Y]
    
    
    def get_next_random_node(self,p,q,i,prev=None):
        '''
        p: return parameter
        q: advance parameter
        i: 'head' of current walk
        prev: previous node in current walk (if there was one)
        
        returns: the index of the next node in the walk, taking all probabilities into account
        '''
        candidates = list(np.arange(len(self.nodewise_departure_dict[0])+1))

        if prev is None:
            probabilities = self.vect_with_zero(i)
        else:
            probabilities = self.pq_func(p,q,i,prev)

        j = np.random.choice(candidates,
                             p=probabilities)
        return j

    
    def generate_training_set(self,number_of_pairs=None,size_of_nbhds=None):
        
        if number_of_pairs is None:
            number_of_pairs = self.n
        if size_of_nbhds is None:
            size_of_nbhds = len(self.NBHDs[0])
        if size_of_nbhds > len(self.NBHDs[0]):
            size_of_nbhds = len(self.NBHDs[0])
        
        if number_of_pairs < self.n:
            CHOOSE_inputs = np.random.choice(list(self.NBHDs.keys()),size=number_of_pairs)
        else:
            CHOOSE_inputs = list(self.NBHDs.keys())

        outputs = []
        for i in CHOOSE_inputs:
            if len(self.NBHDs[i]) > size_of_nbhds:
                CHOOSE_out_nodes = np.random.choice(self.NBHDs[i],size=size_of_nbhds)
            else:
                CHOOSE_out_nodes = self.NBHDs[i].copy()
            prob_vector = [0.0 for j in range(len(list(self.NBHDs.keys())))]

            for o in CHOOSE_out_nodes:
                prob_vector[o] += 1.0
                
            s = np.sum(prob_vector)
            
            prob_vector /= s

            outputs.append(prob_vector)
            
            # very bad check just to ensure that these were being probability vectors
            #if int(np.sum(prob_vector)*100) not in [99,100]:
            #    print('x_E\'s not prob vectors')
            #    print(prob_vector)
            #    print(np.sum(prob_vector))

        # make one-hot vectors for each of the indices left in CHOOSE_inputs
        inputs = [[int(i==ri) for i in range(len(list(self.NBHDs.keys())))] for ri in CHOOSE_inputs]

        return inputs,outputs
    
    
    
    def wasserstein_full(self):
        
        '''
        Compute and apply the wasserstein gradient
        '''
        
        print('wa_full_1: '+str(round(time.time()%1000,3)))
        self.timer_dict[self.epoch_counter]['wa_full_1'] = (round(time.time()%1000,3))
        
        # if mini-batching, need to compute target homology each step
        if self.batches:
            self.mini_batch()
        # otherwise, you need to generate target homology precisely ONCE
        else:
            generate = False
            for h in self.HD:
                if h not in self.target_cycles_nontrivial.keys():
                    generate = True
                    
            if generate:
                for h in self.HD:
                    self.acquire_target_cycles(h,batch=False)

                    
        
        
        
        
        print('wa_full_2: '+str(round(time.time()%1000,3)))
        self.timer_dict[self.epoch_counter]['wa_full_2'] = (round(time.time()%1000,3))
        
        
        print('wa_full_3: '+str(round(time.time()%1000,3)))
        self.timer_dict[self.epoch_counter]['wa_full_3'] = (round(time.time()%1000,3))
        
        self.generate_source_objects()
        
        print('wa_full_4: '+str(round(time.time()%1000,3)))
        self.timer_dict[self.epoch_counter]['wa_full_4'] = (round(time.time()%1000,3))
        
        self.wasserstein_gradient()
        
        print('wa_full_5: '+str(round(time.time()%1000,3)))
        self.timer_dict[self.epoch_counter]['wa_full_5'] = (round(time.time()%1000,3))
        
        M,L = self.n,self.embed_dim
        self.final_wass_gradient = {}
        
        for i in self.HD:
            if len(self.P0[i]) == 0 or len(self.P1[i]) == 0:
                self.final_wass_gradient[i] = np.zeros((M,L))
            else:
                self.final_wass_gradient[i] = self.grad_W_P[i].reshape(M,L)
        print('wa_full_6: '+str(round(time.time()%1000,3)))
        self.timer_dict[self.epoch_counter]['wa_full_6'] = (round(time.time()%1000,3))
                
    
    def generate_source_objects(self):
        
        print('gen_source_1: '+str(time.time()))
        self.timer_dict[self.epoch_counter]['gen_source_pd_1'] = (round(time.time()%1000,3))
        
        if self.batches:
            self.source_pointcloud = [self.W1[i] for i in self.batch_indices]
        else:
            self.source_pointcloud = self.W1 #.copy() **** I don't think the copy is necessary? And if we're trying to maximize speed anything helps; cloning an entire n x m matrix is pretty unnecessary.
        
        print('gen_source_2: '+str(time.time()))
        self.timer_dict[self.epoch_counter]['gen_source_pd_2'] = (round(time.time()%1000,3))
        
        # **** currently new most computation-intensive component of the code.
        self.source_distance_matrix = self.PWDM(self.source_pointcloud)
        
        print('gen_source_3: '+str(time.time()))
        self.timer_dict[self.epoch_counter]['gen_source_pd_3'] = (round(time.time()%1000,3))
        
        self.source_PD_dth_diagram = self.gen_PD(self.source_distance_matrix)
        
        print('gen_source_4: '+str(time.time()))
        self.timer_dict[self.epoch_counter]['gen_source_pd_4'] = (round(time.time()%1000,3))
        
        self.source_cycles_nontrivial = {}
        
        for h in self.HD:            
                
            PD = self.source_PD_dth_diagram[h]
            
            nti = [j for j in range(len(PD['simplices']))]
                 
            scn = [{'birth_simplex':PD['simplices'][j][0],
                    'death_simplex':PD['simplices'][j][1],
                    'birth_time':PD['times'][j][0],
                    'death_time':PD['times'][j][1],
                    'original_sequential_index':j} for j in nti]                 
                 
            self.source_cycles_nontrivial[h] = scn
            
        print('gen_source_5: '+str(time.time()))
        self.timer_dict[self.epoch_counter]['gen_source_pd_5'] = (round(time.time()%1000,3))

                
            
    def wasserstein_gradient(self):
        '''
        regenerates all W1-related objects (as W1 will have updated since the last time) and returns
        the full gradient of the Wasserstein distance between W1's pointcloud and data0's pointcloud
        with respect to the individual coordinates of the W1 pointcloud/matrix.
        '''
        
        print('ENTER wasserstein_gradient')
        
        if self.PD_window is not None:
            [x,y] = self.PD_window
            for h in self.target_cycles_nontrivial.keys():
                self.target_cycles_nontrivial[h] = [p for p in self.target_cycles_nontrivial
                              if x<p['death_time']-p['birth_time']
                              and p['death_time']-p['birth_time']<y]
                self.source_cycles_nontrivial[h] = [p for p in self.source_cycles_nontrivial
                              if x<p['death_time']-p['birth_time']
                              and p['death_time']-p['birth_time']<y]
        
        # purely for notational sanity
        self.P0 = self.target_cycles_nontrivial
        self.P1 = self.source_cycles_nontrivial
        
        
           
        X0 = {h:[[p['birth_time'],p['death_time']] for p in self.P0[h]] for h in self.P0.keys()}
        X1 = {h:[[p['birth_time'],p['death_time']] for p in self.P1[h]] for h in self.P1.keys()}
                
        self.WD = {}
        self.grad_W_B = {}
        self.grad_W_B_LIN = {}
        self.grad_B_P = {}
        self.grad_W_P = {}
        
        for h in self.HD:
            
            if len(self.P1[h]) == 0 or len(self.P0[h]) == 0:
                # particularly with high-dimensional homology, this might happen a lot.
                # setting self.WD[i] to zero is necessary to cause no interruption where? ****
                self.WD[h] = 0
                continue
            
            grad_W_B,self.WD[h] = self.generate_grad_W_B(target=X0[h],source=X1[h],h=h)

            if pd.isna(self.WD[h]):
                print(str(h)+'-th WASSERSTEIN DISTANCE IS INFINITE')
        
            self.grad_W_B[h] = grad_W_B
            
            print('grad_W_B for elucidation')
            print(self.grad_W_B[h])
        
            grad_W_B_lin = grad_W_B.reshape(grad_W_B.shape[0]*grad_W_B.shape[1],1)
            self.grad_W_B_LIN[h] = grad_W_B_lin
            
            self.grad_B_P[h] = self.generate_grad_B_P(h)
            
            #grad_B_P_BATCH = self.generate_grad_B_P(h)
            #if not self.batches:
            #    self.grad_B_P[h] = grad_B_P_BATCH
            #else:
            #    self.grad_B_P[h] = np.zeros(grad_B_P_BATCH.shape)
                
                # copy in COLUMNS but swapped around using batch_to_ind
                # oh good gods
                # for every sequential BATCH index
                # oh no wait, don't; take every birth and death time (this is what generate_grad_B_P returns)
            #    for c in range(grad_B_P_BATCH.shape[1]):
                    
                    # divide by 2 and take the floor
                    # this is now the actual sequential batch index
                    # WAAAAAAAAAAAAAIT a second
                    # this is wrong? This is super wrong
                    # we're going by COLUMNS here, not rows
                    # the columns of grad_B_P_BATCH (the object returned by generate_grad_B_P)
                    #     are m*n — every coordinate of every point of the pointcloud
                    #     if I'm trying to get the point itself without coordinate fluff,
                    #     I need to divide by self.embed_dim,
                    #     NOT by 2.0. In fact, this would explain why things were working fine in 2D
                    #     but went to hell in higher dimensions.
                    # THE GUILTY:
                    #nonlinear_c = int(float(c)/2.0)
            #        nonlinear_c = int(float(c)/self.embed_dim)
                    
                    # save the information of if you were odd or not, basically
                    # MORE GUILTY
                    #adjust = c%2
            #        adjust = c%self.embed_dim
                    
                    
                    #if nonlinear_c in self.batch_to_original.keys():
                    
                    # now, for every birth death coordinate of every non-trivial generator ————
                    # OOPS, I WAS PUTTING MINIBATCH STUFF INTO A FULL MATRIX WITHOUT RESETTING COORDINATES
                    #     so I've now fixed that in generate_grad_B_P, so now now NOW,
                    # this goes through every column, i.e., every birth and death coord of the nontrivial source generators
            #        for r in range(grad_B_P_BATCH.shape[0]):
                        # and it says, hey, wait, now that I've changed stuff to make any goddamn sense, I do not NEED this?
            #            self.grad_B_P[h][r][2*self.batch_to_original[nonlinear_c]+adjust] += grad_B_P_BATCH[r][c]
                        
                    #else:
                    #    col_sum = np.sum([grad_B_P_BATCH[r][c] for r in range(grad_B_P_BATCH.shape[0])])
                    #    if col_sum > 0:
                    #        print('I\'ll eat my hat.')
            
            print('grad_W_B_LIN for elucidation')
            print(self.grad_W_B_LIN[h])
            
            print('grad_B_P for elucidation')
            print(self.grad_B_P[h])
            
            self.grad_W_P[h] = np.matmul(np.transpose(self.grad_W_B_LIN[h]),self.grad_B_P[h])
            
            print('grad_W_P for elucidation')
            print(self.grad_W_P[h])
        
            
    def mini_batch(self):
        
        haus = None
        counter = 0

        while (haus is None or haus > self.hausdorff_subsample_threshold) and counter < 100:

            self.batch_indices = np.random.choice(list(range(self.W1.shape[0])),self.minibatch_size,replace=False)

            # now check if this is an ok subsample.
            subdata = pd.DataFrame(self.W1).loc[self.batch_indices]

            haus = scipy.spatial.distance.directed_hausdorff(self.W1,subdata)[0]
            counter += 1

        if haus > self.hausdorff_subsample_threshold and counter >= 100:
            print('****Escaped subsampling loop due to counter. Recommend increasing minibatch size.')

        print('-- generated batch indices for wasserstein')
        #print(self.source_indices)
        
        # you don't put in a batch's index, you put in a SEQUENTIAL number for the batch index you're interested in, and it gives you the sequential index of the original index — so basically, batch to original is pointless lol, but ok
        self.batch_to_original = {i:self.batch_indices[i] for i in range(len(self.batch_indices))}
        self.original_to_batch = {self.batch_indices[i]:i for i in range(len(self.batch_indices))}

        
        # SUBOPTIMAL — subselect pointcloud first and then just use super fast PWDM, bro ****************
        self.target_distance_matrix_BATCH = np.zeros((len(self.batch_indices),len(self.batch_indices)))
        for r in range(self.target_distance_matrix_BATCH.shape[0]):
            for c in range(self.target_distance_matrix_BATCH.shape[1]):
                self.target_distance_matrix_BATCH[r][c] += self.target_distance_matrix[self.batch_to_original[r]][self.batch_to_original[c]]

        for h in self.HD:
            self.acquire_target_cycles(h,batch=True)
