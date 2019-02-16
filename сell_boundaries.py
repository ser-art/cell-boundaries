import numpy as np
import pandas as pd
import scipy as sp

from scipy.stats import multivariate_normal as norm, multinomial, uniform

from scipy.misc import logsumexp

import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression

import time

def to_categorial(x, num_cat):
    arr = np.zeros(shape=(num_cat,))
    arr[x] = 1
    return arr

def norm_density(params, x):
    xy = x[['x', 'y']].values
    return np.log(params['w']) + norm.logpdf(xy, params['mean'], params['cov_matrix'])

def multinom_density(params, x):
    class_ = np.array(list(x['Class'].values))
    return np.log(params['w']) + multinomial.logpmf(class_, n=1, p=params['prob'])

def noise_density(params, x):
    xy = x[['x', 'y']].values
    dens_x = uniform.logpdf(xy[:,0], loc=xy[:,0].min(), scale=xy[:,0].max()-xy[:,0].min())
    dens_y = uniform.logpdf(xy[:,1], loc=xy[:,1].min(), scale=xy[:,1].max()-xy[:,1].min())
    return np.log(params['w']) + dens_x + dens_y

def expect(x, dens_params, densities):
    g = []
    denses = np.array([dens(p, x) for p, dens in zip(dens_params, densities)]).T
    dens_sums = logsumexp(denses, axis=1)
    if np.any(dens_sums < np.log(1e-50)):
        raise Exception('Zero densities exists. Try different params')
    
    if np.any(np.isnan(dens_sums)):
        raise Exception('Densities sums to nans.')
    
    for dens, dens_sum in zip(denses, dens_sums):
        g_i = np.array([d for d in dens]) - dens_sum
        g.append(g_i)
    return np.exp(np.array(g))

def norm_maximize(x, hidden_param):
    xy = x[['x', 'y']].values
    
    params = []
    w_new = hidden_param.mean()
    mean_new = np.mean(xy.T * hidden_param, axis=1) / w_new
        
    sigma_new = np.mean(hidden_param * ((xy - mean_new)**2).T, axis=1) / w_new
    cov_new = np.mean(hidden_param * (xy[:,0] - mean_new[0]) * (xy[:,1] - mean_new[1]), 
                          axis=0)
    
    return {'w' : w_new, 'mean' : mean_new, 'cov_matrix' : np.array([[sigma_new[0], cov_new], [cov_new, sigma_new[1]]])}

def noise_maximize(x, hidden_param):  
    return {'w' : hidden_param.mean()}

def multinom_maximize(x, hidden_param):
    class_ = np.array(list(x['Class'].values))
    
    w_new = hidden_param.mean()

    prob_new = np.array([hidden_param[class_[:,c].astype(bool)].sum() for c in range(x.shape[1])])
    prob_new /= (hidden_param.shape[0] * w_new)
         
    return {'w' : w_new, 'prob' : prob_new}

def EM(x, dens_params, densities, maximize_functions, max_iters=1000, iters_del=4, eps=1e-8):
    ts = time.time()
    g_i_1 = np.zeros(shape=(x.shape[0], len(densities)))
    norm = 1e10
    print('EM fit started')
    for i in range(max_iters):
        g_i = expect(x, dens_params, densities)
        assert np.all(np.isclose(g_i.sum(axis=1), 1)), 'Probablies don\'t sum to 1'
        
        dens_params = []       
        for mix_max_func, mix_g in zip(maximize_functions, g_i.T):
            mix_params = mix_max_func(x, mix_g)
            dens_params.append(mix_params)
        
        norm = np.linalg.norm(g_i - g_i_1)
        if (i + 1) % int(max_iters / iters_del) == 0:
            print('EM part done for {} s in {} iterations. Norm {}'.format(int(time.time() - ts), i + 1, norm))
            
        if i > 0 and norm < eps:
            break
        g_i_1 = g_i
    print('EM part done for {} s in {} iterations. Norm {}'.format(int(time.time() - ts), i + 1, norm))
    
    return np.argmax(g_i, axis=1), dens_params

def plot_clusters(df_spatial, clusters_info, color_cluster=False, fig_size=(9, 9)):
    angles = []

    segments = df_spatial.groupby('Class')
    
    f, ax = plt.subplots(figsize=fig_size)
    for segment, (index_ci, data_ci) in zip(segments, clusters_info.iterrows()):
        lr = LinearRegression()
        lr.fit(segment[1].x.values.reshape((-1, 1)), 
               segment[1].y.values.reshape((-1, 1)))
        angle = np.arctan(lr.coef_[0][0]) * 180/np.pi
        angles.append(angle)
        
        cmap = segment[1].Cmap
        if color_cluster:
            cmap = None
        
        ellipse = Ellipse(xy=(data_ci.MeanX, data_ci.MeanY), width=2*data_ci.StdX, height=2*data_ci.StdY, angle=360 - angle,
                         fill=False, linewidth=1.5)
        ax.add_patch(ellipse)
        plt.scatter(segment[1].x, segment[1].y, s=4, color=cmap)
        plt.plot([data_ci.MeanX], [data_ci.MeanY], marker='*', color='black')
        plt.xlabel('x', fontsize=15)
        plt.ylabel('y', fontsize=15)

    angles = np.array(angles)
    print(angles.mean())