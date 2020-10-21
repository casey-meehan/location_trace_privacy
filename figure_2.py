#Produces plots seen in Figure 2 

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import cvxpy as cp
from tqdm import tqdm
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

from pathlib import Path

import matplotlib as mpl
mpl.rcdefaults()
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('axes', linewidth=3)     #border of figures 
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
plt.rc('legend', fontsize=30)    # legend fontsize
plt.rc('figure', titlesize=30)  # fontsize of the figure title
plt.rc('legend', frameon = False)
plt.rc('lines', linewidth = 4)
plt.rc('lines', markersize = 8)
plt.rc('axes', titlesize = 40)
plt.rc('figure', figsize = (12,8))

def RBF_cov(n_pts, len_scale_sq = None, var = 1): 
    """ returns RBF covariance matrix 
    inputs: 
        - n_pts: number of points of GP 
        - var: \sigma^2 of RBF kernel, defaults to n_pts / 2
    returns: 
        - Sigma: covariance matrix of n_pts X n_pts
    """
    t_pts = np.arange(n_pts)[:,None]
    if len_scale_sq == None: 
        len_scale_sq = n_pts / 2
    Sigma = var * np.exp(- (t_pts - t_pts.T)**2  / (2*len_scale_sq))
    return Sigma

def PER_cov(n_pts, len_scale_sq = None, period = None, var = 1):
    """ returns periodic covariance matrix
    inputs:
        - n_pts: number of points of GP
        - var: \sigma^2 of RBF kernel, defaults to n_pts / 2
    returns:
        - Sigma: covariance matrix of n_pts X n_pts
    """
    t_pts = np.arange(n_pts)[:,None]
    if len_scale_sq == None:
        len_scale_sq = n_pts / 2
    if period == None:
        period = n_pts / 2
    Sigma = var * np.exp(- 2*np.sin( np.pi * np.abs(t_pts - t_pts.T) / period)**2  / (len_scale_sq))
    return Sigma

def get_A_and_B_info(Sigma, s_idx): 
    """gets the matrix A = $\Sigma_{us}\Sigma_{ss}^{-1}$. 
    For matrix B = $(\Sigma_{u|s} + \Sigma_{uu}^{(z)})^{-1}$, returns $\Sigma_{u|s}$. 
    Inputs: 
        - Sigma: n X n covariance matrix of conditional prior
        - s_idx: k-element 1d np array with indices of hypothesized points s
    Returns: 
        - A: |U| X |S| matrix 
        - Sigma_ugs: |U| X |U| matrix 
        
    """
    n_pts = Sigma.shape[0]
    u_idx = np.array([True]*n_pts)
    u_idx[s_idx] = False
    u_idx = np.arange(n_pts)[u_idx]
    s_idx = s_idx[:,None]
    u_idx = u_idx[:,None]
    
    #get submatrices 
    Sig_us = Sigma[u_idx, s_idx.T]
    Sig_ss = Sigma[s_idx, s_idx.T]
    Sig_su = Sigma[s_idx, u_idx.T]
    Sig_uu = Sigma[u_idx, u_idx.T]
    
    #inverse of Sig_ss
    Sig_ss_inv = np.linalg.pinv(Sig_ss)
    
    #get A matrix
    A = Sig_us.dot(Sig_ss_inv)
    
    #get Sigma_ugs
    Sig_ugs = Sig_uu - Sig_us.dot(Sig_ss_inv).dot(Sig_su)
    
    return A, Sig_ugs

def get_Sigma_z(Sigma, s_idx, MSE_max, is_print = False): 
    """Finds noise covariance matrix optimizing privacy loss for secret s_idx
    Inputs: 
        - Sigma: n X n conditional prior covarience matrix
        - s_idx: np array of indices of your secret
        - MSE_max: MSE constraint on noise covariance matrix
        - is_print: boolean determining whether to print status info 
    Outputs: 
        - Sigma_z: u X u block of the noise covariance matrix Sigma_z
        - beta_star: minimum eigenvalue found for A'BA
    """
    #Construct problem data 
    n_pts = np.shape(Sigma)[0]
    n_s = len(s_idx)
    n_u = n_pts - n_s
    u_idx = np.delete(np.arange(n_pts), s_idx)
    A, Sig_ugs = get_A_and_B_info(Sigma, s_idx)

    #augment matrices to n pts 
    A_tilde = np.concatenate((np.eye(n_s), A), axis = 0)
    Sig_ugs_tilde = np.zeros((n_pts, n_pts))
    Sig_ugs_tilde[n_s:, n_s:] = Sig_ugs

    # #get pseudo inverse 
    A_tilde_i, res, _, _ = np.linalg.lstsq(A_tilde, np.eye(n_pts), rcond = None)

    # Build problem 
    B_tilde_i = cp.Variable((n_pts,n_pts),symmetric = True)
    beta_star = cp.Variable(1)

    #Construct constraints 
    constraints = [beta_star >= 0, 
                   A_tilde_i@B_tilde_i@A_tilde_i.T >> beta_star*np.eye(n_s),
                   B_tilde_i >> Sig_ugs_tilde, 
                   cp.trace(B_tilde_i) <= np.trace(Sig_ugs_tilde) + MSE_max]

    #objective
    obj = cp.Maximize(beta_star)

    #create SDP
    prob = cp.Problem(obj, constraints)
    if is_print: 
        print('\nProblem DCP?:', prob.is_dcp())

    #Solve SDP
    _ = prob.solve()
    if is_print: 
        print("status:\n", prob.status)

    Sigma_z = np.zeros((n_pts, n_pts))
    Sigma_z[s_idx, s_idx] = (B_tilde_i.value[np.arange(n_s),np.arange(n_s)]).mean()
    Sigma_z[u_idx[:,None],u_idx[:,None].T] = (B_tilde_i.value - Sig_ugs_tilde)[n_s:, n_s:]
    
    return Sigma_z

#def sig_eff(Sigma, Sigma_z, s_idx):
#    """get effective Sigma for computing Renyi divergence given s indices
#    Inputs:
#        - Sigma: n X n covariance matrix being considered
#        - Sigma_z: n X n noise covariance matrix
#        - s_idx: k-element 1d np array with indices of hypothesized points s
#    Returns:
#        - Sig_eff: k X k effective matrix for computing Renyi divergence loss
#    """
#    n_pts = Sigma.shape[0]
#    u_idx = np.array([True]*n_pts)
#    u_idx[s_idx] = False
#    u_idx = np.arange(n_pts)[u_idx]
#    u_idx = u_idx[:,None]
#
#    A, Sig_ugs = get_A_and_B_info(Sigma, s_idx)
#    Sig_z_uu = Sigma_z[u_idx, u_idx.T]
#    B = np.linalg.pinv(Sig_ugs + Sig_z_uu) 
#    Sig_eff = A.T.dot(B).dot(A)
#    return Sig_eff

#def eig_vecs_vals(Sigma):
#    """Return eigenvectors and eigenvalues of matrix in descending order of magnitude
#    Inputs:
#        - Sigma: n X n matrix
#    Returns:
#        - v: n X n matrix of eigenvecs starting with v[:,0], the largest in magnitude
#        - w: n vector of eigenvals corresponding to each eigenvec
#    """
#    w, v = np.linalg.eig(Sigma)
#    v = v[:,np.argsort(w)[::-1]]
#    w = w[np.argsort(w)[::-1]]
#    return w, v

#def get_priv_loss(Sigma, Sigma_z, s_idx, lam = 5, r = 1): 
#    """Returns the Renyi worst case privacy loss for noise covariance matrix Sigma_z_sq  with
#    conditional 's' points given by s_idx
#    Inputs: 
#        Sigma: Covariance matrix of X points 
#        Sigma_z: Noise covariance matrix  
#        s_idx: indices of points to condition on 
#    Outpus: 
#        R_s_st: Worst case Renyi privacy loss 
#        alph_str: alpha star (inferential privacy loss)
#        ind_loss: loss of the independent noise (GI privacy loss)
#    """
#    #dependent loss
#    Sig_eff = sig_eff(Sigma, Sigma_z, s_idx)
#    w, v = eig_vecs_vals(Sig_eff)
#    alph_str = w[0]
#    
#    #independent loss
#    sig_s_sq = Sigma_z[s_idx, s_idx]
#    ind_loss = (1 / sig_s_sq).sum() / len(s_idx)
#    
#    R_s_st = 2 * lam * len(s_idx) * r**2 * (ind_loss + alph_str)
#    return R_s_st, alph_str, ind_loss

def plot_sweep_data(l_effs, len_scales_sq, SDP_CIs, ISO_uni_CIs, ISO_conc_CIs, fname, title, misspec = False, ylabel = False): 
    
    #Get distribution stats
    quartile1, median, quartile3 = np.percentile(l_effs, [25, 50, 75])
    
    #plot interval
    if ISO_conc_CIs is not None:
        minval = 0.9*np.min(np.concatenate((SDP_CIs, ISO_uni_CIs, ISO_conc_CIs)))
        maxval = 1.1*np.max(np.concatenate((SDP_CIs, ISO_uni_CIs, ISO_conc_CIs)))
    else: 
        minval = 0.9*np.min(np.concatenate((SDP_CIs, ISO_uni_CIs)))
        maxval = 1.1*np.max(np.concatenate((SDP_CIs, ISO_uni_CIs)))
        
    plt.fill_betweenx(np.linspace(minval, maxval, 10), quartile1, quartile3, color = 'gray', alpha = 0.2)
    plt.plot(median * np.ones(10), np.linspace(minval, maxval, 10), '--',color = 'black')
    
    #plot data
    if misspec: 
        plt.plot(np.sqrt(len_scales_sq), np.array(SDP_CIs), '-o', label = 'Good prior')
        plt.plot(np.sqrt(len_scales_sq), np.array(ISO_uni_CIs),'-o', label = 'Over Corr. Prior')
        plt.plot(np.sqrt(len_scales_sq), np.array(ISO_conc_CIs),'-o', label = 'Under Corr. Prior')
        plt.xlabel("Prior Dependence ($l_{eff}$ of Mech.)")
    else: 
        plt.plot(np.sqrt(len_scales_sq), np.array(SDP_CIs), '-o', label = 'SDP')
        plt.plot(np.sqrt(len_scales_sq), np.array(ISO_uni_CIs),'-o', label = 'Ind. Unif.')
        if ISO_conc_CIs is not None: 
            plt.plot(np.sqrt(len_scales_sq), np.array(ISO_conc_CIs),'-o', label = 'Ind. Conc.')
        plt.xlabel("Prior Dependence ($l_{eff}$)")
    plt.legend(loc = 'upper right')
    if ylabel:
        plt.ylabel("Posterior Uncertainty Interval")
    plt.title(title)
    plt.savefig('./images/' + fname, bbox_inches = 'tight', pad_inches = 0)
    plt.show()


#def plot_sweep_data(l_effs, len_scales_sq, SDP_totals, ISO_totals, fname, title):
#    """Plot util for figures 3c) -> h) 
#    """
#    fig, axs = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'hspace': 0, 'height_ratios':[0.3,1]})
#
#    #Get distribution stats
#    quartile1, median, quartile3 = np.percentile(l_effs, [25, 50, 75])
#    lmin = np.sqrt(np.min(len_scales_sq))
#    lmax = np.sqrt(np.max(len_scales_sq))
#
#    #Plot distribution
#    l_effs_filter = (l_effs > lmin) * (l_effs < lmax)
#    l_effs_ = l_effs[l_effs_filter]
#    parts = axs[0].violinplot(l_effs_, vert = False,
#                               showmeans = False, showmedians = False, showextrema=False)
#
#    parts['bodies'][0].set_facecolor('gray')
#    parts['bodies'][0].set_edgecolor('black')
#    parts['bodies'][0].set_alpha(0.7)
#
#    axs[0].hlines(1, quartile1, quartile3, color='k', linestyle='-', lw=10)
#    axs[0].hlines(1, lmin, lmax, color='k', linestyle='-', lw=1)
#    axs[0].scatter(median, 1, color='white', marker = 'o', s=500, zorder=3)
#
#    axs[0].axes.get_yaxis().set_visible(False)
#
#    #plot data
#    axs[1].plot(np.sqrt(len_scales_sq), np.array(SDP_totals), '-o', label = 'SDP')
#    axs[1].plot(np.sqrt(len_scales_sq), np.array(ISO_totals),'-o', label = 'Isotropic')
#    plt.legend()
#    plt.xlabel("$l_{eff}$")
#    plt.ylabel("$L_{priv}$")
#    axs[0].set_title(title)
#    plt.savefig('./images/' + fname, bbox_inches = 'tight', pad_inches = 0)
#    plt.show()

def get_super_sigma_z(Sigma, secrets, MSE_max, is_print = False):
    """Finds an optimal noise covariance matrix for each listed secret
    and solves for a minimum trace noise covariance that maintains the privacy of each 
    Inputs:  
        -Sigma: model covariance 
        -secrets: list of numpy array secrets to protect 
        -MSE_max: the maximum trace of the individual secret covariance matrices
    Outputs: 
        -Sigma_z: the final covariance matrix 
        -Sigma_zs: the individual secret-optimal covariance matrices
    """
    n_pts = np.shape(Sigma)[0]
    #list of individual secret matrices
    Sigma_zs = []
    #get individual Sigma_z's 
    for s_idx in secrets: 
        Sigma_zs.append(get_Sigma_z(Sigma, s_idx, MSE_star, is_print=is_print))
    
    #Run program to pick final Sigma_z
    # Build problem 
    Sigma_z = cp.Variable((n_pts,n_pts),symmetric = True)

    #Construct constraints 
    constraints = [Sigma_z >> Sigma_zs[i] for i in range(len(secrets))]

    #objective
    obj = cp.Minimize(cp.trace(Sigma_z))

    #create SDP
    prob = cp.Problem(obj, constraints)
    if is_print: 
        print('\nProblem DCP?:', prob.is_dcp())

    #Solve SDP
    _ = prob.solve()
    if is_print: 
        print("status:\n", prob.status)

    Sigma_z = Sigma_z.value
    return Sigma_z, Sigma_zs

def get_posterior_cov(Sigma, Sigma_z):
    """
    return the normal posterior distribution of X values
    Inputs:
        - Sigma: covariance matrix of X
        - Sigma_z: covariance matrix of mechanism (G)
    Returns:
        - Sigma_xgz: posterior covariance of X_s
    """
    #get inverses (need the 1e-6 for stability)
    Sigma_i = np.linalg.pinv(Sigma + 1e-6 * np.mean(Sigma.diagonal()) * np.eye(len(Sigma)))
    Sigma_z_i = np.linalg.pinv(Sigma_z + 1e-6 * np.mean(Sigma_z.diagonal()) * np.eye(len(Sigma_z)))

    #posterior covariance of all Xs
    Sigma_xgz = np.linalg.pinv(Sigma_i + Sigma_z_i)

    return Sigma_xgz

#def get_multi_secret_loss(Sigma, Sigma_z, secrets, is_print = False): 
#    #Look at privacy loss and compare with isotropic
#    """Looks at the privacy loss due to direct and inferential loss for 
#    covariance matrix Sigma_z for each secret and compares with isotropic noise of same trace 
#    Inputs: 
#        -Sigma: data covariance matrix 
#        -Sigma_z: noise covariance matrix to compare
#        -secrets: list of secrets to check privacy loss for
#        -is_print: print problem status
#    Returns: 
#        -total_SDP: total privacy loss for the SDP
#        -total ISO: total privacy loss for isotropic noise 
#    """
#    n_pts = Sigma.shape[0]
#    alph_star_sdp = []
#    dir_loss_sdp = []
#
#    alph_star_iso = []
#    dir_loss_iso = []
#
#    Sigma_z_iso = (np.trace(Sigma_z) / n_pts) * np.eye(n_pts)
#
#    for s_idx in secrets: 
#        if is_print: 
#            print('Secret:', s_idx)
#        _, alph_star, dir_loss = get_priv_loss(Sigma, Sigma_z, s_idx)
#        alph_star_sdp.append(alph_star)
#        dir_loss_sdp.append(dir_loss)
#        if is_print:
#            print('SDP Total:', alph_star + dir_loss, 'alph:', alph_star, 'direct:', dir_loss)
#
#        _, alph_star, dir_loss = get_priv_loss(Sigma, Sigma_z_iso, s_idx)
#        alph_star_iso.append(alph_star)
#        dir_loss_iso.append(dir_loss)
#        if is_print:
#            print('ISO Total:', alph_star + dir_loss, 'alph:', alph_star, 'direct:', dir_loss)
#    total_SDP = np.sum(alph_star_sdp) + np.sum(dir_loss_sdp)
#    total_ISO = np.sum(alph_star_iso) + np.sum(dir_loss_iso)
#    if is_print:
#        print('\nTotal SDP:', np.sum(alph_star_sdp) + np.sum(dir_loss_sdp))
#        print('\nTotal ISO:', np.sum(alph_star_iso) + np.sum(dir_loss_iso))
#    return total_SDP, total_ISO

#def plot_sweep_iso_data(l_effs, len_scales_sq, noise_vars, L_privs, fname, title): 
#    """plot utility for figures 3a) -> 3b) 
#    """
#    fig, axs = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'hspace': 0, 'height_ratios':[0.3,1]})
#    
#    #Get distribution stats
#    quartile1, median, quartile3 = np.percentile(l_effs, [25, 50, 75])
#    lmin = np.sqrt(np.min(len_scales_sq))
#    lmax = np.sqrt(np.max(len_scales_sq))
#    
#    #Plot distribution
#    l_effs_filter = (l_effs > lmin) * (l_effs < lmax)
#    l_effs_ = l_effs[l_effs_filter]
#    parts = axs[0].violinplot(l_effs_, vert = False, 
#                               showmeans = False, showmedians = False, showextrema=False)
#    
#    parts['bodies'][0].set_facecolor('gray')
#    parts['bodies'][0].set_edgecolor('black')
#    parts['bodies'][0].set_alpha(0.7)
#    
#    axs[0].hlines(1, quartile1, quartile3, color='k', linestyle='-', lw=10)
#    axs[0].hlines(1, lmin, lmax, color='k', linestyle='-', lw=1)
#    axs[0].scatter(median, 1, color='white', marker = 'o', s=500, zorder=3)
#    
#    axs[0].axes.get_yaxis().set_visible(False)
#    
#    #plot data
#    for nv_idx in range(len(noise_vars)): 
#        p = axs[1].plot(np.sqrt(len_scales_sq), L_privs[:, nv_idx], label = 'MSE = {0:n}%'.format(100 * noise_vars[nv_idx]))
#        #plot baseline
#        axs[1].plot(np.sqrt(len_scales_sq), inds[:, nv_idx], '--', color = p[0].get_color(),
#                linewidth = 3)
#
#    plt.legend()
#    plt.xlabel("$l_{eff}$")
#    plt.ylabel("$L_{priv}$")
#    axs[0].set_title('Iso. Mech\'s: RBF Basic Secret')
#
#    plt.savefig('./images/' + fname, bbox_inches = 'tight', pad_inches = 0)
#    plt.show()

#Make figure directory 
Path("./images").mkdir(parents=True, exist_ok=True)

#load l_eff data for RBF (location) 
l_eff_x = np.load('./saved_data/l_eff_x.npy')
l_eff_y = np.load('./saved_data/l_eff_y.npy')
l_effs = np.concatenate((l_eff_x, l_eff_y), axis = 0)

#load l_eff data for Periodic (home temperature) 
l_eff_temp = np.load('./saved_data/l_eff_temp.npy') 

#which figures to make
two_a = True 
two_b = True 
two_c = True
two_d = True 
two_e = True 
two_f = True  
two_g = True 
two_h = True 


#################
### FIGURE 2A ###
#################
if two_a == True: 
    print('Making Figure 2A')
    n_pts = 50
    s_idx = np.array([24])
    len_scales_sq = np.linspace(1, 10**2, 20)
    stab_factor = 10
    MSE_star = stab_factor * n_pts * 0.02
    SDP_CIs = []
    ISO_uni_CIs = []
    ISO_conc_CIs = []
    for l in tqdm(len_scales_sq): 
        Sigma = RBF_cov(n_pts, len_scale_sq = l, var = 1)
        Sigma_z = get_Sigma_z(Sigma, s_idx, MSE_star, is_print = False) / stab_factor
        Sigma_xgz = get_posterior_cov(Sigma, Sigma_z)
        CI = Sigma_xgz[s_idx, s_idx]
        SDP_CIs.append(CI)
        
        Sigma_z_iso_uni = (np.trace(Sigma_z)/n_pts) * np.eye(n_pts)
        Sigma_xgz_iso_uni = get_posterior_cov(Sigma, Sigma_z_iso_uni)
        CI_iso_uni = Sigma_xgz_iso_uni[s_idx, s_idx]
        ISO_uni_CIs.append(CI_iso_uni)
        
        Sigma_z_iso_conc = np.zeros((n_pts, n_pts))
        Sigma_z_iso_conc[s_idx, s_idx] = np.trace(Sigma_z) / len(s_idx)
        Sigma_xgz_iso_conc = get_posterior_cov(Sigma, Sigma_z_iso_conc)
        CI_iso_conc = Sigma_xgz_iso_conc[s_idx, s_idx]
        ISO_conc_CIs.append(CI_iso_conc)

    plot_sweep_data(l_effs, len_scales_sq, 2*np.sqrt(np.array(SDP_CIs)), 2*np.sqrt(np.array(ISO_uni_CIs)), 2*np.sqrt(np.array(ISO_conc_CIs)), 'figure_2a.png', 'RBF Basic Secret', ylabel = True)

#################
### FIGURE 2B ###
#################
if two_b == True: 
    n_pts = 50
    s_idx = np.array([24, 25])
    len_scales_sq = np.linspace(1, 10**2, 20)
    stab_factor = 1000
    MSE_star = stab_factor * n_pts * 0.02
    SDP_CIs = []
    ISO_uni_CIs = []
    ISO_conc_CIs = []
    for l in tqdm(len_scales_sq): 
        Sigma = RBF_cov(n_pts, len_scale_sq = l, var = 1)
        Sigma_z = get_Sigma_z(Sigma, s_idx, MSE_star, is_print = False) / stab_factor
        Sigma_xgz = get_posterior_cov(Sigma, Sigma_z)[s_idx[:,None], s_idx[None,:]]
        CI = np.min(np.abs(np.linalg.eigvals(Sigma_xgz)))
        SDP_CIs.append(CI)
        
        Sigma_z_iso_uni = (np.trace(Sigma_z)/n_pts) * np.eye(n_pts)
        Sigma_xgz_iso_uni = get_posterior_cov(Sigma, Sigma_z_iso_uni)[s_idx[:,None], s_idx[None,:]]
        CI_iso_uni = np.min(np.abs(np.linalg.eigvals(Sigma_xgz_iso_uni)))
        ISO_uni_CIs.append(CI_iso_uni)
        
        Sigma_z_iso_conc = np.zeros((n_pts, n_pts))
        Sigma_z_iso_conc[s_idx, s_idx] = np.trace(Sigma_z) / len(s_idx)
        Sigma_xgz_iso_conc = get_posterior_cov(Sigma, Sigma_z_iso_conc)[s_idx[:,None], s_idx[None,:]]
        CI_iso_conc = np.min(np.abs(np.linalg.eigvals(Sigma_xgz_iso_conc)))
        ISO_conc_CIs.append(CI_iso_conc)

    plot_sweep_data(l_effs, len_scales_sq, 2*np.sqrt(np.array(SDP_CIs)), 2*np.sqrt(np.array(ISO_uni_CIs)), 2*np.sqrt(np.array(ISO_conc_CIs)), 'figure_2b.png', 'RBF Compound Secret')

#################
### FIGURE 2C ###
#################
if two_c == True: 
    print('Making Figure 2C')
    n_pts = 50
    len_scales_sq = np.linspace(1, 10**2, 20)
    SDP_totals = []
    ISO_totals = []
    stab_factor = 1e4
    secrets = [np.array([i]) for i in range(n_pts)]
    MSE_star = stab_factor * n_pts * 0.02 #/ len(secrets) #trace of each single basic secret covariance mat
    SDP_CIs = []
    ISO_uni_CIs = []
    
    for l in tqdm(len_scales_sq): 
        Sigma = RBF_cov(n_pts, len_scale_sq = l, var = 1)
        Sigma_z, Sigma_zs = get_super_sigma_z(Sigma, secrets, MSE_star)
        Sigma_z = Sigma_z / stab_factor #renormalize trace to targeted noise level
        Sigma_xgz = get_posterior_cov(Sigma, Sigma_z)
        CI = np.mean(Sigma_xgz.diagonal())
        SDP_CIs.append(CI)
        
        Sigma_z_iso_uni = (np.trace(Sigma_z)/n_pts) * np.eye(n_pts)
        Sigma_xgz_iso_uni = get_posterior_cov(Sigma, Sigma_z_iso_uni)
        CI_iso_uni = np.mean(Sigma_xgz_iso_uni.diagonal())
        ISO_uni_CIs.append(CI_iso_uni)

    plot_sweep_data(l_effs, len_scales_sq, 2*np.sqrt(np.array(SDP_CIs)), 2*np.sqrt(np.array(ISO_uni_CIs)), None, 'figure_2c.png', 'RBF All Basic Secrets')

#################
### FIGURE 2D ###
#################
if two_d == True: 
    print('Making Figure 2D')
    n_pts = 50
    s_idx = np.array([24])
    len_scales_sq = np.linspace(1, 10**2, 20)
    stab_factor = 10
    MSE_star = stab_factor * n_pts * 0.02
    Proper_CIs = []
    lm1_CIs = []
    lp1_CIs = []
    for l in tqdm(len_scales_sq): 
        Sigma = RBF_cov(n_pts, len_scale_sq = l, var = 1)
        Sigma_z = get_Sigma_z(Sigma, s_idx, MSE_star, is_print = False) / stab_factor
        Sigma_xgz = get_posterior_cov(Sigma, Sigma_z)
        CI = Sigma_xgz[s_idx, s_idx]
        Proper_CIs.append(CI)
        
        #1.5lenscale
        Sigma_p1 = RBF_cov(n_pts, len_scale_sq = (1.5*np.sqrt(l))**2, var = 1)
        Sigma_xgz_p1 = get_posterior_cov(Sigma_p1, Sigma_z)
        CI_p1 = Sigma_xgz_p1[s_idx, s_idx]
        lp1_CIs.append(CI_p1)
        
        #0.5lenscale 
        Sigma_m1 = RBF_cov(n_pts, len_scale_sq = (0.5*np.sqrt(l))**2, var = 1)
        Sigma_xgz_m1 = get_posterior_cov(Sigma_m1, Sigma_z)
        CI_m1 = Sigma_xgz_m1[s_idx, s_idx]
        lm1_CIs.append(CI_m1)

    plot_sweep_data(l_effs, len_scales_sq, 2*np.sqrt(np.array(Proper_CIs)), 2*np.sqrt(np.array(lm1_CIs)), 2*np.sqrt(np.array(lp1_CIs)), 'figure_2d.png', 'RBF Misspec. Prior', misspec = True)
    
#################
### FIGURE 2E ###
#################
if two_e == True: 
    print('Making Figure 2E') 
    n_pts = 48
    len_scales_sq = np.linspace(0.5**2, 1.5**2, 20)
    SDP_CIs = []
    ISO_uni_CIs = []
    ISO_conc_CIs = []
    s_idx = np.array([24])
    stab_factor = 10
    MSE_star = stab_factor * n_pts * 0.02
    for l in tqdm(len_scales_sq): 
        Sigma = PER_cov(n_pts, period = 24, len_scale_sq = l, var = 1)
        Sigma_z = get_Sigma_z(Sigma, s_idx, MSE_star, is_print = False) / stab_factor
        Sigma_xgz = get_posterior_cov(Sigma, Sigma_z)
        CI = Sigma_xgz[s_idx, s_idx]
        SDP_CIs.append(CI)
        
        Sigma_z_iso_uni = (np.trace(Sigma_z)/n_pts) * np.eye(n_pts)
        Sigma_xgz_iso_uni = get_posterior_cov(Sigma, Sigma_z_iso_uni)
        CI_iso_uni = Sigma_xgz_iso_uni[s_idx, s_idx]
        ISO_uni_CIs.append(CI_iso_uni)
        
        Sigma_z_iso_conc = np.zeros((n_pts, n_pts))
        Sigma_z_iso_conc[s_idx, s_idx] = np.trace(Sigma_z) / len(s_idx)
        Sigma_xgz_iso_conc = get_posterior_cov(Sigma, Sigma_z_iso_conc)
        CI_iso_conc = Sigma_xgz_iso_conc[s_idx, s_idx]
        ISO_conc_CIs.append(CI_iso_conc)

    plot_sweep_data(l_eff_temp, len_scales_sq, 2*np.sqrt(np.array(SDP_CIs)), 2*np.sqrt(np.array(ISO_uni_CIs)), 2*np.sqrt(np.array(ISO_conc_CIs)), 'figure_2e.png', 'PER Basic Secret', ylabel = True)

#################
### FIGURE 3F ###
#################
if two_f == True: 
    print('Making Figure 2F')
    n_pts = 48
    len_scales_sq = np.linspace(0.5**2, 1.5**2, 20)
    SDP_CIs = []
    ISO_uni_CIs = []
    ISO_conc_CIs = []
    s_idx = np.array([16,32])
    stab_factor = 10
    MSE_star = stab_factor * n_pts * 0.02
    for l in tqdm(len_scales_sq): 
        Sigma = PER_cov(n_pts, period = 24, len_scale_sq = l, var = 1)
        Sigma_z = get_Sigma_z(Sigma, s_idx, MSE_star, is_print = False) / stab_factor
        Sigma_xgz = get_posterior_cov(Sigma, Sigma_z)[s_idx[:,None], s_idx[None,:]]
        CI = np.min(np.abs(np.linalg.eigvals(Sigma_xgz)))
        SDP_CIs.append(CI)
        
        Sigma_z_iso_uni = (np.trace(Sigma_z)/n_pts) * np.eye(n_pts)
        Sigma_xgz_iso_uni = get_posterior_cov(Sigma, Sigma_z_iso_uni)[s_idx[:,None], s_idx[None,:]]
        CI_iso_uni = np.min(np.abs(np.linalg.eigvals(Sigma_xgz_iso_uni)))
        ISO_uni_CIs.append(CI_iso_uni)
        
        Sigma_z_iso_conc = np.zeros((n_pts, n_pts))
        Sigma_z_iso_conc[s_idx, s_idx] = np.trace(Sigma_z) / len(s_idx)
        Sigma_xgz_iso_conc = get_posterior_cov(Sigma, Sigma_z_iso_conc)[s_idx[:,None], s_idx[None,:]]
        CI_iso_conc = np.min(np.abs(np.linalg.eigvals(Sigma_xgz_iso_conc)))
        ISO_conc_CIs.append(CI_iso_conc)

    plot_sweep_data(l_eff_temp, len_scales_sq, 2*np.sqrt(np.array(SDP_CIs)), 2*np.sqrt(np.array(ISO_uni_CIs)), 2*np.sqrt(np.array(ISO_conc_CIs)), 'figure_2f.png', 'PER Compound Secret')


#################
### FIGURE 3G ###
#################
if two_g == True: 
    print('Making Figure 2G')
    n_pts = 48
    len_scales_sq = np.linspace(0.5**2, 1.5**2, 20)
    stab_factor = 1e4
    secrets = [np.array([i]) for i in range(n_pts)]
    MSE_star = stab_factor * n_pts * 0.02 #/ len(secrets) #trace of each single basic secret covariance mat
    SDP_CIs = []
    ISO_uni_CIs = []
    
    for l in tqdm(len_scales_sq):
        Sigma = PER_cov(n_pts, period = 24, len_scale_sq = l, var = 1)
        Sigma_z, Sigma_zs = get_super_sigma_z(Sigma, secrets, MSE_star)
        Sigma_z = Sigma_z / stab_factor #renormalize
        Sigma_xgz = get_posterior_cov(Sigma, Sigma_z)
        CI = np.mean(Sigma_xgz.diagonal())
        SDP_CIs.append(CI)
    
        Sigma_z_iso_uni = (np.trace(Sigma_z)/n_pts) * np.eye(n_pts)
        Sigma_xgz_iso_uni = get_posterior_cov(Sigma, Sigma_z_iso_uni)
        CI_iso_uni = np.mean(Sigma_xgz_iso_uni.diagonal())
        ISO_uni_CIs.append(CI_iso_uni)

    plot_sweep_data(l_eff_temp, len_scales_sq, 2*np.sqrt(np.array(SDP_CIs)), 2*np.sqrt(np.array(ISO_uni_CIs)), None, 'figure_2g.png', 'PER All Basic Secrets')


#################
### FIGURE 3H ###
#################
if two_h == True: 
    print('Making Figure 2H')
    n_pts = 48
    s_idx = np.array([24])
    len_scales_sq = np.linspace(0.5**2, 1.5**2, 20)
    stab_factor = 10
    MSE_star = stab_factor * n_pts * 0.02
    Proper_CIs = []
    lm1_CIs = []
    lp1_CIs = []
    for l in tqdm(len_scales_sq):
        Sigma = PER_cov(n_pts, period = 24, len_scale_sq = l, var = 1)
        Sigma_z = get_Sigma_z(Sigma, s_idx, MSE_star, is_print = False) / stab_factor
        Sigma_xgz = get_posterior_cov(Sigma, Sigma_z)
        CI = Sigma_xgz[s_idx, s_idx]
        Proper_CIs.append(CI)
    
        #lenscale + 1
        Sigma_p1 = PER_cov(n_pts, len_scale_sq = (1.5*np.sqrt(l))**2, var = 1)
        Sigma_xgz_p1 = get_posterior_cov(Sigma_p1, Sigma_z)
        CI_p1 = Sigma_xgz_p1[s_idx, s_idx]
        lp1_CIs.append(CI_p1)
    
        #lenscale - 1
        Sigma_m1 = PER_cov(n_pts, len_scale_sq = (0.5*np.sqrt(l))**2, var = 1)
        Sigma_xgz_m1 = get_posterior_cov(Sigma_m1, Sigma_z)
        CI_m1 = Sigma_xgz_m1[s_idx, s_idx]
        lm1_CIs.append(CI_m1)

    plot_sweep_data(l_eff_temp, len_scales_sq, 2*np.sqrt(np.array(Proper_CIs)), 2*np.sqrt(np.array(lm1_CIs)), 2*np.sqrt(np.array(lp1_CIs)), 'figure_2h.png', 'PER Misspec. Prior', misspec = True)
