"""
Inferring Brownian motion parameters with maximum likelihood estimation

Note: adjust path/file name in section "import track data" and variables in last 
    section "getting output of whole script" as desired
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import i0e                # for Bessel function
from timeit import default_timer as timer
import pandas as pd
from scipy.stats import multivariate_normal, norm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')

#%% import track data

track_df = pd.read_csv('G337_manualtracks/G337_3_frames_merged.csv')

#%% Computing displacement and acceleration of tracked cells
    
def displacement(track_df, minust_frame, zero_frame, t):
    """
    
    returns (2 x numberofcells) matrix containing displacements of tracked cells
    between frame 'zero_frame' and frame 'minust_frame'.
        - row 0: cell displacement dimension x
        - row 1: cell displacement dimension y 

    """
    
    # sort dataframe at t=0 according to track and target id
    track_df_zero = track_df.loc[track_df['FRAME'] == zero_frame].sort_values(['TRACK_ID', 'TARGET_ID'], ascending=[True, True])
    originids = track_df_zero['ORIGIN_ID'].values.tolist()              # origin ids of cells at t=0

    # extract x and y coordiantes as a matrix
    xy_zero = track_df_zero[['POSITION_X_EXACT', 'POSITION_Y_EXACT']].values.T
    xy_minust = np.array( [ np.array([track_df.loc[track_df['SOURCE_ID'] == ID].POSITION_X_EXACT.values[0] for ID in originids ]),
                            np.array([track_df.loc[track_df['SOURCE_ID'] == ID].POSITION_Y_EXACT.values[0] for ID in originids ]) ]
                        )
    
    return (xy_zero - xy_minust) / t

def acceleration(track_df, zero_frame, t):
    """
    
    returns (2 x numberofcells) matrix containing acceleration of tracked cells
    between frame 'zero_frame' + 1 and frame 'minust_frame'.
        - row 0: cell acceleration dimension x
        - row 1: cell acceleration dimension y 

    """
    # x and y coordiantes at t=0
    xy_zero = track_df.loc[track_df['FRAME'] == zero_frame][['POSITION_X_EXACT', 'POSITION_Y_EXACT']].values.T
    sourceids = track_df['SOURCE_ID'].loc[track_df['FRAME'] == zero_frame].values.tolist()        # source ids of spots at t=0
    targetids = track_df.loc[track_df['SOURCE_ID'].isin(sourceids), 'TARGET_ID'].values.tolist()  # target ids of spots at t=0
    originids = track_df.loc[track_df['SOURCE_ID'].isin(sourceids), 'ORIGIN_ID'].values.tolist()  # origin ids of cells at t=0

    # x and y coordiantes at t=t
    xy_t = track_df.loc[track_df['SOURCE_ID'].isin(targetids)][['POSITION_X_EXACT', 'POSITION_Y_EXACT']].values.T
    
    # x and y coodrinates at t=-t
    xy_minust = np.array( [ np.array([track_df.loc[track_df['SOURCE_ID'] == ID].POSITION_X_EXACT.values[0] for ID in originids ]),
                            np.array([track_df.loc[track_df['SOURCE_ID'] == ID].POSITION_Y_EXACT.values[0] for ID in originids ]) ]
                        )
    
    return (xy_t + xy_minust - 2*xy_zero) / (t)**1.5

#%% Find optimal motion model parameters

def findbtrackmotionparameters(displ, accel, t) :
    """ 
    extracts drag (beta) and diffusion (D) from displacemts and accelerations 
    with maximum likelihood estimation
    """
    
    # set data:
    time1 = timer()                                 # to measure time-consumption
    nobootstrapsamples = 1000                       # number of bootstrap samples
    nocells = displ.shape[1]                        # number of cells
    dim = displ.shape[0]                            # number of spatial dimensions
    displ_abs = np.linalg.norm(displ, axis=0)       # absolute displacements
    accel_abs = np.linalg.norm(accel, axis=0)       # absolute accelerations
    beta_hist = np.zeros(nobootstrapsamples); sigma_mot_hist = np.zeros(nobootstrapsamples)         # initialise
    beta_abs_hist = np.zeros(nobootstrapsamples); sigma_mot_abs_hist = np.zeros(nobootstrapsamples) # initialise
    print( " Info - findbtrackmotionparameters: Use %d bootstrap samples." %(nobootstrapsamples) )
    print( " Info - findbtrackmotionparameters: Start looking for best motion parameter now, for %d cells, %d dimensions (after %1.3f sec)." %(nocells,dim,timer()-time1) )
    
    for j_boot in range(0,nobootstrapsamples) :
        # get bootstrap data:
        choice_here = np.ceil( nocells*np.random.rand(nocells)) - 1; choice_here = choice_here.astype(int)    # "-1" due to Python index-notation (peter option)
        #choice_here = np.random.randint(0, nocells, nocells)
        displ_here = displ[:,choice_here];    accel_here = accel[:,choice_here];    displ_abs_here = displ_abs[choice_here];    accel_abs_here = accel_abs[choice_here]
        # work on full vector displacements/accelerations:
        # ...get optimal parameters:
        minimiseme = ( lambda x: -logtargetforbestsigmamot( x[0],displ_here,accel_here, t )[0] )
        cov_here = np.cov(np.reshape(displ_here,dim*nocells),np.reshape(accel_here,dim*nocells)) 
        beta_init = (3/4)*cov_here[1,1]/cov_here[0,0]     # assumes small deltat
        minresult = minimize( minimiseme, beta_init, method='nelder-mead' )          # use COBYLA or TNC as alternative optimizing method
        beta_best = minresult.x[0]; (minbest,sigma_mot_best) = logtargetforbestsigmamot(beta_best, displ_here, accel_here, t)
        # ...copy into memory:
        beta_hist[j_boot] = beta_best;  sigma_mot_hist[j_boot] = sigma_mot_best  
        
        if( dim==2 ) :
            # ...get optimal parameters:
            minimiseme = ( lambda x: -logtarget_abs( x[0],x[1], displ_abs_here,accel_abs_here, t ) )
            cov_here = np.zeros((2,2)); cov_here[0,0] = np.mean(displ_abs_here**2)/dim; cov_here[1,1] = np.mean(accel_abs_here**2)/dim; sigma_mot_abs_init = np.sqrt((3/4)*cov_here[1,1]); beta_abs_init = (sigma_mot_abs_init**2)/cov_here[0,0]    # assumes small deltat
            minresult = minimize( minimiseme, [beta_abs_init,sigma_mot_abs_init], method='nelder-mead' )
            beta_abs_best = minresult.x[0]; sigma_mot_abs_best = minresult.x[1]
            # ...copy into memory:
            beta_abs_hist[j_boot] = beta_abs_best;  sigma_mot_abs_hist[j_boot] = sigma_mot_abs_best
    
    # command-window output:
    print( " Info - findbtrackmotionparameters: Best motion parameter found after %1.3f sec:" %(timer()-time1) )
    print( "  Equations of motion:" )
    print( "   dx = v*dt" )
    print( "   dv = -beta*v*dt + sqrt(2)*sigma_mot*xi" )
    print( "   (for xi: standard Brownian motion, xi(t)-xi(0) ~ N(0,t))" )
    print( "   (    D:  (sigma_mot/beta)^2)" )
    print( "  On full vectors:" )
    print( "   beta:       %1.5e +- %1.5e" %( np.mean(beta_hist),np.std(beta_hist) ) )
    print( "   sigma_mot:  %1.5e +- %1.5e" %( np.mean(sigma_mot_hist),np.std(sigma_mot_hist) ) )
    print( "   D:          %1.5e +- %1.5e" %( np.mean((sigma_mot_hist/beta_hist)**2),np.std((sigma_mot_hist/beta_hist)**2) ) )
    if( dim==2 ) :
        print( "  On absolute values:" )
        print( "   beta:       %1.5e +- %1.5e" %( np.mean(beta_abs_hist),np.std(beta_abs_hist) ) )
        print( "   sigma_mot:  %1.5e +- %1.5e" %( np.mean(sigma_mot_abs_hist),np.std(sigma_mot_abs_hist) ) )
        print( "   D:          %1.5e +- %1.5e" %( np.mean((sigma_mot_abs_hist/beta_abs_hist)**2),np.std((sigma_mot_abs_hist/beta_abs_hist)**2) ) )
    else :
        print( " Warning - findbtrackmotionparameters: 'abs' values not valid, as dim=%d." %(dim) )
    
    return np.mean(beta_hist), np.mean(sigma_mot_hist), np.mean((sigma_mot_abs_hist/beta_abs_hist)**2)


def getvariance( beta, sigma_mot, deltat ) :
    # computes variance of joint displacement-acceleration distribution
    # set auxiliary parameters:
    resctime = beta*deltat
    term1 = (1 + 2*resctime - (2-np.exp(-resctime))**2)/(resctime**2)
    term2 = ((1-np.exp(-resctime))/resctime)**2
    # output:
    return np.array([[ term1+term2, -term1/np.sqrt(deltat) ],[-term1/np.sqrt(deltat), 2*term1/deltat]])*((sigma_mot**2)/beta)

def logtargetforbestsigmamot( beta, displ, accel, deltat ) :
    # computes logtarget of given beta and data
    
    # get auxiliary parameters:
    myvar = getvariance( beta, 1.0, deltat )            # variance up to sigma_mot
    myprec = np.linalg.inv(myvar)                       # precision up to sigma_mot
    nocells = displ.shape[1]                            # number of cells in data
    dim = displ.shape[0]                                # spatial dimensions
    dadim = 2                                           # always 2 (one for displ, one for accel)
    
    allvalues = np.zeros( (dim,nocells) ); allvalues[:,:] = float('nan')
    # allvalues = np.full((dim, nocells), np.nan)
    for j_dim in range(0,dim) :
        allvalues[j_dim,:] = [ np.matmul([displ[j_dim, j_cell],accel[j_dim,j_cell]],np.matmul(myprec,[displ[j_dim,j_cell],accel[j_dim,j_cell]])) for j_cell in range(0,nocells) ]
    sigma2_mot_best = np.mean(allvalues)/dadim          # square of sigma_mot_best
    myvar_best = myvar*sigma2_mot_best                  # variance including sigma_mot
    logtarget = ( -np.log(np.linalg.det(myvar_best)) - dadim )*(dim*nocells/2)
    return logtarget, np.sqrt(sigma2_mot_best)

def logtarget( beta,sigma_mot, displ,accel, deltat ) :
    # computes logtarget for given beta,sigma_mot and data
    
    # get auxiliary parameters:
    myvar = getvariance( beta, sigma_mot, deltat )      # variance
    myprec = np.linalg.inv(myvar)                       # precision
    nocells = displ.shape[1]                            # number of cells in data
    dim = displ.shape[0]                                # spatial dimensions
    
    allvalues = np.zeros( (dim,nocells) )
    for j_dim in range(0,dim) :
        allvalues[j_dim,:] = [ np.matmul([displ[j_dim, j_cell],accel[j_dim,j_cell]],np.matmul(myprec,[displ[j_dim,j_cell],accel[j_dim,j_cell]])) for j_cell in range(0,nocells) ]
    
    logtarget = (-1/2)*np.sum(allvalues) - np.log(np.linalg.det(myvar))*(dim*nocells/2)
    return logtarget

def logtarget_abs( beta_abs,sigma_mot_abs, displ_abs,accel_abs, deltat ) :
    # computes logtarget for given beta_abs, sigma_mot_abs and data_abs
    
    # get auxiliary parameters:
    myvar = getvariance( beta_abs,sigma_mot_abs,deltat )# variance
    myprec = np.linalg.inv(myvar)                       # precision
    nocells = displ_abs.shape[0]                        # number of cells in data
    
    allvalues = [ np.log(np.linalg.det(myprec)) + (-1/2)*(myprec[0,0]*(displ_abs[j_cell]**2)+myprec[1,1]*(accel_abs[j_cell]**2)) + np.log(i0e(myprec[0,1]*displ_abs[j_cell]*accel_abs[j_cell])) + np.abs((myprec[0,1]*displ_abs[j_cell]*accel_abs[j_cell])) + np.log(displ_abs[j_cell]*accel_abs[j_cell]) for j_cell in range(0,nocells) ]
    return np.sum(allvalues)

def joint_covariance(beta, sigma_mot):
    """
    analytical covariance expression of joint normal distribution

    """
    # equation (3.10) in pdf "about how to extract.."
    covar = ( np.array([ [ (1-np.exp(-beta*t)**2)/(beta*t), 0], [0, 0] ]) \
        + np.array([ [1, -1/np.sqrt(t)], [-1/np.sqrt(t), 2/t] ]) \
        * ( (1 + 2*beta*t - (2-np.exp(-beta*t))**2) / (beta*t)**2 ) ) \
        * (sigma_mot**2)/beta
        
    return covar
            
#%% Visual Analysis of parameter estimates

def plots_2d(joint_mean, joint_covar):
    """
    2D joint probability distribution and pooled empirical displ-accel pairs

    """
    
    # computing joint probability for each combi of accel and displ
    displ_range = np.linspace(min(displ.flatten())-1, max(displ.flatten())+1, num=100)
    accel_range = np.linspace(min(accel.flatten())-1, max(accel.flatten())+1, num=100)
    displ_grid, accel_grid = np.meshgrid(displ_range, accel_range)      # grid with combinations of displ and accel
    joint_prob = multivariate_normal.pdf(np.dstack((displ_grid, accel_grid)), mean=joint_mean, cov=joint_covar)
    
    # 2D Contour plot of joint probability / relative frequency
    fig = plt.figure(figsize=(5,4))
    contour = plt.contourf(displ_grid, accel_grid, joint_prob, cmap='plasma')
    plt.scatter(displ.flatten(), accel.flatten(), c='white', alpha=0.5, label='Empirical D-A pairs')
    plt.xlabel('Displacement')
    plt.ylabel('Acceleration')
    plt.title('Joint Probability Distribution')
    plt.grid(True)
    plt.legend()
    
    cbar = fig.colorbar(contour)
    cbar.set_label('Density')
    
    plt.tight_layout()

    return None

def plots_3d(joint_mean, joint_covar):
    """
    3D joint probability distribution vs empirical frequency of pooled 
    displ-accel pairs

    """
        
    # --- joint probability / relative frequency ---
    displ_range = np.linspace(min(displ.flatten()), max(displ.flatten()), num=100)
    accel_range = np.linspace(min(accel.flatten()), max(accel.flatten()), num=100)
    displ_grid, accel_grid = np.meshgrid(displ_range, accel_range)      # grid with combinations of displ and accel
    joint_prob = multivariate_normal.pdf(np.dstack((displ_grid, accel_grid)), mean=joint_mean, cov=joint_covar)  # joint prob. for each combi of accel and displ
    
    # --- empirical frequency ---
    freq, displ_range, accel_range = np.histogram2d(displ.flatten(), accel.flatten(), bins=100)
    displ_centers = 0.5 * (displ_range[1:] + displ_range[:-1])
    accel_centers = 0.5 * (accel_range[1:] + accel_range[:-1])
    displ_grid_empirical, accel_grid_empirical = np.meshgrid(displ_centers, accel_centers)
    height = freq.ravel()                   # flatten frequencies
    width = depth = 15                      # width and depth of each bar

    # --- 3D plot ---
    fig = plt.figure(figsize=(9, 4))
    axes1 = fig.add_subplot(121, projection='3d')
    axes1.plot_surface(displ_grid, accel_grid, joint_prob, cmap='viridis', linewidth=0, antialiased=True)
    axes1.set_xlabel('Displ')
    axes1.set_ylabel('Accel')
    axes1.set_zlabel('Probability')
    axes1.set_title('Joint Probability Distribution')
    
    axes2 = fig.add_subplot(122, projection='3d')
    axes2.bar3d(displ_grid_empirical.ravel(), accel_grid_empirical.ravel(), np.zeros_like(height), width, depth, height, shade=False, color='blue', alpha=0.8, linewidth=0.5)
    axes2.set_xlabel('Displ')
    axes2.set_ylabel('Accel')
    axes2.set_zlabel('Absolute Frequency')
    axes2.set_title('Empirical Joint Frequency')
    
    return None

def density_comparison():
    """
    comparison of marginal distributions (empirical vs analytical) of pooled 
    displacements and accelerations

    """

    # range of values to compute pdf for
    displ_range = np.linspace(-max(abs(displ.flatten())), max(abs(displ.flatten())), 100)
    accel_range = np.linspace(-max(abs(accel.flatten())), max(abs(accel.flatten())), 100)
    
    # parameters of marginal distributions according to equation (3.4) and (3.6) in pdf "about how to extract.."
    mean_displ, mean_accel = 0, 0
    var_displ = ((sigma_mot**2)/beta) * (2 * (np.exp(-beta*t) - (1 - beta*t))) / (beta*t)**2
    var_accel = ((2/t) * (sigma_mot**2)/beta) * (1 + 2*beta*t - (2-np.exp(-beta*t))**2) / (beta*t)**2
    
    # marginal probability
    displ_marginal_prob = norm.pdf(displ_range, mean_displ, np.sqrt(var_displ))
    accel_marginal_prob = norm.pdf(accel_range, mean_accel, np.sqrt(var_accel))

    # plot empirical and analytical distribution
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    
    # marginals for displacements
    axes[0].hist(displ.flatten(), bins=30, color='blue', alpha=0.4, density=True, label='Empirial PDF')
    axes[0].plot(displ_range, displ_marginal_prob, color='blue', label='Analytical PDF')
    axes[0].set_title('Empirical vs. Analytical Distribution')
    axes[0].set_xlabel('Displacement')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(None)
    
    # marginals for acceleration
    axes[1].hist(accel.flatten(), bins=30, color='orange', alpha=0.4, density=True, label='Empirical PDF')
    axes[1].plot(accel_range, accel_marginal_prob, color='orange', label='Analytical PDF')
    axes[1].set_xlabel('Acceleration')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(None)

    plt.tight_layout()
    
    return None

def density_comparison_each_dim():
    """
    comparison of marginal distributions (empirical vs analytical) for each dimension
 
    """
    
    # range of values to compute pdf for
    displ_range = np.linspace(-max(abs(displ.flatten())), max(abs(displ.flatten())), 100)
    accel_range = np.linspace(-max(abs(accel.flatten())), max(abs(accel.flatten())), 100)
 
    # parameters of marginal distributions according to equation (3.4) and (3.6) in pdf
    mean_displ, mean_accel = 0, 0
    var_displ = ((sigma_mot**2)/beta) * (2 * (np.exp(-beta*t) - (1 - beta*t))) / (beta*t)**2
    var_accel = ((2/t) * (sigma_mot**2)/beta) * (1 + 2*beta*t - (2-np.exp(-beta*t))**2) / (beta*t)**2
    
    # marginal probability
    displ_marginal_prob = norm.pdf(displ_range, mean_displ, np.sqrt(var_displ))
    accel_marginal_prob = norm.pdf(accel_range, mean_accel, np.sqrt(var_accel))

    # plot empirical and analytical distribution
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    
    # marginals for displacements in x dimension
    axes[0,0].hist(displ[0], bins=30, color='blue', alpha=0.4, density=True, label='Empirical PDF')
    axes[0,0].plot(displ_range, displ_marginal_prob, color='blue', label='Analytical PDF')
    axes[0,0].set_title('Empirical vs. Analytical Distribution - Displacement')
    axes[0,0].set_xlabel('Displacement (dim. x)')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()
    axes[0,0].grid(None)
    # marginals for displacements in y dimension
    axes[1,0].hist(displ[1], bins=30, color='blue', alpha=0.4, density=True, label='Empirical PDF')
    axes[1,0].plot(displ_range, displ_marginal_prob, color='blue', label='Analytical PDF')
    axes[1,0].set_xlabel('Displacement (dim. y)')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    axes[1,0].grid(None)
    
    # marginals for acceleration in x dimension
    axes[0,1].hist(accel[0], bins=30, color='orange', alpha=0.4, density=True, label='Empirical PDF')
    axes[0,1].plot(accel_range, accel_marginal_prob, color='orange', label='Analytical PDF')
    axes[0,1].set_title('Empirical vs. Analytical Distribution - Acceleration')
    axes[0,1].set_xlabel('Acceleration (dim. x)')
    axes[0,1].set_ylabel(' ')
    axes[0,1].legend()
    axes[0,1].grid(None)
    # marginals for acceleration in y dimension
    axes[1,1].hist(accel[1], bins=30, color='orange', alpha=0.4, density=True, label='Empirical PDF')
    axes[1,1].plot(accel_range, accel_marginal_prob, color='orange', label='Analytical PDF')
    axes[1,1].set_xlabel('Acceleration (dim. y)')
    axes[1,1].set_ylabel(' ')
    axes[1,1].legend()
    plt.grid(None)

    plt.tight_layout()

    return None

#%% getting output of whole script

if( __name__ == "__main__" ) :
    # specify variables
    t = 0.5                                   # time interval between consecutive frames
    zero_frame = 1                            # frame at time point 0
    minust_frame = zero_frame-1               # frame at time point t
    
    # computing empirical displacement and acceleration
    displ = displacement(track_df, minust_frame, zero_frame, t)
    accel = acceleration(track_df, zero_frame, t)
    
    # fit parameters with MLE
    beta, sigma_mot, D = findbtrackmotionparameters(displ, accel, t)
    
    # mean and covariance matrix of joint distribution with optimal param
    joint_mean = np.array([0, 0])
    joint_covar = joint_covariance(beta, sigma_mot)
    
    # plots to evaluate param estiamtes
    plots_2d(joint_mean, joint_covar)
    plots_3d(joint_mean, joint_covar)
    density_comparison()
    density_comparison_each_dim()
    
    