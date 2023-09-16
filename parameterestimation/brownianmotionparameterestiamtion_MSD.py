
"""
Inferring Brownian motion parameters from fitting MDS curve with generalized 
least squares

Note: adjust path/file name in section "import data" and variables in last 
    section "getting output of whole script" as desired
        
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

from scipy.stats import shapiro
sns.set()
sns.set_style('whitegrid')

#%% import data

track_df = pd.read_csv('G337_manualtracks/G337_40_frames_merged.csv')

#%% computing mean squared displacements (MSD) of manually tracked cells

def calc_squared_displacement(track_df, t_frame, zero_frame, random):
    """
    returns an array of squared displacements of each track between frame 0 
    (zero_frame) and any desired future frame (t_frame).
    
    In case of cell division, only one daughter cell track is chosen at random 
    for the calculations.

    """
    if random == True:
        # extracting y and x coordinates
        yx_zero = track_df[track_df['FRAME'] == zero_frame].sample(frac=1).drop_duplicates(subset='TRACK_ID').sort_index()[['POSITION_Y_EXACT', 'POSITION_X_EXACT']].values
        yx_t = track_df[track_df['FRAME'] == t_frame].sample(frac=1).drop_duplicates(subset='TRACK_ID').sort_index()[['POSITION_Y_EXACT', 'POSITION_X_EXACT']].values
    
    else:
        track_df_t = track_df[track_df['FRAME'] == t_frame].sort_values(['TRACK_ID', 'TARGET_ID'], ascending=[True, True])
        trackids_t = track_df_t['TRACK_ID'].values
        track_df_zero = track_df[track_df['FRAME'] == zero_frame].sort_values(['TRACK_ID', 'TARGET_ID'], ascending=[True, True])
        
        # extract x and y coordinates
        yx_t = track_df_t[['POSITION_Y_EXACT', 'POSITION_X_EXACT']].values
        yx_zero = np.array( [ np.array([track_df_zero.loc[track_df_zero['TRACK_ID'] == track].POSITION_Y_EXACT.values[0] for track in trackids_t ]),
                                np.array([track_df_zero.loc[track_df_zero['TRACK_ID'] == track].POSITION_X_EXACT.values[0] for track in trackids_t ]) ]
                            ).T
       
    # computing squared displacements
    yx_diff = yx_t - yx_zero
    norm = np.linalg.norm(yx_diff, axis=1)
    
    return abs(norm)**2

def calc_msd(squared_displacements):
    """

    returns MSD as scalar value representing the average of all squared displacements 
    in a given array.

    """
    
    return np.mean(squared_displacements)

#%% assessing empirical MSDs over different time intervals

def msd_random_plot(msd, errors, frames):
    
    """
    plotting empirical NSD curve
    """
    
    plt.figure()
    sns.lineplot(x=frames+1, y=empirical_msd, marker='o', color='red', linewidth=0.4, label='MSD')
    plt.errorbar(x=frames+1, y=empirical_msd, yerr=errors, fmt='none', ecolor='grey', elinewidth=0.8, label='+- 1 SE')
    plt.title('MDS for different time intervals')
    plt.xlabel('\u0394t (frames)')
    plt.ylabel('MSD (pixels\u00B2)')
    plt.legend()
    plt.grid(None)
    
    return None

def plot_individual_traces( allsquareddisplacements, frames, subframes ) :
    
    """
    plot overlay of the squared displacements of each individual trace together with their mean
    input is array of arrays of shape numberofframes x numberoftracks
    """
    
    allsquareddisplacements = np.array(allsquareddisplacements)
    meansquareddisplacements = np.mean( allsquareddisplacements[:,subframes], axis=1 )
    plt.figure()
    for j_track in subframes :
        sns.lineplot( x=frames+1, y=allsquareddisplacements[:,j_track], marker='', linewidth=0.4, label='' )
    sns.lineplot( x=frames+1, y=meansquareddisplacements, marker='o', color='red', linewidth=1.0, label="MSD" )
    plt.xlabel("number of frames")
    plt.ylabel("(mean) squared displacement")
    plt.title('Squared displacements of individual tracks')
    plt.grid(None)
    

def msd_random_all_plot(empirical_msd):
    """
    plot comparing empirical msd with random sampling vs msd with all daughter 
    cells.
    
    """
    
    # computing empirical msd with all daugther cells
    allsquareddisplacements_all = [ calc_squared_displacement(track_df, frame_i, 0, random=False) for frame_i in frames ]
    empirical_allmsd = np.array( [ calc_msd(allsquareddisplacements_all[i]) for i in range(len(allsquareddisplacements)) ] )
    
    # plot both curves
    plt.figure()
    sns.lineplot(x=frames+1, y=empirical_msd, marker='o', color='red', linewidth=0.4, label='MSD (random daughter sampling)')
    plt.fill_between(frames+1, empirical_msd-msd_se, empirical_msd+msd_se, alpha=0.2, facecolor='red', label='+- 1 Standard Error')
    sns.lineplot(x=frames+1, y=empirical_allmsd, marker='o', color='green', linewidth=0.4, label='MSD (all cells)')
    plt.title('MDS for different time intervals')
    plt.xlabel('\u0394t (frames)')
    plt.ylabel('MSD (pixels\u00B2)')
    plt.legend(loc='upper left')
    plt.grid(None)
    plt.show()

def assessing_empiricalMSD(allsquareddisplacements, empirical_msd, msd_se):
    
    print('-')
    print('Empirical MSD over different time intervals with random daughter sampling')
    print('     MSD samples = %d' %(len(frames)))
    print('     Tracks per MSD = %d' %(notracks))

    # plot empirical MSD with random sampling
    msd_random_plot(empirical_msd, msd_se, frames)
    
    # plot empirical MSD and individual tracks
    subtracks = np.linspace(0, notracks-1, notracks, dtype=int)
    plot_individual_traces( allsquareddisplacements, frames, subtracks )
    
    # plot random sampling vs all daughter cells
    msd_random_all_plot(empirical_msd)

    # test for independent standard errors (calculates autocorrelations internally)
    lag = 1
    results = sm.stats.acorr_ljungbox(msd_se, lags=[lag])  # Change the lag value as needed
    print('Ljung-box test results')
    print('   H0: MSD standard errors at lag %d are independent/not autocorrelated' %(lag))
    print(    results)
    print('-')

#%% optimize sigma_mot analytically using known generalized least squares 
#   expression and beta numerically

def squareddispl_covariance(allsquareddisplacements):
    """
    empirical covariance matrix of squared displacements as an approximation
    for the covariance of errors.
    """
    covar_matrix = np.cov(np.array(allsquareddisplacements), rowvar=True)
    
    return covar_matrix

def squareddispl_precision(covar_matrix, frames, reg = 1e-6):
    
    """
    returns precision matrix of squared displacements after cholesky decomposition
    """
    
    print('Computing precision matrix of MSD sampling standard errors')
    
    covar_matrix_reg = covar_matrix + reg * np.eye(len(frames))     # regularization to obtain positive definite matrix
    L = np.linalg.cholesky(covar_matrix_reg)                        # cholesky decomp.
    L_inv = np.linalg.inv(L)
    prec_matrix = L_inv @ L_inv.T

    # check whether precision matrix is symmetric
    if ( prec_matrix == prec_matrix.T ).all():
        print('     precision matrix is symmetric')
        print('-')
    else:
        print('     precision matrix is NOT symmetric, please double check!')
        print('-')

    return prec_matrix

def msd_analytical_withoutsigma(beta, time_lag):
    """
    
    analytical MSD expression without sigma_mot 

    """
    
    # equation (14) in report for reference
    term1 = 4/ beta**3
    term2 = beta*time_lag - 1 + np.exp(-beta*time_lag)
    return term1*term2

def best_sigma_mot_2(beta, time_lag, empirical_msd, prec_matrix):
    """
    generalized least square optimal parameter value for sigma_mot2 according 
    to expression Y - aX = Y - sigma_mot2 * X(beta, t)
    
    optimal sigma_mot^2 is conditioned on optimal beta value

    """
    X = msd_analytical_withoutsigma(beta, time_lag)             # design matrix dependent on beta and time
    
    # best sigma_mot^2 according to equation (18) in report
    return (X.T @ prec_matrix @ empirical_msd) / (X.T @ prec_matrix @ X) 

def objective_function(beta, time_lag, empirical_msd, prec_matrix):
    """
    target function which needs to be maximized numerically to obtain optimal 
    beta value.

    """
    X = msd_analytical_withoutsigma(beta, time_lag)
    
    # equation (19) in report for reference
    num = X.T @ prec_matrix @ empirical_msd
    denom = X.T @ prec_matrix @ X
    
    return ((num / denom ) * num)

def residuals_plots(residuals):
    """
    Assessing fit of analytical MSD curve with residual diagnostic plots
    
    """
    
    # plot residuals against frames    
    fig = plt.figure(figsize=(6,4))
    plt.scatter(x=frames + 1, y=residuals, marker='o', color='black', facecolors='none', edgecolors='black')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=np.mean(residuals), color='red', linestyle='dashdot', linewidth=0.6, label='Mean Residuals')
    plt.title('Residuals of MSD fit')
    plt.xlabel('\u0394t (frames)')
    plt.ylabel('Residuals (pixels\u00B2)')
    plt.legend()
    plt.grid(None)
    plt.show()
    
    # QQ plot of standardized residuals
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    fig, ax = plt.subplots(figsize=(6,4))
    sm.qqplot(standardized_residuals, line='s', ax=ax)  
    ax.set_title("QQ Plot of Standardized Residuals from MSD fit")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.get_lines()[0].set_markerfacecolor('None')
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[0].set_markersize(6)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linestyle('--')
    plt.grid(None)
    plt.show()

def findmotionparameters_gls(empirical_msd, t):
    
    """
    finds the optimal values of beta and sigma_mot and returns diagnostic plots
    on the model fit
    """
    
    # define covar/precision of squared displacements
    covar_matrix = squareddispl_covariance(allsquareddisplacements)
    prec_matrix = squareddispl_precision(covar_matrix, frames, reg = 1e-6)

    # time variable values
    time_lag = frames*t

    # positive bound on beta 
    bounds = [(0.000001, None)]

    # get optimal beta and sigma_mot
    initial_guess_beta = 10000*np.random.rand(1)
    minimiseme = ( lambda x: objective_function(x[0], time_lag, empirical_msd, prec_matrix) )
    results = minimize(minimiseme, initial_guess_beta, method='SLSQP')
    beta_best = results.x[0]; sigma_mot_best = np.sqrt( best_sigma_mot_2(beta_best, time_lag, empirical_msd, prec_matrix))
    
    # standard deviation on beta estimate with finite differencing
    jacobian = results.jac                                      # first derivative of objective function w.r.t. each param
    epsilon = 1e-4                                              # small value for finite differences
    beta_perturbed = beta_best + epsilon
    gls_perturbed = objective_function(beta_perturbed, time_lag, empirical_msd, prec_matrix)
    gls_best = objective_function(beta_best, time_lag, empirical_msd, prec_matrix)
    jacobian_perturbed = (gls_best - gls_perturbed) / epsilon            # gradient of perturbed beta estimate 
    hessian = (jacobian - jacobian_perturbed) / epsilon   
    beta_best_var = 1/hessian
    beta_best_sd = np.sqrt(beta_best_var)     

    # standard deviation on sigma_mot estimate using GLS equation (2) in report
    residuals = (empirical_msd - ((sigma_mot_best**2) * msd_analytical_withoutsigma(beta_best, time_lag))).reshape(-1,1)
    sigma2 = residuals.T @ prec_matrix @ residuals * (1/(len(empirical_msd) - 1))      # -1 bc one parameter of interest in linear model
    X = msd_analytical_withoutsigma(beta_best, time_lag).reshape(-1,1)
    sigma2_mot_best_var = sigma2 * np.linalg.inv(X.T @ prec_matrix @ X)
    sigma_mot_best_var = np.sqrt(sigma2_mot_best_var)
    sigma_mot__best_sd = np.sqrt(sigma_mot_best_var)
    
    print( "Fitting motion model parameters beta and sigma_mot to to empirical MSD curve with GLS")
    print( " Initial guess:")
    print( "    beta = %+1.5e           sigma_mot = %+1.5e           residual cost = %1.5e" %(initial_guess_beta, best_sigma_mot_2(initial_guess_beta, time_lag, empirical_msd, prec_matrix), minimiseme(initial_guess_beta)) )
    print( " Best guess:")
    print( "    beta = %+1.5e +- %1.5e  sigma_mot = %+1.5e +- %1.5e  residual cost = %1.5e" %(beta_best, beta_best_sd, sigma_mot_best, sigma_mot__best_sd, minimiseme(np.array([beta_best]))) )
    
    if not np.isnan(beta_best):
        # plot fitted curve against empirical curve
        plt.figure()
        sns.lineplot(x=frames+1, y=empirical_msd, marker='o', color='red', linewidth=0.4, label='empirical MSD')
        plt.errorbar(x=frames+1, y=empirical_msd, yerr=msd_se, fmt='none', ecolor='grey', elinewidth=0.8, label='+- 1 Standard Error')
        sns.lineplot(x=frames+1, y=(sigma_mot_best**2) * msd_analytical_withoutsigma(beta_best, time_lag), color='blue', linewidth=1.0, label='analytical MSD fit')
        plt.title('MDS for different time intervals')
        plt.xlabel('\u0394t (frames)')
        plt.ylabel('MSD (pixels\u00B2)')
        plt.legend(loc='upper left')
        plt.grid(None)
        
        residuals_plots(residuals)
        
    else:
        print('unable to display plots because beta and/or sigma_mot are NaN.')

#%% getting output of whole script

if( __name__ == "__main__" ) :
    # specify variables
    t = 0.5
    noframes = 3                                                        # number of frames to be excluded (based on Shapiro-Wilk test on MSD standard errors)
    frames = track_df['FRAME'].unique()[noframes+1:]                    # remaining frames
    notracks = track_df.loc[track_df['FRAME'] == 0].shape[0]            # number of tracks in dataframe

    # compute MSD
    allsquareddisplacements = [ calc_squared_displacement(track_df, frame_i, 0, random=True) for frame_i in frames ]     # squared displacements for different time frames 
    empirical_msd = np.array( [ calc_msd(allsquareddisplacements[i]) for i in range(len(allsquareddisplacements)) ] )    # MSD for different time frames 
    msd_sd = np.array( [ np.std(allsquareddisplacements[i]) for i in range(len(allsquareddisplacements)) ])              # standard deviation of squared displacement to MSD
    msd_se = (msd_sd / np.sqrt(notracks))                               # sampling standard error 
    
    # analyse MSD curve and fit motion parameters
    assessing_empiricalMSD(allsquareddisplacements, empirical_msd, msd_se)
    findmotionparameters_gls(empirical_msd, t)




