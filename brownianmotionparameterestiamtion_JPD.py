
"""
Inferring Brownian motion parameters with sampling from the joint posterior
distribution.

Note: adjust path/file name in section "import track data" and variables in last 
    section "getting output of whole script" as desired
"""

import numpy as np
import scipy
import math                                 # for gamma
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy
import pandas as pd

from scipy.stats import multivariate_normal, norm
from scipy.stats import entropy

#%% import track data

track_df = pd.read_csv('G337_manualtracks/G337_3_frames_merged.csv')

#%% Computing displacement and acceleration of tracked cells
t = 0.5

def displacement(track_df, minust_frame, zero_frame, t):
    
    """
    returns (2 x numberofcells) matrix containing displacements of tracked cells
    between frame 'zero_frame' and frame 'minust_frame'.
        - row 0: cell displacement dimension x
        - row 1: cell displacement dimension y 

    """
    
    # sort dataframe at t=0 according to track and target id
    track_df_zero = track_df.loc[track_df['FRAME'] == zero_frame].sort_values(['TRACK_ID', 'TARGET_ID'], ascending=[True, True])
    originids = track_df_zero['ORIGIN_ID'].values.tolist()         # origin ids of cells at t=0

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

def findbtrackmotionparametersMC(displ, accel, t) :
    """
    extracts drag (beta) and diffusion (D) from displacemts and accelerations

    """
    
    # set data:
    time1 = timer()                         # to measure time-consumption
    k = 1.2                                 # shape parameter of prior on sigma_mot
    theta = 500                             # rate parameter of prior on sigma_mot
    falloff = 100000.0                      # scale parameter of prior on beta
    withgraphical = True                    # 'True' for graphical output of inferred parameters, 'False' otherwise
    disctype = -1                           # '+1' for displacement measured at times 0,+t; '-1' for displacement measured at time -t,0
    (v,w,x, nocells,dim) = getsummarydata( displ, accel, disctype )
    print( " Info - findbtrackmotionparametersMC: Start looking for motion parameter now, for %d cells, %d dimensions, disctype %d (after %1.3f sec)." %(nocells,dim,disctype,timer()-time1) )
    pars_hist_chain = multisampler( v,w,x, nocells,dim, t, k,theta,falloff, withgraphical,time1 )
    beta_hist = pars_hist_chain[0,:];   sigma_mot_hist = pars_hist_chain[1,:]
    
    # command-window output:
    print( " Info - findbtrackmotionparametersMC: Motion parameter found after %1.3f sec:" %(timer()-time1) )
    print( "  Equations of motion:" )
    print( "   dx = v*dt" )
    print( "   dv = -beta*v*dt + sqrt(2)*sigma_mot*xi" )
    print( "   (for xi: standard Brownian motion, xi(t)-xi(0) ~ N(0,t))" )
    print( "   (    D:  (sigma_mot/beta)^2)" )
    print( "  Prior:" )
    print( "   beta:       %1.5e +- %1.5e" %(falloff,falloff) )
    mu = (np.sqrt(theta)*math.gamma(k-1/2)/math.gamma(k) if (k>0.5) else np.inf)    # see Eichenlaub 2021, eq8
    err = (np.sqrt( theta/(k-1) - mu**2 ) if (k>1) else np.inf)
    print( "   sigma_mot:  %1.5e +- %1.5e" %(mu,err) )
    print( "  Posterior:" )
    print( "   beta:       %1.5e +- %1.5e" %( np.mean(beta_hist),np.std(beta_hist) ) )
    print( "   sigma_mot:  %1.5e +- %1.5e" %( np.mean(sigma_mot_hist),np.std(sigma_mot_hist) ) )
    print( "   D:          %1.5e +- %1.5e" %( np.mean((sigma_mot_hist/beta_hist)**2),np.std((sigma_mot_hist/beta_hist)**2) ) )
    
    # graphical output:
    if( withgraphical ) :
        plotdata( displ,accel, disctype, np.mean(beta_hist),np.mean(sigma_mot_hist),t )
        print( " Plot for beta-sigma_mot shows posterior samples, black lines are rescaled prior." )
        print( " Plot for displ-accel shows input samples, turquoise lines show analytical assumptions for beta and sigma_mot equal their posterior means." )
    
    return np.mean(beta_hist), np.mean(sigma_mot_hist)
    
def getsummarydata( displ, accel, disctype ) :
    # computes summary of data: v=var(displ),w=var(accel), x=cov(displ,accel)
    nocells = displ.shape[1]                # number of cells
    dim = displ.shape[0]                    # number of spatial dimensions
    cov_here = np.cov(np.reshape(displ,dim*nocells),np.reshape(accel,dim*nocells))
    v = cov_here[0,0]; w = cov_here[1,1]
    if( disctype==-1 ) :                    # displacement measured at times -t,0
        x = +cov_here[0,1]
    elif( disctype==+1 ) :                  # displacement measured at times 0,+t
        x = -cov_here[0,1]
    else :                                  # unknown disctype
        print( " Warning - getsummarydata: disctype %d not defined, needs to be +1 or -1." %(disctype) )
    return v,w,x, nocells,dim
def getvarianceanddet( beta, sigma_mot, t ) :
    # computes variance of joint displacement-acceleration distribution
    # set auxiliary parameters:
    resctime = beta*t
    # output:
    if( resctime<1e-4 ) :                                                   # use small-values approximation
        mycov = np.array( [[1/beta,(-2/3)*np.sqrt(t)],[(-2/3)*np.sqrt(t),4/3]] )*(sigma_mot**2)
        mydet = np.linalg.det(mycov)#mycov[0,0]*mycov[1,1]-mycov[0,1]*mycov[1,0]
    elif( resctime>1e+3 ) :                                                 # use large-values approximation
        mycov = np.array( [[t,-np.sqrt(t)],[-np.sqrt(t),2]] )*(2*(sigma_mot**2)/(resctime**2))
        mydet = np.linalg.det(mycov)#mycov[0,0]*mycov[1,1]-mycov[0,1]*mycov[1,0]
    else :                                                                  # use full formula
        term1 = (1 + 2*resctime - (2-np.exp(-resctime))**2)/(resctime**2)
        term2 = ((1-np.exp(-resctime))/resctime)**2
        mycov = np.array([[ term1+term2, -term1/np.sqrt(t) ],[-term1/np.sqrt(t), 2*term1/t]])*((sigma_mot**2)/beta)
        mydet = np.linalg.det(mycov)#mycov[0,0]*mycov[1,1]-mycov[0,1]*mycov[1,0]
    return mycov, mydet
def getprecanddet( beta, sigma_mot, t ) :
    # computes precision and determinant of displacement-acceleration distribution
    (mycov,mydet) = getvarianceanddet( beta,sigma_mot, t )
    myprec = np.array([ [mycov[1,1],-mycov[0,1]], [-mycov[1,0],mycov[0,0]] ])/mydet
    return myprec, 1/mydet              # determinant is of precision
def getlogmargtarget( beta, v,w,x, nocells,dim, t, k,theta,falloff ) :
    # computes log of beta-marginal of target
    # priors are Gamma(k,theta) for 1/sigma_mot^2 (shape,rate-parameter)
    # ...exp(-beta/falloff) for beta
    if( beta<=0 ) :                                                     # test prior
        return -np.inf
    (myprec,myprecdet) = getprecanddet( beta, 1.0, t )
    if( np.abs(myprecdet)==np.inf ) :
        print( " Warning - getlogmargtarget: Got small determinant %+1.5e for beta=%1.5e,t=%1.5e" %(myprecdet,beta,t) )
    fullk = k + nocells*dim
    fulltheta = theta + (nocells*dim/2)*( myprec[0,0]*v + myprec[1,1]*w + 2*myprec[0,1]*x )
    if( ((beta*t)>5e3) and (nocells*dim*((beta*t)**2))>falloff*1e5 ) :# use large-values approximation
        rescprec = np.array( [[2/t,1/np.sqrt(t)],[1/np.sqrt(t),1]] )
        logtarget_here = -beta/falloff - np.log( t )*(nocells*dim/2) - np.log(((beta*t)**2)/2)*k - np.log( (nocells*dim/2)*( rescprec[0,0]*v + rescprec[1,1]*w + 2*rescprec[0,1]*x ) )*fullk
    else :                                                              # use full formula
        logtarget_here = -beta/falloff + np.log( myprecdet )*(nocells*dim/2) - np.log(fulltheta)*fullk
    if( np.isnan(logtarget_here) | np.isinf(logtarget_here) ) :
        print( " Warning - getlogmargtarget: Got pathological logtarget = %+1.5e, fullk = %+1.5e, fulltheta = %+1.5e, precdet = %+1.5e" %(logtarget_here,fullk,fulltheta,myprecdet) )
    return logtarget_here, fullk,fulltheta
def samplesigma_motfromconditional( beta, v,w,x, nocells,dim, t, k,theta, fullk=np.nan,fulltheta=np.nan ) :
    # Gibbs-sampler for sigma_mot
    # prior is Gamma(k,theta) for 1/sigma_mot^2 (shape,rate-parameter)
    
    if( np.isnan(fullk) | np.isnan(fulltheta) ) :           # fullk and fulltheta not already given
        myprec = getprecanddet( beta, 1.0, t )[0]
        fullk = k + nocells*dim
        fulltheta = theta + (nocells*dim/2)*( myprec[0,0]*v + myprec[1,1]*w + 2*myprec[0,1]*x )
    return 1/np.sqrt(np.random.gamma( fullk, 1/fulltheta ))          # second argument is scale parameter
def updatebeta_rw( beta_curr,logtarget_curr,fullk_curr,fulltheta_curr, v,w,x, nocells,dim, t, k,theta,falloff, updatepars ) :
    # updates beta via rw-proposal
    # get proposal:
    beta_prop = np.abs( beta_curr + updatepars[0]*(2.0*np.random.rand()-1.0) )  # rw-proposal
    (logtarget_prop,fullk_prop,fulltheta_prop) = getlogmargtarget( beta_prop, v,w,x, nocells,dim, t, k,theta,falloff )
    # accept/reject:
    logdiff = logtarget_prop-logtarget_curr
    updatepars[1] += 1.0 - min(1.0,np.exp(logdiff));    updatepars[2] += 1      # rejection and proposal counters
    if( np.log(np.random.rand())<logdiff ) :                                    # accept
        #print( " Info - updatebeta_rw: Accept beta %1.5e-->%1.5e, logdiff = %+1.5e" %(beta_curr,beta_prop,logdiff) )
        beta_curr = beta_prop; logtarget_curr = logtarget_prop; fullk_curr = fullk_prop; fulltheta_curr = fulltheta_prop
    return beta_curr,logtarget_curr,fullk_curr,fulltheta_curr, updatepars
def jointsampler( beta_init, v,w,x, nocells,dim, t, k,theta,falloff, updatepars, burnin,MCmax,subsample ) :
    # MCsampler beta,sigma_mot
    
    # initialise:
    pars_hist = np.zeros((2,MCmax))                             # initialise memory
    beta_curr = beta_init                                       # initial beta
    (logtarget_curr,fullk_curr,fulltheta_curr) = getlogmargtarget( beta_curr, v,w,x, nocells,dim, t, k,theta,falloff )
    if( np.isnan(logtarget_curr) | np.isinf(logtarget_curr) ) :
        print( " Warning - jointsampler: Pathological initial condition, beta_init = %+1.5e, logtarget_init = %+1.5e" %(beta_curr,logtarget_curr) )
    
    for MCit in range(0,MCmax) :
        for j_sub in range(0,subsample) :
            (beta_curr,logtarget_curr,fullk_curr,fulltheta_curr, updatepars) = updatebeta_rw( beta_curr,logtarget_curr,fullk_curr,fulltheta_curr, v,w,x, nocells,dim, t, k,theta,falloff, updatepars )
            (sigma_mot_curr) = samplesigma_motfromconditional( beta_curr, v,w,x, nocells,dim, t, k,theta, fullk_curr,fulltheta_curr )
            updatepars = optimisestepsizes( updatepars, MCit,burnin )
        #print( " Info - jointsampler (%d): beta = %1.5e, sigma_mot = %1.5e, logtarget = %+1.5e, fullk = %1.5e, fulltheta = %1.5e" %(MCit, beta_curr,sigma_mot_curr, logtarget_curr,fullk_curr,fulltheta_curr) )
        pars_hist[:,MCit] = [beta_curr,sigma_mot_curr]
    return pars_hist, updatepars
def multisampler( v,w,x, nocells,dim, t, k,theta,falloff, withgraphical,starttime, nochains=5,burnin=5000,MCmax=10000,subsample=5 ) :
    # multiple independent joint samplers
    
    # set auxiliary parameters:
    pars_hist_chain = np.zeros((2,MCmax,nochains))              # initialise
    GRRwarning_threshold = 1.01                                 # threshold for GRR when to warn
    
    # run individual chains:
    print( " Info - multisampler: Start sampling with [%d..%d]x%d samples and %d chains (after %1.3f sec)." %(burnin+1,MCmax,subsample, nochains, timer()-starttime) )
    for j_chain in range(0,nochains) :
        updatepars = [ 0.05*falloff, 0.0, 0.0, 2.0 ]            # initialise update-parameters
        beta_init = -falloff*np.log( np.random.rand() )         # initialise from prior
        (pars_hist, updatepars) = jointsampler( beta_init, v,w,x, nocells,dim, t, k,theta,falloff, updatepars, burnin,MCmax,subsample )
        pars_hist_chain[:,:,j_chain] = pars_hist
        #print( " Info - multisampler: Chain %d statistics: beta = %1.5e+-%1.5e, sigma_mot=%1.5e+-%1.5e, rej-rate=%1.5e, step=%1.5e (after %1.3f sec)." %(j_chain, np.mean(pars_hist[0,burnin:]),np.std(pars_hist[0,burnin:]), np.mean(pars_hist[1,burnin:]),np.std(pars_hist[1,burnin:]), updatepars[1]/updatepars[2], updatepars[0], timer()-starttime) )
    
    # output:
    # ...control-window:
    GRR_simple = getGRR( pars_hist_chain,burnin )
    if( ~(max(GRR_simple)<GRRwarning_threshold) ) :             # large?
        print( " Info - multisampler: Some summary statistics, as %1.5f>%1.5f:" %(max(GRR_simple),GRRwarning_threshold) )
        print( "  beta      = %1.5e+-%1.5e   (GRRs = %1.5f)" %(np.mean(pars_hist_chain[0,burnin:,:]),np.std(pars_hist_chain[0,burnin:,:]), GRR_simple[0]) )
        print( "  sigma_mot = %1.5e+-%1.5e   (GRRs = %1.5f)" %(np.mean(pars_hist_chain[1,burnin:,:]),np.std(pars_hist_chain[1,burnin:,:]), GRR_simple[1]) )
        for j_chain in range(0,nochains) :
            print( "   chain %d: beta = %1.5e+-%1.5e, sigma_mot = %1.5e+-%1.5e" %(j_chain, np.mean(pars_hist_chain[0,burnin:,j_chain]),np.std(pars_hist_chain[0,burnin:,j_chain]), np.mean(pars_hist_chain[1,burnin:,j_chain]),np.std(pars_hist_chain[1,burnin:,j_chain]) ) )
        if( not withgraphical ) :
            withgraphical = True; print( " Info - multisampler: Set withgraphical to true." )   # override withgraphical to get graphical output for analysis
    # ...graphical:
    if( withgraphical ) :
        ( myedges_beta,myedges_sigma ) = getbinsforhistograms( pars_hist_chain, burnin,MCmax )
        if( max(GRR_simple)>GRRwarning_threshold ) :
            plotmarginalposteriorperchain( pars_hist_chain, burnin,MCmax, k,theta,falloff, myedges_beta,myedges_sigma )
        plotjointposterior( pars_hist_chain, burnin,MCmax, k,theta,falloff, myedges_beta,myedges_sigma )
    
    return pars_hist_chain[:,burnin:,:]                     # without burnin
def optimisestepsizes( updatepars, MCit,burnin ) :
    # optimises stepsizes to be in reasonablerange
    if( MCit>burnin ) :                                     # do not adjust stepsizes past burnin
        return updatepars
    
    without = False                                         # 'true' for output when adjusting, 'false' otherwise
    minminsamples = 1000                                    # minimum of minimum number of samples
    minsamples = np.int_(np.ceil( np.sqrt(burnin) ))
    minsamples = np.maximum( minminsamples, minsamples )
    
    if( updatepars[2]>=minsamples ) :
        oldsteps = copy.deepcopy(updatepars[0])             # memory for output
        middle = 0.78
        span = 0.05
        rejrate = updatepars[1]/updatepars[2]
        deviation = ( rejrate - middle )/span
        myadjfctr = np.abs( updatepars[3] )
        # check if overshot:
        ssign = np.sign( deviation ) if (deviation!=0) else (2*np.random.randint(2)-1)
        if( ssign*np.sign(updatepars[3])<0 ) :              # overshot
            newadjfctr = pow(myadjfctr,1/2)
            if( not ((newadjfctr==0) or np.isinf(newadjfctr) or np.isnan(newadjfctr)) ) :           # ie no numerical problems with adjusting adjfctrs
                myadjfctr = newadjfctr                      # otherwise keep as is
        updatepars[3] = ssign*myadjfctr             # same sign as deviation
        # adjust stepsize:
        newpar_stps = updatepars[0]*pow(myadjfctr,-deviation)
        if( not ((newpar_stps==0) or np.isinf(newpar_stps) or np.isnan(newpar_stps)) ) :    # ie no numerical problems with adjusting adjfctrs
            updatepars[0] = newpar_stps             # otherwise keep as is
        # update rejection parameters:
        updatepars[1] = 0.0; updatepars[2] = 0
        # output changes to control-window:
        if( without ) :
            print( " Info - optimisestepsizes (%d): Adjust update, [ %1.5e ]->[ %1.5e ]\t(rej %1.3f)" %(MCit, oldsteps, updatepars[0], rejrate) )
    return updatepars

def getGRR( myMCMCarray,burnin ) :
    # computes GR statistic for each parameter
    # myMCMCarray is numpy array with first index is parameter, second index is iteration, third index is chain
    
    # get auxiliary parameters:
    nochains = myMCMCarray.shape[2]         # number of independent chains
    nopar = myMCMCarray.shape[0]            # should be same for all
    statsrange = range(burnin,myMCMCarray.shape[1])# should be same for all
    nstatsrange = len(statsrange)           # number of samples
    GRR_simple = np.zeros(nopar)            # initialise
    for j_par in range(0,nopar) :           # evaluate each parameter independently
        # get auxiliary parameters:
        between = np.array([ np.mean(myMCMCarray[j_par,statsrange,j_chain]) for j_chain in range(0,nochains) ])
        within = np.array([ np.std(myMCMCarray[j_par,statsrange,j_chain],ddof=1) for j_chain in range(0,nochains) ])
        B = np.var( between, ddof=1 )*nstatsrange
        W = np.mean( within**2 )
        V_simple = (1-(1/nstatsrange))*W + (1/nstatsrange)*B
        # get output:
        GRR_simple[j_par] = np.sqrt( V_simple/W )
    return GRR_simple
def getbinsforhistograms( pars_hist_chain, burnin,MCmax ) :
    # gets bin-edges for histograms
    
    # set auxiliary parameters:
    nochains = pars_hist_chain.shape[2]                     # number of chains
    
    # get bins:
    minedges = +np.inf; maxedges = -np.inf; dedges = +np.inf    # initialise
    for j_chain in range(0,nochains) :
        myedges_here = np.histogram_bin_edges( pars_hist_chain[0,burnin:,j_chain], bins='auto' )
        minedges = min(minedges,myedges_here[0]); maxedges = max(maxedges,myedges_here[-1]); dedges = min(dedges,myedges_here[1]-myedges_here[0])
    myedges_beta = np.arange( minedges,maxedges, dedges )
    minedges = +np.inf; maxedges = -np.inf; dedges = +np.inf    # initialise
    for j_chain in range(0,nochains) :
        myedges_here = np.histogram_bin_edges( pars_hist_chain[1,burnin:,j_chain], bins='auto' )
        minedges = min(minedges,myedges_here[0]); maxedges = max(maxedges,myedges_here[-1]); dedges = min(dedges,myedges_here[1]-myedges_here[0])
    myedges_sigma = np.arange( minedges,maxedges, dedges )
    
    return myedges_beta,myedges_sigma
def plotmarginalposteriorperchain( pars_hist_chain, burnin,MCmax, k,theta,falloff, myedges_beta,myedges_sigma ) :
    # graphical output for marginals and traces
    
    # set auxiliary parameters:
    nochains = pars_hist_chain.shape[2]                     # number of chains
    dedges_beta = myedges_beta[1]-myedges_beta[0]           # increment
    dedges_sigma = myedges_sigma[1]-myedges_sigma[0]        # increment
    
    # graphical output:
    # ...trace beta:
    plt.figure()
    for j_chain in range(0,nochains) :
        plt.plot( np.array(range(burnin,MCmax)), pars_hist_chain[0,burnin:,j_chain], label=("chain %d" %(j_chain)) )
    plt.xlabel('MCiteration'); plt.ylabel('beta'); plt.legend()
    plt.show()
    # ...histogram beta:
    plt.figure()
    for j_chain in range(0,nochains) :
        plt.hist( pars_hist_chain[0,burnin:,j_chain], bins=myedges_beta, alpha=0.5, label=("chain %d" %(j_chain)) )
    norm_here = np.sum( np.exp(-myedges_beta/falloff) )*dedges_beta
    plt.plot( myedges_beta, np.exp(-myedges_beta/falloff)*(dedges_beta*(MCmax-burnin)/norm_here), c='red', label='prior' )
    plt.xlabel('beta'); plt.ylabel('freq'); plt.legend()
    plt.show()
    # ...trace sigma_mot:
    plt.figure()
    for j_chain in range(0,nochains) :
        plt.plot( np.array(range(burnin,MCmax)), pars_hist_chain[1,burnin:,j_chain], label=("chain %d" %(j_chain)) )
    plt.xlabel('MCiteration'); plt.ylabel('sigma_mot'); plt.legend()
    plt.show()
    # ...histogram sigma_mot:
    plt.figure()
    for j_chain in range(0,nochains) :
        plt.hist( pars_hist_chain[1,burnin:,j_chain], bins=myedges_sigma, alpha=0.5, label=("chain %d" %(j_chain)) )
    norm_here = np.sum( scipy.stats.gamma(k+3/2).pdf(theta*(myedges_sigma**(-2))) )*dedges_sigma
    plt.plot( myedges_sigma, (scipy.stats.gamma(k+3/2).pdf(theta*(myedges_sigma**(-2))))*(dedges_sigma*(MCmax-burnin)/norm_here), c='red', label='prior' )
    plt.xlabel('sigma_mot'); plt.ylabel('freq'); plt.legend()
    plt.show()
def plotjointposterior( pars_hist_chain, burnin,MCmax, k,theta,falloff, myedges_beta,myedges_sigma ) :
    # plots joint posterior between beta and sigma_mot
    
    # set auxiliary parameters:
    nochains = pars_hist_chain.shape[2]                     # number of chains
    nosamples = (MCmax-burnin)*nochains                     # total number of samples (pooled over chains)
    dx = myedges_beta[1]-myedges_beta[0]; beta_range = np.arange(myedges_beta[0],myedges_beta[-1], dx/100 )
    dy = myedges_sigma[1]-myedges_sigma[0]; sigma_range = np.arange(myedges_sigma[0],myedges_sigma[-1], dy/100 )
    
    # graphical output:
    plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5])
    axr = plt.subplot(gs[1,1]); plt.xlabel('freq')          # right
    axt = plt.subplot(gs[0,0]); plt.ylabel('freq')          # top
    ax = plt.subplot(gs[1,0], sharex=axt, sharey=axr)       # main
    
    axr.hist( np.reshape(pars_hist_chain[1,burnin:,:],nosamples), bins=myedges_sigma, color='turquoise', orientation="horizontal" )
    norm_here = np.sum( scipy.stats.gamma(k+3/2).pdf(theta*(sigma_range**(-2))) )*(sigma_range[1]-sigma_range[0])
    axr.plot( scipy.stats.gamma(k+3/2).pdf(theta*(sigma_range**(-2)))*(nosamples*dy/norm_here), sigma_range, color='black', linewidth=2 )
    axt.hist( np.reshape(pars_hist_chain[0,burnin:,:],nosamples), bins=myedges_beta, color='turquoise' )
    norm_here = np.sum( np.exp(-beta_range/falloff) )*(beta_range[1]-beta_range[0])
    axt.plot( beta_range, np.exp(-beta_range/falloff)*(nosamples*dx/norm_here), color='black', linewidth=2 )
    ax.hist2d( np.reshape(pars_hist_chain[0,burnin:,:],nosamples), np.reshape(pars_hist_chain[1,burnin:,:],nosamples), bins=[myedges_beta,myedges_sigma] )
    plt.xlabel('beta'); plt.ylabel('sigma_mot')
    plt.show()
def plotdata( displ,accel, disctype, beta_best,sigma_mot_best, t ) :
    # graphical depiction of input data and analytic shape given parameter estimates
    
    # set auxiliary parameters:
    nodata = displ.size                                         # number of datapoints in total
    mycov = getvarianceanddet( beta_best, sigma_mot_best, t )[0]   # covariance matrix
    if( disctype==-1 ) :                    # displacement measured at times -t,0
        1+1 # nothing to do here
    elif( disctype==+1 ) :                  # displacement measured at times 0,+t
        mycov[0,1] = -mycov[0,1]; mycov[1,0] = -mycov[1,0]
    else :                                  # unknown disctype
        print( " Warning - plotdata: disctype %d not defined, needs to be +1 or -1." %(disctype) )
    mysig= scipy.linalg.sqrtm( mycov )
    myedges_displ = np.histogram_bin_edges( displ, bins='auto' )# edges of marginal on displacements
    myedges_accel = np.histogram_bin_edges( accel, bins='auto' )# edges of marginal on accelerations
    arcrange = np.arange(0,2*np.pi,0.01); mycircle = np.array( [np.cos(arcrange),np.sin(arcrange)] )
    dx = myedges_displ[1]-myedges_displ[0]; displ_range = np.arange(myedges_displ[0],myedges_displ[-1], dx/100 )
    dy = myedges_accel[1]-myedges_accel[0]; accel_range = np.arange(myedges_accel[0],myedges_accel[-1], dy/100 )
    
    # graphical output:
    plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5])
    axr = plt.subplot(gs[1,1]); plt.xlabel('freq')          # right
    axt = plt.subplot(gs[0,0]); plt.ylabel('freq')          # top
    ax = plt.subplot(gs[1,0], sharex=axt, sharey=axr)       # main
    
    axr.hist( np.matrix.flatten(accel), bins=myedges_accel, color='orangered', orientation="horizontal" )
    axr.plot( scipy.stats.norm.pdf(accel_range, loc=0, scale=np.sqrt(mycov[1,1]))*(nodata*dy), accel_range, color='turquoise', linewidth=2 )
    axt.hist( np.matrix.flatten(displ), bins=myedges_displ, color='orangered' )
    axt.plot( displ_range, scipy.stats.norm.pdf(displ_range, loc=0, scale=np.sqrt(mycov[0,0]))*(nodata*dx), color='turquoise', linewidth=2 )
    ax.hist2d( np.matrix.flatten(displ), np.matrix.flatten(accel), bins=[myedges_displ,myedges_accel], cmap='hot' )    
    ax.scatter( 0,0, color='turquoise' )
    sigmacircle = mysig@mycircle; ax.plot( sigmacircle[0,:], sigmacircle[1,:], color='turquoise', linewidth=2 )
    sigma2circle = 2*mysig@mycircle; ax.plot( sigma2circle[0,:], sigma2circle[1,:], color='turquoise', linewidth=2 )
    plt.xlabel('displ'); plt.ylabel('accel')
    plt.show()

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
    
    # fit parameters with JPD approach
    beta, sigma_mot = findbtrackmotionparametersMC(displ, accel, t)
    
    # mean and covariance matrix of joint distribution with optimal param
    joint_mean = np.array([0, 0])
    joint_covar = joint_covariance(beta, sigma_mot)
    
    # plots to evaluate param estiamtes
    plots_2d(joint_mean, joint_covar)
    plots_3d(joint_mean, joint_covar)
    density_comparison()
    density_comparison_each_dim()
    
    