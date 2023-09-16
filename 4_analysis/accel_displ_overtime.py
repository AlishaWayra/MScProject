"""
Comparing density of cell displacements and accelerations at different points
in time and assessing deviation from zero-mean distribution.

Note: adjust path/file name in section "import track data" and variables in last 
    section "getting output of whole script" as desired
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')

#%% import track data

track_df = pd.read_csv('G337_manualtracks/G337_40_frames_merged.csv')

#%% Computing displacement and acceleration

def displacement(track_df, minust_frame, zero_frame, t):
    """
    
    returns (2 x numberofcells) matrix containing displacements of tracked cells
    between frame 'zero_frame' and frame 'minust_frame'.
        - row 0: cell displacement dimension x
        - row 1: cell displacement dimension y 

    """
    
    # sort dataframes according to track and target id
    track_df_zero = track_df.loc[track_df['FRAME'] == zero_frame].sort_values(['TRACK_ID', 'TARGET_ID'], ascending=[True, True])
    track_df_minust = track_df.loc[track_df['FRAME'] == minust_frame].sort_values(['TRACK_ID', 'TARGET_ID'], ascending=[True, True])
    
    # eliminate duplicate spots/rows if there is no division between frame at time 0 and frame at time -t
    if track_df_zero.shape[0] > track_df_minust.shape[0]:
        track_df_zero = track_df_zero.drop_duplicates(subset=['SOURCE_ID'])
    
    # extract x and y coordiantes as a matrix
    xy_zero = track_df_zero[['POSITION_X_EXACT', 'POSITION_Y_EXACT']].values.T
    xy_minust = track_df_minust[['POSITION_X_EXACT', 'POSITION_Y_EXACT']].values.T
    
    return (xy_zero - xy_minust) / t
    
def acceleration(track_df, zero_frame, t):
    """
    
    returns (2 x numberofcells) matrix containing acceleration of tracked cells
    between frame 'zero_frame' + 1 and frame 'minust_frame'.
        - row 0: cell acceleration dimension x
        - row 1: cell acceleration dimension y 
    """
    
    # sort dataframes according to track and target id
    track_df_t = track_df.loc[track_df['FRAME'] == zero_frame+1].sort_values(['TRACK_ID', 'TARGET_ID'], ascending=[True, True])
    track_df_zero = track_df.loc[track_df['FRAME'] == zero_frame].sort_values(['TRACK_ID', 'TARGET_ID'], ascending=[True, True])
    track_df_minust = track_df.loc[track_df['FRAME'] == zero_frame-1].sort_values(['TRACK_ID', 'TARGET_ID'], ascending=[True, True])

    # eliminate duplicate rows if there is no division between frame at time t and frame at time 0
    if track_df_t.shape[0] > track_df_zero.shape[0]:
        track_df_t = track_df_t.drop_duplicates(subset=['SOURCE_ID'])
    
    originids = track_df_zero['ORIGIN_ID'].values.astype(int).tolist()  # origin ids of cells at t=0

    # extract x and y coordiantes as a matrix at t=-t, t=0 and t=t
    xy_minust = np.array( [ np.array([track_df_minust.loc[track_df_minust['SOURCE_ID'] == ID].POSITION_X_EXACT.values[0] for ID in originids ]),
                            np.array([track_df_minust.loc[track_df_minust['SOURCE_ID'] == ID].POSITION_Y_EXACT.values[0] for ID in originids ]) ]
                        )
    xy_zero = track_df_zero[['POSITION_X_EXACT', 'POSITION_Y_EXACT']].values.T 
    xy_t = track_df_t[['POSITION_X_EXACT', 'POSITION_Y_EXACT']].values.T
    
    return (xy_t + xy_minust - 2*xy_zero) / (t)**1.5

#%% Analyse displacement over time

def displ_slidingwindow(first_frame, dim, bin_width=7):
    """
    returns plot of displacement densities as sliding window between frame 
    'first_frame' and 9 consecutive frames and t-test results

    """
    print('-')
    print('Assessing deviation of displacement from zero-mean distribution with t-tests')
    print('     H0: mean of displacements is not different from 0')
    print(' Test results:')
    
    if dim == 0:
        dim_name = 'x'
    else:
        dim_name = 'y'
        
    last_frame=first_frame+9
    # adjust last_frame if last frame in track_df is reached
    if last_frame > track_df['FRAME'].unique().max():
        last_frame = track_df['FRAME'].unique().max()
        
    custom_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'yellow', 'darkblue', 'pink']
    fig = plt.figure()
    for frame in range(first_frame,last_frame):
        color = custom_colors[frame - first_frame]                  # select color from custom colors
        displ_i = displacement(track_df, frame, frame+1, t)
        ttest = stats.ttest_1samp(displ_i[dim], 0)                  # test whether significantly different from zero-mean distribution
        print(f'    Frame: {frame}-{frame + 1}')
        print('         test statistic: %e   p-value: %e' %(ttest[0], ttest[1]))
        num_bins = int((displ_i[dim].max() - displ_i[dim].min()) / bin_width)
        plt.hist(displ_i[dim], bins=num_bins, color=color, alpha=0.4, density=True, label=f'Frame {frame}-{frame + 1}')
        plt.title(f'Displacement Empirical Density - Frames[{first_frame},{last_frame}]')
        plt.xlabel(f'Displ ({dim_name} dim.)')
        plt.ylabel('Density')
    plt.grid(None)
    plt.legend() 
    plt.show

#%% Analyse accel over time

def accel_slidingwindow(first_frame, dim, bin_width=7):
    """
    returns plot of acceleration densities as sliding window between frame 
    'first_frame' and 10 consecutive frames and t-test results

    """
    print('-')
    print('Assessing deviation of acceleration from zero-mean distribution with t-tests')
    print('     H0: mean of accelerations is not different from 0')
    print(' Test results:')

    if dim == 0:
        dim_name = 'x'
    else:
        dim_name = 'y'
        
    last_frame=first_frame+10
    # adjust last_frame if last frame in track_df is reached
    if last_frame > track_df['FRAME'].unique().max():
        last_frame = track_df['FRAME'].unique().max()
        
    custom_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'yellow', 'darkblue', 'pink']
    fig = plt.figure()
    for frame in range(first_frame+1,last_frame):
        accel_i = acceleration(track_df, frame, t)
        ttest = stats.ttest_1samp(accel_i[dim], 0)                  # test whether significantly different from zero-mean distribution
        print(f'    Frame: {frame -1}-{frame}-{frame + 1}')
        print('         test statistic: %e   p-value: %e' %(ttest[0], ttest[1]))
        color = custom_colors[frame-1 - first_frame]                # select color from custom colors
        num_bins = int((accel_i[dim].max() - accel_i[dim].min()) / bin_width)
        plt.hist(accel_i[dim], bins=num_bins, color=color, alpha=0.4, density=True, label=f'Frame {frame -1}-{frame}-{frame + 1}')
        plt.title(f'Acceleration Empirical Density - Frames[{first_frame},{last_frame}]')
        plt.xlabel(f'Accel ({dim_name} dim.)' )
        plt.ylabel('Density')
    plt.grid(None)
    plt.legend() 
    plt.show

def accel_slidingwindow_every5(dim, bin_width=7):
    """
    returns plot of acceleration densities as sliding window for every fifth 
    frame in a 40-frame sequence and t-test results

    """
    
    print(' Test results of acceleration at every fifth frame in 40-frame sequence:')

    if dim == 0:
        dim_name = 'x'
    else:
        dim_name = 'y'
    
    custom_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'yellow', 'pink']
    fig = plt.figure()
    for frame in range(38, 0,-5):
        accel_i = acceleration(track_df, frame, t)
        ttest = stats.ttest_1samp(accel_i[dim], 0)                    # select color from custom colors
        print('     Frame: %d' %frame)
        print('         test statistic: %e   p-value: %e' %(ttest[0], ttest[1]))
        color = custom_colors[frame // 5]
        num_bins = int((accel_i[dim].max() - accel_i[dim].min()) / bin_width)
        plt.hist(accel_i[dim], bins=num_bins, color=color, alpha=0.4, density=True, edgecolor='none', label=f'Frame {frame}')
        plt.title('Acceleration at every fifth frame')
        plt.xlabel(f'Accel ({dim_name} dim.)')
        plt.ylabel('Density')
        # uncomment next line to adjust x scale for better visualization
        #plt.xlim(-200,200) 
    plt.legend(loc='upper right') 
    plt.grid(None)
    plt.show 
    print('-')


#%% getting output of whole script

if( __name__ == "__main__" ) :
    # specify variables for plots
    t = 0.5                                     # time interval between consecutive frames
    first_frame = 0                             # start sliding window at this frame
    dim = 1                                     # dim = [0,1] (0 = dimension x, 1 = dimension y)
    
    # plots to evaluate distribution of displ and accel over time
    displ_slidingwindow(first_frame, dim, bin_width=7)
    accel_slidingwindow(first_frame, dim, bin_width=7)
    if track_df['FRAME'].unique().size == 40:
        accel_slidingwindow_every5(dim, bin_width=7)
