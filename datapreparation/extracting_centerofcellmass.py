"""
Automating the extraction of center of mass coordinates from segmented cells
and computing average cell movement indicators.

Note: the manual changes are to decide whether a whole folder of segmented
        files should be imported or a predefined list of files. Those have to 
        be done in the first section "import segmentation files as tif". Also, 
        folder path and csv file name need to be adjusted accordingly.
        In the last section "getting output of whole script", frame intervals
        can be adjusted to compute mean displacement and acceleration.
        
"""

from skimage.io import imread
import numpy as np
import os
import pandas as pd

#%% Import segmentation files as tif

folder_path = 'G337_manualsegmentation/'                                # change to your folder path with segmented files

# extract file names of whole folder
tif_files = sorted( [ file for file in os.listdir(folder_path) if file.endswith('.tif') ] )

# uncomment next lines if goal is to extract specific file names only in chronological order
#tif_files = [ 'H2BtdTomato_C2_2_00d19h30m_cp_masks.tif',
#              'H2BtdTomato_C2_2_00d20h00m_cp_masks.tif',
#              'H2BtdTomato_C2_2_00d20h30m_cp_masks.tif' ]

# read in each seg file and store as list
segmentation_list = []
for file_name in tif_files:
    file_path = os.path.join(folder_path, file_name)
    segmentation = imread(file_path)
    segmentation_list.append(segmentation)

#%% Import files of manually tracked cells (from Fiji) as csv

spots = pd.read_csv('G337_manualtracks/G337_40_frames_spots.csv',         # change to desired csv file
                    usecols = ['LABEL', 'ID', 'TRACK_ID', 'POSITION_X', 'POSITION_Y', 'FRAME'],
                    skiprows=range(1,4)).sort_values(['TRACK_ID', 'FRAME'], ascending=[True, True])
    
if len(spots['FRAME'].unique()) != len(segmentation_list):
    print('WARNING: Mismatch in number of imported segmentation files and number of frames in csv, please double check and run again.')

#%% Extracting center of mass of tracked cells given the dataframe "spots"

def extractxenterofmass_trackedcells(spots):
    """
    
    returns dataframe "spots" with additional columns containing exact center of mass of masks

    """
    
    print('-')
    print('Updating cell locations in progress...')
    print('-')

    # new columns for center of mass of masks
    spots['POSITION_X_EXACT'], spots['POSITION_Y_EXACT'] = np.repeat(np.nan, spots.shape[0]), np.repeat(np.nan, spots.shape[0])
    
    # compute center of mass for all cells per frame
    for frame in np.sort(spots['FRAME'].unique()):
        
        spots_coor = spots.loc[spots['FRAME'] == frame][['POSITION_Y', 'POSITION_X']].values    # y and x coordinates from manual track
        seg_file = segmentation_list[frame]                                                     # corresponding seg file for frame
        
        # intermediate dataframe with spot id's only
        spots_exact = spots.loc[spots['FRAME']==frame, 'ID'].copy().to_frame()
        mask_center_all = np.zeros((spots_exact.shape[0], 2))
        
        # extract center of mask for each spot
        for i in range(0, spots_exact.shape[0]):
            spot = spots_coor[i]                                           # y and x coordinates for one spot only
            spot_rounded = np.round(spot).astype(int)
            mask_label = seg_file[spot_rounded[0], spot_rounded[1]]        # label of spot in segmentation file
            mask_pixels = np.array(np.where(seg_file == mask_label)).T     # all pixels with the same label
            mask_center = np.mean(mask_pixels, axis=0)
            if (spot[0] - mask_center[0] >= 10) or (spot[1] - mask_center[1] >= 10):
                print('WARNING: new center of mass of mask is more than 10 pixels away from original spot')
                print('Frame:   %d      ID: %d      Iteration: %d' %(frame, spots_exact['ID'].values[i], i))
            mask_center_all[i] = mask_center
        
        # columns in intermediate dataframe with center of mask of all spots
        spots_exact['POSITION_X_EXACT'], spots_exact['POSITION_Y_EXACT'] = mask_center_all[:,1], mask_center_all[:,0]
            
        # update columns of original dataframe with values from new dataframe
        spots.set_index('ID', inplace=True)
        spots_exact.set_index('ID', inplace=True)
        spots.update(spots_exact)
        spots.reset_index(inplace=True)
        
    print('-')
    print('Successfully updated cell locations of %d tracks aross %d frames' %(len(spots['TRACK_ID'].unique()), spots['FRAME'].unique().shape[0]))
    print('-')

    return spots
    
#%% Extracting center of mass of not tracked cells at

def extractcenterofmass_nontrackedcells(spots, frame):
    
    """
    returns (numberofcells x 2 ) matrix with center of mass of cells which were not tracked in given frame
        - column 0: cell location dimension y
        - column 1: cell location dimension x
    """
    
    # approx. location of tracked cells in given frame as (y,x) matrix
    yx_tracked = spots.loc[spots['FRAME'] == frame][['POSITION_Y', 'POSITION_X']].values
    
    # manually segmented frame
    seg_file = segmentation_list[frame]    
    
    # labels of tracked cells 
    labels_tracked = []
    for i in range(0, yx_tracked.shape[0]):                                 # for each row / tracked cell
        spot = np.round(yx_tracked[i]).astype(int)                          # extract approx. location
        mask_label = seg_file[spot[0], spot[1]]                             # label of apporx. location in segmented file
        labels_tracked.append(mask_label)
    
    # center of mass of not tracked cells
    labels_notracked = [ label for label in np.unique(seg_file)[1:] if label not in labels_tracked]
    yx_notracked = np.zeros((len(labels_notracked), 2))
    for index,label in enumerate(labels_notracked):                         # excluding label 0 for black/background
        mask_pixels = np.array(np.where(seg_file == label)).T               # all pixels with the same label, transpose to have two-column format (y, x)
        mask_center = np.mean(mask_pixels, axis=0)
        yx_notracked[index] = mask_center
        
    return yx_notracked                                            

def cellmovement_nontrackedcells(frame_minust, frame_zero, frame_t, t):
    
    """
    computes mean location, mean displacement and mean acceleration for all non-tracked
    cells in specified 3-frame sequence
    
    """
    
    # center of mass of non tracked cels for each frame
    yx_minust = extractcenterofmass_nontrackedcells(spots, frame_minust)
    yx_zero = extractcenterofmass_nontrackedcells(spots, frame_zero)
    yx_t = extractcenterofmass_nontrackedcells(spots, frame_t)
    
    # global mean location per frame
    global_minust = np.mean(yx_minust, axis=0)
    global_zero = np.mean(yx_zero, axis=0)
    global_t = np.mean(yx_t, axis=0)
    print(' Movement of non-tracked cells [dim. y, dim. x]:')
    print('     mean location at -t:           ', global_minust)
    print('     mean location at 0:            ', global_zero)
    print('     mean location at t:            ', global_t)
    
    # global mean displacement and acceleration (Equation (3.7) in pdf "how to extract..." )
    displ_zero = (global_zero - global_minust) / t
    displ_t = (global_t - global_zero) / t
    accel_minustzerot = (global_minust + global_t - 2*global_zero) / t**1.5
    print('     mean displacement [0,-t]:      ', displ_zero)
    print('     mean displacement [t, 0]:      ', displ_t)
    print('     mean acceleration [t, 0, -t]:  ', accel_minustzerot)
    print('-')


#%% mean location for tracked cells

def cellmovement_trackedcells(frame_minust, frame_zero, frame_t, t):
    
    """
    computes mean location, mean displacement and mean acceleration for all racked
    cells in specified 3-frame sequence
    
    """
    
    # center of mass of tracked cels for each frame
    yx_minust_tracked = spots.loc[spots['FRAME'] == frame_minust][['POSITION_Y_EXACT', 'POSITION_X_EXACT']].values
    yx_zero_tracked = spots.loc[spots['FRAME'] == frame_zero][['POSITION_Y_EXACT', 'POSITION_X_EXACT']].values
    yx_t_tracked = spots.loc[spots['FRAME'] == frame_t][['POSITION_Y_EXACT', 'POSITION_X_EXACT']].values
    
    # mean location per frame
    global_minust_tracked = np.mean(yx_minust_tracked, axis=0)
    global_zero_tracked = np.mean(yx_zero_tracked, axis=0)
    global_t_tracked = np.mean(yx_t_tracked, axis=0)
    print('Assessing  movement of cells across frames %d - %d:' %(frame_minust, frame_t))
    print('     -t = %d' %(frame_minust))
    print('      0 = %d' %(frame_zero))
    print('      t = %d' %(frame_t))
    print('  ')
    print(' Movement of tracked cells [dim. y, dim. x]:')
    print('     mean location at -t:            ', global_minust_tracked)
    print('     mean location at 0:             ', global_zero_tracked)
    print('     mean location at t:             ', global_t_tracked)
    
    # mean displacement and acceleration (Equation (3.7) in pdf "how to extract..." )
    displ_zero_tracked = (global_zero_tracked - global_minust_tracked) / t
    displ_t_tracked = (global_t_tracked - global_zero_tracked) / t
    accel_minustzerot_tracked = (global_minust_tracked + global_t_tracked - 2*global_zero_tracked) / t**1.5
    print('     mean displacement [0,-t]:       ', displ_zero_tracked)
    print('     mean displacement [t, 0]:       ', displ_t_tracked)
    print('     mean acceleration [t, 0, -t]:   ', accel_minustzerot_tracked )
    print('  ')

#%% getting output of whole script

if( __name__ == "__main__" ) :
    spots = extractxenterofmass_trackedcells(spots)                      # save this dataframe with updated cell locations
    
    # specify variables to compute global movement indicators
    frame_minust = 37
    frame_zero = 38
    frame_t = 39
    deltat = 0.5                                                         # time between two frames in hours
    
    cellmovement_trackedcells(frame_minust=frame_minust, frame_zero=frame_zero, frame_t=frame_t, t=deltat)
    cellmovement_nontrackedcells(frame_minust=frame_minust, frame_zero=frame_zero, frame_t=frame_t, t=deltat)
