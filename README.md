# Fitting Brownian motion parameters to glioblastoma cell data

## Project Description
This project focused on fitting the **drag** $\beta$ and **scaling** $\sigma_{mot}$ parameter from a Brownian motion model [1] to describe the **motion of patient-derived glioblastoma cells**. This work was motivated by the previous study [2] on the same cell lines where the deep learning based cell tracking algorithm Btrack was not successful in recreating cell lineage trees from glioblastoma time-lapse microscopy images. In the future, the **fitted Brownian motion parameters can be used to inform the global motion parameters of Btrack** in order to improve its performance in lineage tree construction.
Three parameter estimation approaches were investigated to fit the parameters beta and sigma_mot to a sample of manually tracked cells in Fiji from time-lapse microscopy images. 

## Data
The code files were written to read in csv-files which are exported by Fiji after manually tracking cells on microsocopy images with the TrackMate plugin. There were **two multiple-file datasets** available to this project with quantitative information on manually tracked cells from cell line G337. The first dataset, the **long-term dataset**, contains information on **15 cells** being tracked over **40 frames**. The second dataset, the **short-term dataset**, includes information on **50 cells** being tracked across the last **3 frames** of the 40-frame sequence. Each dataset is composed of two files, a spot.csv and edges.csv file. Additionally, the 40-frame sequence was also available as segmented tif-files, whereby each frame was manually segmented for individual cells by Jonathan Fung.

This is a microscopy example of the first frame of the 40-frame sequence including the manual tracks over the whole sequence. The white dots represent the cell nuclei and each color represents a different cell track.

![Alt Text](G337_40_frames_manualtracks_firstframe.tiff)

## Structure of files
Before the parameters can be fit to the tracking data, the files of folder *2_datapreparation* have to be used to preprocess the raw csv-files of both datasets. First, file *extracring_centerofcellmass.py* can be used to update the cell postions in spots.csv with the manually segmented frames of the same sequence. The segmented frames can be found in subfolder *manualsegmentationsbyJonathan* and should ideally be saved in a separate folder. To update spots.csv of the  **long-term dataset, all segmented frames are imported from the folder** by default when running the script. To update spots.csv of the **short-term dataset, the name of the corresponding frames have to be manually specified** before running the script in the specified lines. For the updated cell position, two new columns are added ('Position_X_EXACT', 'Position_Y_EXACT') to the dataframe. Second, file *merging_csvfiles.py* reads in the updated spots.csv and raw edges.csv and merges the two to combine information in both dataframes. In both py-files, the final dataframes have to be saved by uncommenting the specified line towards the end of the code.

To estimate $\beta$ and $\sigma_{mot}$, the files in *3_parameterestimation* have to be used which also contain contributions by Peter Embacher:
- *brownianmotionparameterestimation_MLE.py*: fits the parameter with maximum likelihod to cell data from the merged dataframe of the **short-term dataset**. 
- *brownianmotionparameterestiamtion_JPD.py*: fits the parameters by sampling from the joint posterior distribution and using cell data from the merged dataframe of the     **short-term dataset**.
- *brownianmotionparameterestiamtion_MSD.py*: fits the parameters to approximate the MSD curve using cell data from the merged dataframe of the **long-term dataset**.

Each of those files can be run independently after performing the data processing steps and details on the estimation procedure can be found in [3], [4]. A quick note at the beginning of each file outlines the manual changes required before running the file, which mostly only involve adjusting the file name or directory. Each file also generates diagnostic plots to evaluate the reliability of parameter estimates by each method.

Finally, *4_analysis* includes the file *displ_accel_overtime.py*, which returns density plots of displacements and accelerations of cells in the **long-term dataset** over time to assess how or whether the density shapes change over time. This file can also be run independently after performing the data processing steps.

All of the scripts in *3_parameterestimation* and *4_analysis* run conditioned on the preprocessing steps, meaning that the scripts use the columns 'Position_X_EXACT' and 'Position_Y_EXACT' instead of the initial cell positions in columns 'POSITION_X' and 'POSITION_Y'. In case a new sample of cells is manually tracked in the future and no manually segmented frames are available to update the positions, the scripts can still be run on raw spots.csv and edges.csv files. For this case, 'Position_X_EXACT' and 'Position_Y_EXACT' would need to be replaced by 'POSITION_X' and 'POSITION_Y' in all the code (on Mac, use CMD + R to replace all occurences in one go).

## Sources/PDFs
[1] Chandrasekhar, S. (1943). Stochastic problems in physics and astronomy (Chapter 2). Reviews of modern physics, 15(1), 1.
[2] Braaksma, J. (2023). Comparative analysis of segmentation algorithms and informing of tracking parameters from integrated single cell imaging. Cancer Biomedicine Research Project - University College London.
[3] Embacher, P. (2023). About how to extract drift, diffusion and measurement error from tracking data. University College London.
[4] Amrein, A. (2023). Extracting and describing cell fate dynamics in glioblastoma from live-cell imaging. MSc Project - University College London.
