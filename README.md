# Fitting Brownian motion parameters to glioblastoma cell data

## Project Description
This project focused on fitting the **drag** (beta) and **scaling** (sigma_mot) parameter from a Brownian motion model to describe the **motion of patient-derived glioblastoma cells**. This work was motivated by fact that the deep learning based cell tracking algorithm Btrack was not successful in recreating cell lineage trees from glioblastoma time-lapse microscopy images. In the future, the **fitted Brownian motion parameters can be used to inform the global motion parameters of Btrack** in order to improve its performance in lineage tree construction.
Three parameter estimation approaches were investigated to fit the parameters beta and sigma_mot to a sample of manually tracked cells in Fiji from time-lapse microscopy images. 

## Data
The files were written to read in csv-files which are exported by Fiji after manually tracking cells with the TrackMate plugin. There were **two multiple-file datasets** available to this project with quantitative information on manually tracked cells from cell line G337. The first dataset, the **long-term dataset**, contains information on **15 cells** being tracked over **40 frames**. The second dataset, the **short-term dataset**, includes information on **50 cells** being tracked across the last **3 frames** of the 40-frame sequence. Each dataset is composed of two files, a spot.csv and edges.csv file. 

## Structure of files
Before the parameters can be fit to the cell data, the files of folder *2_datapreparation* have to be used to preprocess the raw csv-files from manual tracks in Fiji. First, file *extracring_centerofcellmass.py* can be used to update the cell postions in spots.csv if manually segmented frames are available for the same frame sequence. Second, file *mergin_csvfiles.py* reads in the updated spots.csv and raw edges.csv and merges the to two combine information in both dataframes. In both files, the final dataframes have to be saved by uncommenting the specified line towards the end.

To estimate beta and sigma_mot, the files in *3_parameterestimation* have to be used:
- *brownianmotionparameterestimation_MLE.py*: fits the parameter with maximum likelihod to cell data from the merged dataframe of the **short-term dataset**.
- *brownianmotionparameterestiamtion_JPD.py*: fits the parameters by sampling from the joint posterior distribution and using cell data from the merged dataframe of the     **sort-term dataset**.
- *brownianmotionparameterestiamtion_MSD.py*: fits the parameters to approximate the MSD curve using cell data from the merged dataframe of the **long-term dataset**.
Each of those files can be run independently after performing the data processing steps. Each file also generates diagnostic plots to evaluate the reliability of parameter estimates by each method.

Finally, *4_analysis* includes the file *displ_accel_overtime.py*, which returns density plots of displacements and accelerations of cells in the **long-term dataset** over time to assess how or whether the density shapes change over time.

## Sources
[1] Chandrasekhar
[2]
