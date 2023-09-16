# Fitting Brownian motion parameters to glioblastoma cell data

## Project Description
This project focused on fitting the **drag** (beta) and **scaling** (sigma_mot) parameter from a Brownian motion model to describe the **motion of patient-derived glioblastoma cells**. This work was motivated by fact that the deep learning based cell tracking algorithm Btrack was not successful in recreating cell lineage trees from time-lapse microscopy images. In the future, the **fitted Brownian motion parameters can be used to inform the global motion parameters of Btrack** in order to improve its performance in lineage tree construction.
Three parameter estimation approaches were investigated to fit the parameters beta and sigma_mot to a sample of manually tracked cells in Fiji from live-cell imaging. 

## Data
The files were written to read in csv-files which are exported by Fiji after manually tracking cells with the TrackMate plugin. There were **two datasets** with quantitative information on manually tracked cells from cell line G337 available to this project. The first dataset, the **long-term dataset**, contains information on **15 cells** being tracked over **40 frames**. The second dataset, the **short-term dataset**, includes information on **50 cells** being tracked across the last **3 frames** of the 40-frame sequence.

## Structure of files
Before the parameters can be fit to the cell data, the files of folder *2_datapreparation* have to be used to preprocess the csv-files from manual tracks in Fiji. First, file *extracring_centerofcellmass.py* can be used to update the cell postions in spots.csv if manually segmented frames are available for the same frame sequence. The updated dataframe needs to be saved manually. Second, file *mergin_csvfiles.py* reads in the updated spots.csv and raw edges.csv and merges the two combine information in both dataframes.




## Sources
...
