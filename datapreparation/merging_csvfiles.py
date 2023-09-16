
"""
merging the two csv-files "edges" and "spots" exported by Fiji from manual tracks
and adding column "ORIGIN_ID".

Note: uncomment last line to safe merged dataframe

"""

import pandas as pd

# import dataframes
edges = pd.read_csv('G337_manualtracks/G337_3_frames_edges.csv', \
                    usecols = ['LABEL', 'TRACK_ID', 'SPOT_SOURCE_ID', 'SPOT_TARGET_ID', 'EDGE_X_LOCATION', 'EDGE_Y_LOCATION'],
                    skiprows=range(1,4))
edges.columns = ['LABEL',\
                 'TRACK_ID',\
                 'SOURCE_ID',\
                 'TARGET_ID',\
                 'EDGE_X_LOCATION',\
                 'EDGE_Y_LOCATION']
    
spots = pd.read_csv('G337_manualtracks/G337_3_frames_spots_centerofmasks.csv',\
                    usecols = ['LABEL', 'ID', 'TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_X_EXACT', 'POSITION_Y_EXACT', 'FRAME'],
                    ) #add skiprows=range(1,4) at the end in case raw spots file is imported without updated cell locations
spots.rename(columns={"ID": "SOURCE_ID"}, inplace=True)

# merge dataframes
merged = pd.merge(spots, edges, how='outer', on=["SOURCE_ID", "TRACK_ID"])
merged = merged.drop(['LABEL_x', 'LABEL_y', 'EDGE_Y_LOCATION', 'EDGE_X_LOCATION'], axis=1)

def get_previous(x, merged_df):
    """
    get spot ID of predecessor

    """
    
    current_id = x.SOURCE_ID
    y = merged_df.loc[merged_df['TARGET_ID'] == current_id] # row of target id

    if y.empty:
        return None
    else:
        return y.SOURCE_ID.values[0]
    
merged['ORIGIN_ID'] = merged.apply(get_previous, merged_df=merged, axis=1)
#merged.to_csv('G337_manualtracks/G337_3_frames_merged.csv', index=False)

