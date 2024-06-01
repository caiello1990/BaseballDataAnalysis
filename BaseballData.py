import pandas as pd
import matplotlib.pyplot as plt
import pybaseball, csv, os
from pybaseball.utils import split_request, sanitize_input

filename = "BaseballData.csv"
fileExists = os.path.exists(filename)
def downloadToCSV(name_of_file, start_date, end_date):
    statcast_data = pybaseball.statcast(start_dt=start_date, end_dt=end_date)
    statcast_data.to_csv(name_of_file, index=False)

def handlePitchByPitch():
    pitchByPitch_df = pd.read_csv(filename)
    batterIDs = pitchByPitch_df['batter']

    batterObjects = pybaseball.playerid_reverse_lookup(batterIDs)
    pitchByPitch_df['batter_name'] = getBatterNameCol(batterObjects, batterIDs)
    print(pitchByPitch_df.head())
def getBatterNameCol(batterObjects = None, batterIDs = None):
    nameDict = getBatterNameDict(batterObjects)
    batterNames = []
    for id in batterIDs:
        batterNames.append(nameDict.get(id))
    return batterNames



def getBatterNameDict(batterObjects):
    batterNameDict = {}
    for index, batter in batterObjects.iterrows():
        batter_fullName = (batter.name_last + ', ' + batter.name_first).title()
        batterNameDict[batter.key_mlbam] = batter_fullName
    return batterNameDict

if fileExists:
    handlePitchByPitch()
else:
    downloadToCSV(filename, "2023-06-25", "2023-07-04")




