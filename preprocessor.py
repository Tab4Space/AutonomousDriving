import csv
import os
import glob
import numpy as np

def makeCSV(rootpath, csvName):
    SUBDIR_LIST = iter(os.listdir(rootpath))

    wf = open(csvName, mode='w', encoding='utf-8', newline='')
    csv_writer = csv.writer(wf)

    for i in range(len(os.listdir(rootpath))):
        spawn = next(SUBDIR_LIST)

        with open(os.path.join(rootpath, spawn, 'action.csv'), mode='r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)

            for filename in glob.iglob(rootpath+spawn+'/**/**.png', recursive=True):
                steer = np.float32(next(csv_reader)[0])
                steerNorm = round((steer + 1.0) / 2.0, 4)
                line = [filename.replace('\\', '/'), steerNorm]
                csv_writer.writerow(line)
    wf.close()
    
            