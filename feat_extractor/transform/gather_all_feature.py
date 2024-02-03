import glob
import numpy as np
for folder in glob.glob("./data/features/*/*/*"):
    
    for npyfile in glob.glob(f"{folder}/*.npy"):
        data = np.load(npyfile)
        print(len(data))
        np.save( f"./data/gather/{npyfile.split('/')[-1]}", data)