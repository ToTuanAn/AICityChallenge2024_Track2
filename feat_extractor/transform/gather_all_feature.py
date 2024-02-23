import glob
import numpy as np

hit = 0 
for folder in glob.glob("./data/features/test/*/*"):
    
    for npyfile in glob.glob(f"{folder}/*.npy"):
        data = np.load(npyfile)
        print(len(data))
        video_name = npyfile.split('/')[-1][:-4]

        if video_name[-4:] != "BLUR":
            video_name = video_name + ""
            hit += 1
            print("Hit")

        np.save( f"./data/gather/{video_name}.npy", data)

print("Hit count: ", hit)