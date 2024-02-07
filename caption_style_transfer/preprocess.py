import json
import pandas as pd

def read(file_paths):
    o_data = {"sentence": [], "gt": []}
    for file_path in file_paths:
        f = open(file_path)
        data = json.load(f)
        print(len(data["results"]))
        for k, v in data["results"].items():
            o_data["sentence"].append(v["sentence"])
            o_data["gt"].append(v["gt"])

    o_data_pandas = pd.DataFrame.from_dict(o_data)
    o_data_pandas.to_csv('./datasets/BART_denoising_data/train.csv', index=False)

def read_csv(file_path):
    df = pd.read_csv(file_path)

    for x in df['sentence']:
        print(x)

# read([
#     "./datasets/BART_denoising_data/wts_val_preds.json",
#     "./datasets/BART_denoising_data/wts_val_preds_1.json",
# ])
    
read_csv("./datasets/BART_denoising_data/val.csv")