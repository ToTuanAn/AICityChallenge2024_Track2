import json
import pandas as pd

def read(file_path):
    f = open(file_path)
    data = json.load(f)
    o_data = {"sentence": [], "gt": []}
    for k, v in data["results"].items():
        o_data["sentence"].append(v["sentence"])
        o_data["gt"].append(v["gt"])

    o_data_pandas = pd.DataFrame.from_dict(o_data)
    o_data_pandas.to_csv('./datasets/BART_denoising_data/train.csv', index=False)

read("./datasets/BART_denoising_data/wts_val_preds.json")