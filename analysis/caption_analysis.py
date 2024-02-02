import os
import argparse
import enum
import json
import string
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from transformers import AutoTokenizer


class View(enum.Enum):
    Overhead = 'overhead_view'
    Vehicle = 'vehicle_view'

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caption_dir",
        type=str,
        default="./data/caption/train",
        help="Path to the directory containing WTS caption",
    )

    parser.add_argument(
        "--ngram",
        type=int,
        default=1,
        help="ngram to analysis word frequency",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis/output",
        help="Path to the directory containing WTS caption",
    )

    parser.add_argument(
        "--view",
        type=View,
        default=View.Overhead,
        choices=list(View),
        help="Point of view of video to analyze",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/mt5-base",
        help="Huggingface tokenizer name to analyze",
    )

    parser.add_argument("--stopword",
                        action="store_true")

    return parser.parse_args()


def run_analysis(output_dir, caption_dir, model, view, stop_word, ngram):
    print("Start processing...")
    word = {
        "pedestrian": {},
        "vehicle": {}
    }
    len_seq = {
        "pedestrian": [],
        "vehicle": []
    }
    if stop_word:
        try:
            stopword_list = stopwords.words('english')
        except:
            import nltk
            nltk.download('stopwords')
            stopword_list = stopwords.words('english')
    else:
        stopword_list = []

    t5_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, legacy=False)

    for vid in tqdm(os.listdir(caption_dir)):
        caption_path = os.path.join(caption_dir, vid, view)
        try:
            for caption_file in os.listdir(caption_path):
                file_path = os.path.join(caption_path, caption_file)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    events = data["event_phase"]

                    for event in events:
                        for key in len_seq.keys():
                            event_key = f"caption_{key}"
                            token = t5_tokenizer.tokenize(event[event_key])

                            len_seq[key].append(len(token))

                            clean_caption = event[event_key].lower().translate(str.maketrans('', '', string.punctuation))
                            word_list = [w for w in clean_caption.split(" ") if w not in stopword_list]
                            for i in range(len(word_list) - ngram + 1):
                                this_ngram = " ".join(word_list[i:i+ngram])
                                word[key][this_ngram] = word[key].setdefault(this_ngram, 0) + 1

        except FileNotFoundError as e:
            # print(e)
            continue

    len_seq_result = {}

    for k, v in len_seq.items():
        len_seq_result[k] = {}

        len_seq_result[k]["mean"] = np.mean(len_seq[k]).item()
        len_seq_result[k]["std"] = np.std(len_seq[k]).item()
        len_seq_result[k]["sum"] = np.sum(len_seq[k]).item()
        len_seq_result[k]["max"] = np.max(len_seq[k]).item()
        len_seq_result[k]["min"] = np.min(len_seq[k]).item()
        len_seq_result[k]["word"] = dict(sorted(word[k].items(), key=lambda item: item[1], reverse=True))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"caption_analysis_{ngram}-gram.json")
    with open(output_path, 'w') as file:
        json.dump(len_seq_result, file, indent=2)

    print("analysis successfully")


if __name__ == '__main__':
    """
        python ./analysis/caption_analysis.py
    """
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    CAPTION_DIR = args.caption_dir
    MODEL = args.model
    VIEW = str(args.view)
    STOPWORD = args.stopword
    NGRAM = args.ngram

    run_analysis(OUTPUT_DIR, CAPTION_DIR, MODEL, VIEW, STOPWORD, NGRAM)
