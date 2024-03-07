import os
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Download model')
    parser.add_argument('--model_id', type=str, required=True, help='Model id')
    parser.add_argument('--cache_dir', type=str, default="cache", help='Cache dir')
    parser.add_argument('--save_dir', type=str, required=True, help='Save dir')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_id = args.model_id
    cache_dir = args.cache_dir
    save_dir = args.save_dir
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="cpu", cache_dir=cache_dir)

    tokenizer.save_pretrained(os.path.join(save_dir, model_id.split("/")[1]))
    model.save_pretrained(os.path.join(save_dir, model_id.split("/")[1]))