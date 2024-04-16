from datasets import CaptioningDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

CaptioningDataset.make_coco_captioning_dataset_index(
    data_path="/path/to/your_data",
)
