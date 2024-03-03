import os
import json
import torch
import random
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from args import get_args_parser
from model import build_vid2seq_model, _get_tokenizer

from rules_engine.rules_executor import RulesExecutor

class TestDataset(Dataset):
    def __init__(self,
                 json_path,
                 features_path,
                 max_feats=100,
                 features_dim=768):
        self.data = json.load(open(json_path, 'r'))
        self.vids = list(self.data.keys())

        self.features = None
        self.features_path = None
        if os.path.isdir(features_path):
            self.features_path = features_path
        else:
            self.features = torch.load(features_path)

        self.max_feats = max_feats
        self.features_dim = features_dim

    def __len__(self):
        return len(self.data)

    def _get_video(self, video_id):
        if self.features is not None:
            assert video_id in self.features, video_id
            video = self.features[video_id].float()
        else:
            features_path = os.path.join(
                self.features_path, video_id + '.mp4.npy')
            if not os.path.exists(features_path):
                features_path = os.path.join(
                    self.features_path, video_id + '.npy')
            assert os.path.exists(features_path), features_path
            video = torch.from_numpy(np.load(features_path)).float()

        return video

    def pad_video(self, video):
        if self.max_feats == 1:
            tmp = video[len(video) // 2: len(video) // 2 + 1]
            if len(tmp):
                # middle frame
                return video[len(video) // 2: len(video) // 2 + 1]
            else:
                return torch.zeros(self.max_feats, self.features_dim)
        if len(video) >= self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat(
                [video, torch.zeros(
                    self.max_feats - video_len, self.features_dim)], 0
            )
        return video

    def __getitem__(self, idx):
        video_id = self.vids[idx]
        annotations = self.data[video_id]

        video = self._get_video(video_id)
        video = torch.stack([self.pad_video(video[int(x[0]): int(x[1]) + 1])
                            for x in annotations['timestamps']])

        text = ['' for _ in annotations['sentences']]
        label = [label for label in annotations['labels']]

        out = {
            'video_id': video_id,
            'video': video,
            'input_text': text,
            'label' : label
        }

        return out


def test_collate_fn(batch):
    bs = len(batch)
    video_id = [batch[i]['video_id'] for i in range(bs)]
    video = torch.stack([batch[i]['video'] for i in range(bs)])
    input_text = [batch[i]['input_text'] for i in range(bs)]
    label = [batch[i]['label'] for i in range(bs)]
    out = {
        'video_id': video_id,
        'video': video,
        'input_text': input_text,
        'label': label
    }
    return out


@torch.no_grad
def infer_by_phases(model,
          phase,
          tokenizer,
          dataloader,
          device):

    model.to(device)
    model.eval()

    res = {}

    for i, batch_dict in enumerate(dataloader):
        # batch_size_val must be 1
        print(f"Processing video {str(i)} / {str(len(dataloader))}") 
    
        phase_idx = 100
        for idx, label in enumerate(batch_dict['label'][0]):
            if label == str(phase):
                phase_idx = idx
        
        input_text = [batch_dict['input_text'][0][phase_idx]]
        input_tokenized = tokenizer(input_text,
                                    padding='longest',
                                    truncation=True,
                                    max_length=args.max_input_tokens,
                                    return_tensors='pt').to(device)

        print("Phase idx: ", phase_idx)
        print("Len: ", len(batch_dict['video'][0]))
        print("Len of phases: ", len(batch_dict['video'][0][phase_idx]))
        
        video = torch.from_numpy(np.array([batch_dict['video'][0][phase_idx]]))
        print("Len of video: ", len(video))
        video = video.to(device)

        output = model.generate(video=video,
                                input_tokenized=input_tokenized,
                                use_nucleus_sampling=args.num_beams == 0,
                                num_beams=args.num_beams,
                                max_length=args.max_output_tokens,
                                min_length=1,
                                top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty,
                                length_penalty=args.length_penalty,
                                num_captions=1,
                                temperature=1)
        
        print(output)

        video_id = batch_dict['video_id'][0][phase_idx]
        clip_ids = [video_id + "#" + str(phase)]

        for clip_id, pred in zip(clip_ids, output):
            res[clip_id] = pred

    return res


def main(args):
    device = torch.device(args.device)

    # Fix seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build dataloader
    dataset_test = TestDataset(json_path=args.wts_test_json_path,
                               features_path=args.wts_features_path,
                               max_feats=args.max_feats,
                               features_dim=args.features_dim)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=args.batch_size_val,
                                 sampler=sampler_test,
                                 collate_fn=test_collate_fn,
                                 num_workers=args.num_workers)

    # Load pretrained tokenizer
    args.num_bins = 0
    tokenizer = _get_tokenizer(args.model_name, args.num_bins)

    # Load pretrained model
    model = build_vid2seq_model(args, tokenizer)

    for phase in range(0, 5):
        checkpoint = None

        if str(phase) == "0":
            checkpoint = torch.load(args.load_internal_ckpt_phase_0, map_location='cpu')
        elif str(phase) == "1":
            checkpoint = torch.load(args.load_internal_ckpt_phase_1, map_location='cpu')
        elif str(phase) == "2":
            checkpoint = torch.load(args.load_internal_ckpt_phase_2, map_location='cpu')
        elif str(phase) == "3":
            checkpoint = torch.load(args.load_internal_ckpt_phase_3, map_location='cpu')
        elif str(phase) == "4":
            checkpoint = torch.load(args.load_internal_ckpt_phase_4, map_location='cpu')
        else:
            raise NotImplementedError

        # Remove time tokens
        if 't5_model.shared.weight' in checkpoint['model']:
            checkpoint['model']['t5_model.shared.weight'] = checkpoint['model']['t5_model.shared.weight'][:32100]
            checkpoint['model']['t5_model.encoder.embed_tokens.weight'] = checkpoint['model']['t5_model.encoder.embed_tokens.weight'][:32100]
            checkpoint['model']['t5_model.decoder.embed_tokens.weight'] = checkpoint['model']['t5_model.decoder.embed_tokens.weight'][:32100]
            checkpoint['model']['t5_model.lm_head.weight'] = checkpoint['model']['t5_model.lm_head.weight'][:32100]
        model.load_state_dict(checkpoint['model'], strict=False)

        preds = infer_by_phases(model,
                    phase,
                    tokenizer,
                    dataloader_test,
                    device)
        
        if str(phase) == "0":
            with open(args.save_phase_0, 'w') as f:
                json.dump(preds, f, indent=4)
        elif str(phase) == "1":
            with open(args.save_phase_1, 'w') as f:
                json.dump(preds, f, indent=4)
        elif str(phase) == "2":
            with open(args.save_phase_2, 'w') as f:
                json.dump(preds, f, indent=4)
        elif str(phase) == "3":
            with open(args.save_phase_3, 'w') as f:
                json.dump(preds, f, indent=4)
        elif str(phase) == "4":
            with open(args.save_phase_4, 'w') as f:
                json.dump(preds, f, indent=4)
        else:
            raise NotImplementedError
        
        print(f"Phase {str(phase)} inference finished")

    preds = {}

    pred_phase_0_file = open(args.save_phase_0)
    preds_phase_0 = json.load(pred_phase_0_file)

    pred_phase_1_file = open(args.save_phase_1)
    preds_phase_1 = json.load(pred_phase_1_file)
    
    pred_phase_2_file = open(args.save_phase_2)
    preds_phase_2 = json.load(pred_phase_2_file)

    pred_phase_3_file = open(args.save_phase_3)
    preds_phase_3 = json.load(pred_phase_3_file)
    
    pred_phase_4_file = open(args.save_phase_4)
    preds_phase_4 = json.load(pred_phase_4_file)

    preds.update(preds_phase_0) 
    preds.update(preds_phase_1) 
    preds.update(preds_phase_2) 
    preds.update(preds_phase_3) 
    preds.update(preds_phase_4) 

    print("Rule config path: ", args.rule_config_path)
    print("Rule mode: ", args.rule_mode)
    
    # rule_executor = RulesExecutor(config_path=args.rule_config_path)
    # preds = rule_executor.run(preds, rule_mode=args.rule_mode)

    print(f"Finished {args.rule_mode} rules")

    with open(args.save, 'w') as f:
        json.dump(preds, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
