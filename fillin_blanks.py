"""
This file fills in the blank over pre computed templates.
"""
from typing import Literal, Tuple, List, Iterator, Dict, Union
from dataclasses import dataclass, field
from operator import itemgetter
import functools
import json
import logging
from itertools import combinations, product
from pathlib import Path
import random
from tqdm import tqdm
import transformers
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tap


class Arguments(tap.Tap):
    tpl: Path
    filler: Path
    output: Path
    mode: Literal['soon', 'reverie']
    export: Literal['jsonl', 'json'] = 'jsonl'
    num_workers: int = 20


def load_jsonl(filename: Union[str, Path]):
    with open(filename, "r") as fid:
        lines = [json.loads(line.strip()) for line in fid.readlines()]
    return lines

    
class FillInTheBlankDataset(Dataset):
    """
    Load the preprocessed data and produce a list of fillers.
    """
    def __init__(self, filler: Path, template: Path):
        with open(filler) as fid:
            self.samples = json.load(fid)
        with open(template) as fid:
            self.templates = json.load(fid)

        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        obj = self.samples[index]
        sentences = []
        for i in range(3):
            template = random.choice(self.templates)
            neighbor = random.choice(obj['neighbors']) if obj['neighbors'] != [] else 'next to this object'
            rooms = ['kitchen', 'living room', 'bathroom', 'bedroom', 'garage', 'office', 'hallway']
            room = random.choice(rooms)
            sentences.append(
                template.replace("[object]", obj["name"])
                .replace("[attr]", " ".join(obj["attributes"]))
                .replace("[rel_loc]", neighbor)
                .replace("[room]", room)
                .replace("[level]", f"level {obj['level']}")
                .replace("[rel_room]", "which is close to that room")
            )
        
        instr_encoding = self.tokenizer(sentences)['input_ids']
        scan, vp = obj["scanvp"].split("_")

        return [{
            "instructions": stc,
            "instr_encoding": enc,
            "scan": scan,
            "pos_vps": [vp],
            "objid": obj["obj_id"],
            "instr_id": f"{index}_{i}",
            "view_id": obj["view_id"],
        } for i, (stc, enc) in enumerate(zip(sentences, instr_encoding))]
    
   
def save_json(data, output):
    with open(output, "w") as fid:
        json.dump(data, fid, indent=2)
    
def save_jsonl(data, output):
    jsonl = [
        json.dumps(sample) for sample in data
    ]
    with open(output, "w") as fid:
        fid.write("\n".join(jsonl))
    
if __name__ == "__main__":

    args = Arguments().parse_args()
    print(args)

    dataset = FillInTheBlankDataset(args.filler, args.tpl)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=1, collate_fn=lambda x: sum(x, []))

    samples = []
    for item in tqdm(dataloader):
        samples += item

    print(f"Found {len(dataset)}")

    if args.export == "json":
        save_json(samples, args.output)
    elif args.export == "jsonl":
        save_jsonl(samples, args.output)
