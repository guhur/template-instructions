"""
This file extracts metadata for each vp in HM3D.
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
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer as AllenTokenizer
import allennlp_models.structured_prediction
from allennlp_models.pretrained import load_predictor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tap
from difflib import SequenceMatcher


class Arguments(tap.Tap):
    data_dir: Path
    output: Path
    num_workers: int = 20


def load_jsonl(filename: Union[str, Path]):
    with open(filename, "r") as fid:
        lines = [json.loads(line.strip()) for line in fid.readlines()]
    return lines

    
def detect_neighbors(obj, other_objects):
    distances = []
    center = np.array(obj['3d_center'])
    for oth in other_objects:
        if oth == obj: 
            continue
        distances.append(((np.array(oth['3d_center']) - center) ** 2).sum())
    closest = sorted(enumerate(distances), key=itemgetter(1))
    return [other_objects[i]['name'] for i, d in closest[:5]]

LARGE = ["large", 'big', 'huge', 'sizeable', 'substantial', 'immense', 'enormous', 'colossal', 'massive',  "vast",  "giant"]
SMALL = ["little", "small", "compact", "tiny", "mini"]

def detect_attributes(obj):
    size = np.prod(obj['3d_size'])
    if size > 0.5:
        return [random.choice(LARGE)]
    if size < 0.1:
        return [random.choice(SMALL)]
    return []
    
def detect_room(obj):
    rooms = ['kitchen', 'living room', 'bathroom', 'bedroom', 'garage', 'office', 'hallway']
    return random.choice(rooms)

def detect_level(obj):
    z = obj['3d_center'][2]
    if z < -0.1:
        return 0
    if z > 2:
        return 2
    return 1

class FillerDataset(Dataset):
    """
    Load the preprocessed data and produce a list of fillers.
    """
    def __init__(self, data_dir: Path):
        self.samples = []
        for data in data_dir.iterdir():
            for scan in data.iterdir():
                if not scan.is_dir():
                    continue
                vps = load_jsonl(scan / "view_bboxes_merged_by_3d.jsonl")
                for vp in vps:
                    for obj in vp["bboxes"]:
                        self.samples.append({
                            "scanvp": vp['scanvp'],
                            "view_id": obj["view_id"],
                            "obj_id": obj["obj_id"],
                            "level": detect_level(obj),
                            "name": obj['name'],
                            "room": detect_room(obj),
                            "neighbors": detect_neighbors(obj, vp['bboxes']),
                            "attributes": detect_attributes(obj),
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        # Augment samples with room detection, neighbors, and so on...
        return self.samples[index]
    
def export_fillers(fillers, output):
    with open(output, "w") as fid:
        json.dump(fillers, fid, indent=2)
    
    
if __name__ == "__main__":

    args = Arguments().parse_args()
    print(args)

    dataset = FillerDataset(args.data_dir)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=1, collate_fn=lambda x: x)

    fillers = []
    for item in tqdm(dataloader):
        fillers += item

    print(f"Found {len(dataset)}")

    export_fillers(fillers, args.output)
