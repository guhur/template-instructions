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
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tap


class Arguments(tap.Tap):
    tpl: Path
    filler: Path
    output: Path
    mode: Literal['soon', 'reverie']
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
    return [other_objects[i] for i, d in closest[:5]]

LARGE = ['big', 'huge', 'sizeable', 'substantial', 'immense', 'enormous', 'colossal', 'massive', 'mammoth']
# vast cosmic  goodly prodigious tremendous gigantic giant monumental stupendous


def detect_attributes(obj):
    size = np.prod(obj['3d_size'])
    if size > 0.5:
        return [random.choice(LARGE)]
    return []

    
class FillInTheBlankDataset(Dataset):
    """
    Load the preprocessed data and produce a list of fillers.
    """
    def __init__(self, filler: Path, template: Path):
        with open(filler) as fid:
            self.samples = json.load(fid)
        with open(template) as fid:
            self.templates = json.load(fid)

        # "scanvp": vp['scanvp'],
        # "view_id": obj["view_id"],
        # "name": "cabinet",
        # "neighbors": detect_neighbors(obj, vp['bboxes']),
        # "attributes": detect_attributes(obj),

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        obj = self.samples[index]
        sentences = []
        for i in range(3):
            template = random.choice(self.templates)
            sentences.append(
                template.replace("[object]", obj["name"])
                .replace("[attr]", " ".join(obj["attributes"]))
                .replace("[rel_loc]", f"next to {random.choice(obj['neighbors'])}")
                .replace("[room]", "in this room")
                .replace("[rel_room]", "which is close to that room")
            )

        return {
            "instructions": sentences,
            "scanvp": obj["scanvp"],
            "obj_id": obj["obj_id"],
            }
    
def export_fillers(fillers, output):
    with open(output, "w") as fid:
        json.dump(fillers, fid)
    
    
if __name__ == "__main__":

    args = Arguments().parse_args()
    print(args)

    dataset = FillInTheBlankDataset(args.filler, args.tpl)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=1, collate_fn=lambda x: x)

    fillers = []
    for item in tqdm(dataloader):
        fillers += item

    print(f"Found {len(dataset)}")

    export_fillers(fillers, args.filler_dir)
