"""
This file extracts a fill-in-the-blanks template from a VLN dataset.
"""
from typing import Literal, Tuple, List, Iterator, Dict, Union
from dataclasses import dataclass, field
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
import tap
from difflib import SequenceMatcher

DatasetMode = Literal["soon", "reverie"]

# Allennlp is very verbose
logging.getLogger('allennlp.data.vocabulary.plugins').setLevel(logging.WARNING)
logging.getLogger('allennlp.data.fields.sequence_label_field').disabled = True
logging.getLogger('allennlp.common.plugins').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True 
logging.getLogger('cached_path').disabled = True 
logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.WARNING) 

class Arguments(tap.Tap):
    mode: DatasetMode
    data: Path
    output: Path
    num_workers: int = 20


PoSTag = Literal['conn', 'attr', 'object', 'rel_loc', 'room', 'rel_room', 'NP']

@dataclass
class PartOfSpeech:
    sentence: str
    tag: PoSTag
    # span (included):
    start: int
    end: int 
    
    @property
    def text(self):
        return self.sentence[self.start: self.end + 1]

    def __repr__(self):
        return f"<PoS ({self.start}, {self.end}, {self.tag}): {self.text}>"
        bef = self.sentence[:self.start]
        aft = self.sentence[self.end + 1:]
        return f"<PoS ({self.start}, {self.end}, {self.tag}): {bef}[{self.text}]{aft}>"
    
def random_order_cartesian_product(*factors):
    """ https://stackoverflow.com/a/53895551/4986615 """
    amount = functools.reduce(lambda prod, factor: prod * len(list(factor)), factors, 1)
    index_linked_list = [None, None]
    for max_index in reversed(range(amount)):
        index = random.randint(0, max_index)
        index_link = index_linked_list
        while index_link[1] is not None and index_link[1][0] <= index:
            index += 1
            index_link = index_link[1]
        index_link[1] = [index, index_link[1]]
        items = []
        for factor in factors:
            items.append(factor[index % len(factor)])
            index //= len(factor)
        yield items


def filter_node_with(tags: Union[str, List[str]], root_pred: Dict, child_ids: List[int]=[], verbose: bool = False):
    if isinstance(tags, str):
        tags = [tags]
        
    pos = []
    
    node = root_pred
    for child_id in child_ids:
        node = node['children'][child_id]
    
    if verbose:
        print("---" * len(child_ids), node['word'], node['nodeType'])
    
    if node['nodeType'] in tags:
        return [node]
    
    if 'children' not in node:
        return []
    
    for i in range(len(node['children'])):
        pos += filter_node_with(tags, root_pred, child_ids + [i], verbose)
    
    return pos


def fill_with(tag: PoSTag, sentence: str, part_of_speeches: List[PartOfSpeech], verbose: bool = False
) -> List[PartOfSpeech]:
    
    filled: List[PartOfSpeech] = []
    cursor = 0
    for pos in part_of_speeches:
        if pos.start > cursor:
            bef_pos = PartOfSpeech(sentence, tag, cursor, pos.start - 1)
            if verbose:
                print("FILL.BEF", bef_pos)
            filled.append(bef_pos)
        cursor = pos.end + 1
        filled.append(pos)
        
    if cursor < len(sentence):
        aft_pos = PartOfSpeech(sentence, tag, cursor, len(sentence) - cursor - 1)
        if verbose: 
            print("FILL.AFT", aft_pos)
        filled.append(aft_pos)
    return filled


def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):
    """Return best matching substring of corpus.

    Parameters
    ----------
    query : str
    corpus : str
    step : int
        Step size of first match-value scan through corpus. Can be thought of
        as a sort of "scan resolution". Should not exceed length of query.
    flex : int
        Max. left/right substring position adjustment value. Should not
        exceed length of query / 2.

    Outputs
    -------
    output0 : str
        Best matching substring.
    output1 : float
        Match ratio of best matching substring. 1 is perfect match.
    """

    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m : m-1+qlen]))
            if verbose:
                print(query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)

    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted 
        # to optimize bmv_*
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        bmv_l = match_values[p_l // step]
        bmv_r = match_values[p_l // step]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print("\n" + str(f))
                print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))

        return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    if flex >= qlen/2:
        # print("Warning: flex exceeds length of query / 2. Setting to default.")
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step

    pos_left, pos_right, match_value = adjust_left_right_positions()

    return corpus[pos_left: pos_right].strip(), match_value

class NotFindError(Exception):
    """
    Can't find query in sentence
    """

def text_to_pos(text, sentence, tag):
    rephrase, score = get_best_match(text.strip(" ,."), sentence)
    if score < 0.84:
        raise NotFindError(f"Too low score ({score}): '{text}' --> '{rephrase}'. Sentence: {sentence} ")
    text = rephrase
    start = sentence.index(text)
    end = start + len(text) - 1
    return PartOfSpeech(sentence, tag, start, end)

def clean_up(text):
    return text.replace(" ,", ",").replace("  ", " ").replace(" .", ".")

def process_attrs(instr, seg, with_obj=False):
    attrs = instr[0]
    pred = seg.predict(sentence=attrs)
    
    # remove 'this is'
    nodes = filter_node_with(['PP', 'NP'], pred['hierplane_tree']['root'])
    if len(nodes) < 2:
        return [text_to_pos(attrs.strip(",. "), instr[-2], "attr")]
    node = nodes[-1]
    node_stc = clean_up(node['word'])
    
    # extract obj
    obj = filter_node_with(['NN'], node)[-1]
    obj_text = clean_up(obj['word'])

    # attrs are defined as what is not a DT, or an object.
    dt = filter_node_with(['DT'], node)[0]
    dt_text = clean_up(dt['word'])
    all_pos = [text_to_pos(dt_text, node_stc, "conn"), text_to_pos(obj_text, node_stc, "object")]
    attr_text = None
    for pos in fill_with("attr", node_stc, all_pos):
        if pos.tag == "attr":
            attr_text = clean_up(pos.text).strip(",. ")
            break
    assert attr_text is not None, (node, all_pos)
    attr = text_to_pos(attr_text, instr[-2], "attr")
    
    if with_obj:
        return [text_to_pos(obj_text, instr[-2], "object"), attr]

    return [attr]


def process_rel_location(instr, seg, verbose: bool = False):
    rel_loc = instr[1]
    pred = seg.predict(sentence=rel_loc)
    node = pred['hierplane_tree']['root']
    
    # remove 'this is'
    nodes = filter_node_with(['PP', 'NP'], node)
    if len(nodes) < 2:
        return [text_to_pos(rel_loc, instr[-2], "rel_loc")]
    
    node = nodes[1]
    rel_loc_text = clean_up(node['word']).strip(",. ")
    # print(rel_loc_text)
    
    # remove eventual DT
    dts = filter_node_with(['DT'], node)
    if dts != []:
        dt = dts[0]['word'].strip(",. ")
        start = rel_loc_text.index(f"{dt} ") + len(dt)
        if start < 2:
            rel_loc_text = rel_loc_text[start:]
        # print(rel_loc_text)
            
    # strip
    rel_loc_text = rel_loc_text.strip(",. ")
    # print(rel_loc_text)
    
    return [text_to_pos(rel_loc_text, instr[-2], "rel_loc")]


def process_object(instr: List[str], seg: Predictor):
    pos_attrs = process_attrs(instr, seg, with_obj=True)
    for pos in pos_attrs:
        if pos.tag == "object":
            return [pos]
        
    long_instr = instr[-2]
    after_attr = long_instr.index(pos_attrs[0].text) + len(pos_attrs[0].text) + 1
    obj = long_instr[after_attr:].split(" ")[0]
    return [text_to_pos(obj, long_instr, 'object')]
    

def process_room(instr: List[str], seg: Predictor):
    room = instr[2]
    pred = seg.predict(sentence=room)
    node = pred['hierplane_tree']['root']
    
    # remove 'this is'
    nodes = filter_node_with(['PP', 'NP'], node)
    if len(nodes) < 2:
        return [text_to_pos(room, instr[-2], "room")]
    
    node = nodes[1]
    room_text = clean_up(node['word'])
    # print(room_text)
    
    # remove eventual DT
    dts = filter_node_with(['DT'], node)
    if dts != []:
        dt = dts[0]['word'].strip(",. ")
        start = room_text.index(f"{dt} ") + len(dt)
        if start < 2:
            room_text = room_text[start:]
        # print(room_text)
            
    # strip
    room_text = room_text.strip(",. ")
    # print(room_text)
    
    return [text_to_pos(room_text, instr[-2], "room")]
     
def process_rel_room(instr: List[str], seg: Predictor):
    rel_room = instr[3]
    return [text_to_pos(rel_room, instr[-2], "rel_room")]


class SoonDataset(Dataset):
    """
    Load the SOON dataset instructions and create templates.
    Templates are made only with the object description.
    """
    def __init__(self, dataset: Path):
        with open(dataset) as fid:
            data = json.load(fid)
        self.instructions = []
        for item in data:
            self.instructions += item["instructions"]
        self.verbose = False
        
        self.segmenter: Predictor =  Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
        )

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index: int):
        instr = self.instructions[index]
        instr = [clean_up(ins).lower() for ins in instr]
        long_instr = instr[-2]
        try:
            attr = process_attrs(instr, self.segmenter)
            obj = process_object(instr, self.segmenter)
            rel_loc = process_rel_location(instr, self.segmenter)
            room = process_room(instr, self.segmenter)
            rel_room = process_rel_room(instr, self.segmenter)
        except (NotFindError, IndexError, ValueError) as e:
            print("ERROR", e)
            return None
        
        part_of_speeches = attr + obj + rel_loc + room + rel_room
        part_of_speeches = sorted(part_of_speeches, key=lambda item: item.start)
        cursor = 0
        for pos in part_of_speeches:
            pos.start = max(pos.start, cursor)
            cursor = pos.end + 1
        part_of_speeches =  fill_with('conn', long_instr, part_of_speeches, verbose=False)
        # print("-"*80)
        # print(long_instr)
        # print(part_of_speeches)

        return part_of_speeches

    
def collate_without_none(samples):
    return [s for s in samples if s is not None]

class ReverieDataset(SoonDataset):
    """
    Load the REVERIE dataset instructions
    """

def export_templates(templates: List[List[PartOfSpeech]], output: Path):
    tpls = []
    for part_of_speeches in templates:
        tpl = "".join([pos.text if pos.tag == "conn" else f"[{pos.tag}]" for pos in part_of_speeches])
        tpls.append(tpl)
    
    args.output.parent.mkdir(exist_ok=True)
    with open(output, "w") as fid:
        json.dump(tpls, fid, indent=2)
    
    
if __name__ == "__main__":

    args = Arguments().parse_args()
    print(args)

    if args.mode == "soon":
        ClassName = SoonDataset
    elif args.mode == "reverie":
        ClassName = ReverieDataset
    else:
        raise ValueError(f"Unexpected mode {args.mode}")

    dataset = ClassName(args.data)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=1, collate_fn=collate_without_none)
    templates = []

    for item in tqdm(dataloader):
        templates += item

    print("Dataset length:", len(dataset))
    print("Templates length:", len(templates))
        
    export_templates(templates, args.output)
