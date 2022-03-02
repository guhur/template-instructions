"""
This file extracts a fill-in-the-blanks template from a VLN dataset.
"""
from typing import Literal, Tuple, List, Iterator, Dict
from dataclasses import dataclass, field
import functools
import json
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

DatasetMode = Literal["soon", "reverie"]


class Arguments(tap.Tap):
    mode: DatasetMode
    data: Path
    output: Path
    num_workers: int = 20


@dataclass
class Sample:
    instr: str
    path: List[str]
    scan: str
    heading: float


@dataclass
class TokenPerturbation:
    text: str
    span: Tuple[int, int]
    mode: str = "NONE"
    cand: List[str] = field(default_factory=list)


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



# forbiden_words:
# voc: 
# num_perturbations:
@dataclass
class Segmenter:
    num_perturbations: int = 1
    tokenizer: AllenTokenizer = AllenTokenizer()
    predictor: Predictor = field(
        default_factory=lambda: Predictor.from_path(
            # "/gpfsdswork/projects/rech/vuo/uok79zh/.allennlp/elmo/"
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
        )
    )
    # turn is causing a lot of confusion to the parser
    forbidden_words: Tuple = ("turn",)
    min_len: int = 2
    max_len: int = 6
    mode: str = "NP"

    def __call__(self, sample: Sample) -> Iterator[str]:
        sentence = sample.instr
        # We need to add space between each word to avoid a mismatch
        tokens = self.tokenizer.tokenize(sentence.lower().rstrip())
        fake_sentence = " ".join([str(token) for token in tokens])
        segments = self.segment(fake_sentence)

        corruptable = [i for i, tok in enumerate(segments) if tok.mode != "NONE"]
        random.shuffle(corruptable)
        candidates = combinations(corruptable, self.num_perturbations)
        cand_tokens = [list(range(len(segment.cand))) for segment in segments]

        iterators = {}
        for candidate in candidates:
            tokens = [cand_tokens[i] for i in candidate]
            iterators[candidate] = random_order_cartesian_product(*tokens)

        while True:
            if not iterators:
                return
            candidate, it = random.choice(list(iterators.items()))
            try:
                indexes = next(it)
            except StopIteration:
                del iterators[candidate]
                continue

            words = []
            j = 0
            for i, segment in enumerate(segments):
                if i in candidate:
                    words.append(segment.cand[indexes[j]])
                    j += 1
                else:
                    words.append(segment.text)
            yield "".join(words)

    def _retrieve_noun_phrases(
        self, sentence: str, tree: Dict, pos: int = 0
    ) -> List[TokenPerturbation]:
        """
        Return a dictionary with noun phrases and the spanning positions
        max_len is a protection against parser failures
        """
        noun_phrases: List[TokenPerturbation] = []
        next_char = len(tree["word"]) + pos
        if next_char < len(sentence) and sentence[next_char].isspace():
            tree["word"] += " "

        # print(tree["word"], pos)
        # offset the position as we decode this tree
        inner_pos = 0

        for children in tree["children"]:
            next_char = len(children["word"]) + pos + inner_pos
            if next_char < len(sentence) and sentence[next_char].isspace():
                children["word"] += " "
            # print(
            #     "---",
            #     children["word"],
            #     f"{pos+inner_pos} ({inner_pos}+{pos}) => {pos+inner_pos+len(children['word']) - 1}",
            #     sentence[pos + inner_pos : pos + inner_pos + len(children["word"])],
            # )

            if children["nodeType"] == "NP":
                proposal = children["word"]
                num_tokens = len(self.tokenizer.tokenize(proposal))

                if (
                    "." not in proposal
                    and self.min_len <= num_tokens
                    and num_tokens <= self.max_len
                    and all(word not in proposal for word in self.forbidden_words)
                ):
                    start = tree["word"][inner_pos:].find(proposal) + pos + inner_pos
                    end = start + len(proposal) - 1
                    noun_phrases.append(
                        TokenPerturbation(proposal, (start, end), self.mode)
                    )
                    inner_pos += len(children["word"])
                    continue

            if "children" in children:
                start = (
                    tree["word"][inner_pos:].find(children["word"]) + pos + inner_pos
                )
                noun_phrases += self._retrieve_noun_phrases(
                    sentence, children, pos=start
                )

            inner_pos += len(children["word"])
        return noun_phrases

    def segment(self, sentence: str) -> List[TokenPerturbation]:
        preds = self.predictor.predict(sentence=sentence)  # type: ignore
        noun_phrases = self._retrieve_noun_phrases(
            sentence, preds["hierplane_tree"]["root"]
        )

        # sort the noun phrases by start span
        noun_phrases = sorted(noun_phrases, key=lambda item: item.span[0])

        return fill_with_none(sentence, noun_phrases)


def fill_with_none(
    sentence: str, tokens: List[TokenPerturbation]
) -> List[TokenPerturbation]:
    filled: List[TokenPerturbation] = []
    cursor = 0
    for token in tokens:
        start, end = token.span
        if start > cursor:
            filled.append(
                TokenPerturbation(sentence[cursor:start], (cursor, start - 1), "NONE")
            )
        cursor = end + 1
        filled.append(token)
    if cursor < len(sentence):
        filled.append(
            TokenPerturbation(sentence[cursor:], (cursor, len(sentence) - 1), "NONE")
        )
    return filled


class SoonDataset(Dataset):
    """
    Load the SOON dataset instructions
    """
    def __init__(self, dataset: Path):
        with open(dataset) as fid:
            data = json.load(fid)
        instructions = []
        for item in data:
            instructions += item["instructions"]

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index: int):
        return self.instructions[index]

class ReverieDataset(SoonDataset):
    """
    Load the REVERIE dataset instructions
    """


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
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=1, collate_fn=lambda x: x)
    templates = []

    for item in tqdm(dataloader):
        templates.append(item)
        
