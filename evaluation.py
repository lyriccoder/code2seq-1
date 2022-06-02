import pathlib
from typing import List, cast

import torch
from code2seq.data.vocabulary import Vocabulary
from code2seq.model.code2seq import Code2Seq
import argparse
from omegaconf import DictConfig, OmegaConf
from code2seq.data.path_context import LabeledPathContext, Path
from typing import Optional, List, cast
from random import shuffle
from typing import Dict, List, Optional

class PathContextConvert:
    _separator = "|"

    def __init__(self, vocab, config: DictConfig, random_context: bool):
        self._config = config
        self._vocab = vocab
        self._random_context = random_context

    def getPathContext(self, sourceCode: str):
        raw_sample = sourceCode
        raw_label, *raw_path_contexts = raw_sample.split()
        n_contexts = min(len(raw_path_contexts), self._config.max_context)
        if self._random_context:
            shuffle(raw_path_contexts)
            
        raw_path_contexts = raw_path_contexts[:n_contexts]
        if self._config.max_label_parts == 1:
            label = self.tokenize_class(raw_label, self._vocab.label_to_id)
        else:
            label = self.tokenize_label(raw_label, self._vocab.label_to_id, self._config.max_label_parts)
        paths = [self._get_path(raw_path.split(",")) for raw_path in raw_path_contexts]
        return LabeledPathContext(label, paths)

    @staticmethod
    def tokenize_class(raw_class: str, vocab: Dict[str, int]) -> List[int]:
        return [vocab[raw_class]]

    @staticmethod
    def tokenize_label(raw_label: str, vocab: Dict[str, int], max_parts: Optional[int]) -> List[int]:
        sublabels = raw_label.split(PathContextConvert._separator)
        max_parts = max_parts or len(sublabels)
        label_unk = vocab[Vocabulary.UNK]

        label = [vocab[Vocabulary.SOS]] + [vocab.get(st, label_unk) for st in sublabels[:max_parts]]
        if len(sublabels) < max_parts:
            label.append(vocab[Vocabulary.EOS])
            label += [vocab[Vocabulary.PAD]] * (max_parts + 1 - len(label))
        return label

    @staticmethod
    def tokenize_token(token: str, vocab: Dict[str, int], max_parts: Optional[int]) -> List[int]:
        sub_tokens = token.split(PathContextConvert._separator)
        max_parts = max_parts or len(sub_tokens)
        token_unk = vocab[Vocabulary.UNK]

        result = [vocab.get(st, token_unk) for st in sub_tokens[:max_parts]]
        result += [vocab[Vocabulary.PAD]] * (max_parts - len(result))
        return result

    def _get_path(self, raw_path: List[str]) -> Path:
        return Path(
            from_token=self.tokenize_token(raw_path[0], self._vocab.token_to_id, self._config.max_token_parts),
            path_node=self.tokenize_token(raw_path[1], self._vocab.node_to_id, self._config.path_length),
            to_token=self.tokenize_token(raw_path[2], self._vocab.token_to_id, self._config.max_token_parts),
        )


def _transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
    return [cast(List[int], it) for it in zip(*list_of_lists)]


if __name__ == '__main__':
    testCode = """
    int f(n) {
        if (n == 2) {
            return 0;
        }
        else {
            return 1;
        }
    }
    """

    parser = argparse.ArgumentParser(
        description='Run performance testing for program-slicing repo with certain commit id'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        '--code',
        type=str,
        required=False,
        default=testCode
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    # checkpoint_path = config.checkpoint
    c2s = Code2Seq.load_from_checkpoint(args.checkpoint_path)
    c2s.eval()
    id_to_label = {idx: lab for (lab, idx) in c2s._vocabulary.label_to_id.items()}

    converter = PathContextConvert(c2s._vocabulary, config.data, False)

    s = converter.getPathContext('get|post|process|logout|url string,Cls0|Mth|Nm1,METHOD_NAME string,Cls0|Mth|Prm|VDID0,request string,Cls0|Mth|Prm|Cls1,http|servlet|request METHOD_NAME,Nm1|Mth|Prm|VDID0,request METHOD_NAME,Nm1|Mth|Prm|Cls1,http|servlet|request METHOD_NAME,Nm1|Mth|Bk|Ret|Cal0|Nm0,auth|utils METHOD_NAME,Nm1|Mth|Bk|Ret|Cal0|Nm2,request METHOD_NAME,Nm1|Mth|Bk|Ret|Cal0|Fld3|Nm0,am|post|auth|process|interface METHOD_NAME,Nm1|Mth|Bk|Ret|Cal0|Fld3|Nm2,post|process|logout|url METHOD_NAME,Nm1|Mth|Bk|Ret|Cal0|Nm3,get|post|process|url request,VDID0|Prm|Cls1,http|servlet|request request,VDID0|Prm|Mth|Bk|Ret|Cal0|Nm0,auth|utils request,VDID0|Prm|Mth|Bk|Ret|Cal0|Nm2,request request,VDID0|Prm|Mth|Bk|Ret|Cal0|Fld3|Nm0,am|post|auth|process|interface request,VDID0|Prm|Mth|Bk|Ret|Cal0|Fld3|Nm2,post|process|logout|url request,VDID0|Prm|Mth|Bk|Ret|Cal0|Nm3,get|post|process|url http|servlet|request,Cls1|Prm|Mth|Bk|Ret|Cal0|Nm0,auth|utils http|servlet|request,Cls1|Prm|Mth|Bk|Ret|Cal0|Nm2,request http|servlet|request,Cls1|Prm|Mth|Bk|Ret|Cal0|Fld3|Nm0,am|post|auth|process|interface http|servlet|request,Cls1|Prm|Mth|Bk|Ret|Cal0|Fld3|Nm2,post|process|logout|url http|servlet|request,Cls1|Prm|Mth|Bk|Ret|Cal0|Nm3,get|post|process|url auth|utils,Nm0|Cal|Nm2,request request,Nm2|Cal|Fld3|Nm0,am|post|auth|process|interface request,Nm2|Cal|Fld3|Nm2,post|process|logout|url request,Nm2|Cal|Nm3,get|post|process|url am|post|auth|process|interface,Nm0|Fld3|Nm2,post|process|logout|url am|post|auth|process|interface,Nm0|Fld3|Cal|Nm3,get|post|process|url post|process|logout|url,Nm2|Fld3|Cal|Nm3,get|post|process|url')
    from_token = torch.tensor(
        _transpose([path.from_token for path in s.path_contexts]),
        dtype=torch.long)
    path_nodes = torch.tensor(
        _transpose([path.path_node for path in s.path_contexts]),
        dtype=torch.long)
    to_token = torch.tensor(
        _transpose([path.to_token for path in s.path_contexts]),
        dtype=torch.long)
    contexts = torch.tensor([len(s.path_contexts)])
    output = c2s.test(
        from_token=from_token,
        path_nodes=path_nodes,
        to_token=to_token,
        contexts_per_label=contexts,
        output_length=10,
        beam_width=10)
    for seq, val in output.items():
        labels = [
            id_to_label[int(i)] for i in seq
            if id_to_label[int(i)] not in ('<EOS>', '<SOS>')
        ]
        labels_non_f = [
            id_to_label[int(i)] for i in seq  
            if id_to_label[int(i)]
        ]
        print(list(labels_non_f), val[0])
        print(list(labels))
