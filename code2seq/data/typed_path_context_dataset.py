from typing import List

from omegaconf import DictConfig

from code2seq.data import PathContextDataset, TypedPath
from code2seq.data.vocabulary import TypedVocabulary


class TypedPathContextDataset(PathContextDataset):
    def __init__(self, data_file: str, config: DictConfig, vocabulary: TypedVocabulary, random_context: bool):
        super().__init__(data_file, config, vocabulary, random_context)
        self._vocab = vocabulary

    def _get_path(self, raw_path: List[str]) -> TypedPath:
        return TypedPath(
            from_type=self._tokenize_token(raw_path[0], self._vocab.type_to_id, self._config.max_type_parts),
            from_token=self._tokenize_token(raw_path[1], self._vocab.token_to_id, self._config.max_token_parts),
            path_node=self._tokenize_token(raw_path[2], self._vocab.node_to_id, None),
            to_token=self._tokenize_token(raw_path[3], self._vocab.token_to_id, self._config.max_token_parts),
            to_type=self._tokenize_token(raw_path[4], self._vocab.type_to_id, self._config.max_type_parts),
        )
