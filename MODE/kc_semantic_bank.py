"""Validated global-KC and item--KC semantic bank used by MOCD."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Union
import numpy as np
import torch
from .evidence_bank import EvidenceBank

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

class KCSemanticBank:
    """Load one frozen KC-name matrix and sparse KC-conditioned edge evidence."""
    SCHEMA_VERSION = 'egcd_kc_semantic_bank_v1'
    REQUIRED_METADATA = ('schema_version', 'dataset', 'num_items', 'num_concepts', 'num_edges', 'embedding_dim', 'encoder_name', 'encoder_revision', 'max_length', 'candidates_sha256', 'evidence_records_sha256', 'q_matrix_sha256', 'id_mapping_sha256', 'kc_names_sha256', 'text_template_version', 'array_sha256', 'bank_fingerprint')

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser().resolve()
        metadata_path = self.root / 'metadata.json'
        names_path = self.root / 'kc_names.json'
        name_embedding_path = self.root / 'kc_name_embedding.npy'
        for path in (metadata_path, names_path, name_embedding_path):
            if not path.is_file():
                raise FileNotFoundError(f'KC semantic bank is incomplete: {path}')
        with metadata_path.open('r', encoding='utf-8') as handle:
            self.metadata: Dict[str, object] = json.load(handle)
        missing = [key for key in self.REQUIRED_METADATA if key not in self.metadata]
        if missing:
            raise ValueError(f'KC semantic bank metadata is missing: {missing}')
        if self.metadata['schema_version'] != self.SCHEMA_VERSION:
            raise ValueError(f"KC bank requires schema {self.SCHEMA_VERSION!r}, got {self.metadata['schema_version']!r}")
        with names_path.open('r', encoding='utf-8') as handle:
            self.kc_names = json.load(handle)
        self.kc_name_embedding = np.load(name_embedding_path, mmap_mode='r')
        self.edge_bank = EvidenceBank(self.root)
        self._validate(names_path)

    @staticmethod
    def compute_fingerprint(metadata: Mapping[str, object]) -> str:
        keys = ('schema_version', 'dataset', 'num_items', 'num_concepts', 'num_edges', 'embedding_dim', 'encoder_name', 'encoder_revision', 'max_length', 'candidates_sha256', 'evidence_records_sha256', 'q_matrix_sha256', 'id_mapping_sha256', 'kc_names_sha256', 'text_template_version', 'array_sha256')
        payload = {key: metadata.get(key) for key in keys}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def _validate(self, names_path: Path) -> None:
        if not isinstance(self.kc_names, list):
            raise ValueError('kc_names.json must contain a JSON list')
        if any((not str(name).strip() for name in self.kc_names)):
            raise ValueError('KC semantic bank contains an empty KC name')
        if self.kc_name_embedding.ndim != 2:
            raise ValueError('kc_name_embedding must have shape [K, semantic_dim]')
        if not np.all(np.isfinite(self.kc_name_embedding)):
            raise ValueError('kc_name_embedding contains non-finite values')
        num_items = int(self.metadata['num_items'])
        num_concepts = int(self.metadata['num_concepts'])
        embedding_dim = int(self.metadata['embedding_dim'])
        if self.edge_bank.num_items != num_items:
            raise ValueError('KC edge pointer count disagrees with metadata')
        if self.edge_bank.num_concepts > num_concepts:
            raise ValueError('KC edge bank contains an out-of-range KC ID')
        if len(self.kc_names) != num_concepts:
            raise ValueError('KC name count disagrees with metadata')
        if self.kc_name_embedding.shape != (num_concepts, embedding_dim):
            raise ValueError('KC-name embedding dimensions disagree with metadata')
        if self.edge_bank.embedding_dim != embedding_dim:
            raise ValueError('KC-name and item--KC embedding dimensions differ')
        if self.edge_bank.num_edges != int(self.metadata.get('num_edges', -1)):
            raise ValueError('KC edge count disagrees with metadata')
        if _sha256_file(names_path) != self.metadata.get('kc_names_sha256'):
            raise ValueError('KC names fingerprint mismatch')
        array_hashes = self.metadata['array_sha256']
        if not isinstance(array_hashes, dict):
            raise ValueError('KC semantic bank array_sha256 must be a mapping')
        required_arrays = {'kc_name_embedding.npy', 'item_ptr.npy', 'edge_kc_idx.npy', 'solution_embedding.npy', 'distractor_embedding.npy', 'solution_reliability.npy', 'distractor_reliability.npy', 'edge_reliability.npy'}
        if set(array_hashes) != required_arrays:
            raise ValueError('KC semantic bank hashes do not cover exact required arrays')
        for filename, expected in array_hashes.items():
            if _sha256_file(self.root / filename) != expected:
                raise ValueError(f'KC semantic bank array mismatch: {filename}')
        expected_fingerprint = self.compute_fingerprint(self.metadata)
        if self.metadata['bank_fingerprint'] != expected_fingerprint:
            raise ValueError('KC semantic bank metadata fingerprint mismatch')

    @property
    def fingerprint(self) -> str:
        return str(self.metadata['bank_fingerprint'])

    @property
    def num_items(self) -> int:
        return int(self.metadata['num_items'])

    @property
    def num_concepts(self) -> int:
        return int(self.metadata['num_concepts'])

    @property
    def embedding_dim(self) -> int:
        return int(self.metadata['embedding_dim'])

    @property
    def num_edges(self) -> int:
        return self.edge_bank.num_edges

    def name_tensor(self) -> torch.Tensor:
        return torch.from_numpy(np.array(self.kc_name_embedding, dtype=np.float32, copy=True))

    def gather(self, question_idx: torch.Tensor, device=None):
        return self.edge_bank.gather(question_idx, device=device)

    @classmethod
    def validate_dataset_binding(cls, root: Union[str, Path], data_dir: Union[str, Path], dataset: str) -> None:
        root = Path(root).expanduser().resolve()
        data_dir = Path(data_dir).expanduser().resolve()
        metadata_path = root / 'metadata.json'
        if not metadata_path.is_file():
            raise FileNotFoundError(f'KC semantic bank metadata not found: {metadata_path}')
        with metadata_path.open('r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        if metadata.get('schema_version') != cls.SCHEMA_VERSION:
            raise ValueError('KC semantic bank has an unsupported schema')
        if metadata.get('dataset') != dataset:
            raise ValueError(f"KC bank dataset {metadata.get('dataset')!r} does not match {dataset!r}")
        for filename, key in (('q_matrix.csv', 'q_matrix_sha256'), ('id_mapping.json', 'id_mapping_sha256')):
            path = data_dir / filename
            if not path.is_file() or _sha256_file(path) != metadata.get(key):
                raise ValueError(f'KC semantic bank and dataset {filename} are not aligned')
