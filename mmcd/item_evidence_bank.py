"""CPU-resident item-level semantic evidence bank for EGCD."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import torch

class ItemEvidenceBank:
    """Load Solution and per-option Distractor evidence for EGCD.

    Distractor cards use a CSR layout.  ``distractor_total_count`` stores the
    true number of wrong options, which can exceed the number of available
    cards when an Agent abstains or produces an incomplete record.
    """
    SCHEMA_VERSION = 'egcd_item_bank_v3'
    REQUIRED_FILES = ('raw_embedding.npy', 'raw_available_mask.npy', 'solution_embedding.npy', 'solution_reliability.npy', 'solution_available_mask.npy', 'distractor_ptr.npy', 'distractor_embedding.npy', 'distractor_reliability.npy', 'distractor_available_mask.npy', 'distractor_total_count.npy', 'metadata.json')

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f'EGCD evidence bank not found: {self.root}')
        missing = [name for name in self.REQUIRED_FILES if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(f'EGCD evidence bank is incomplete ({self.root}); missing: {missing}')
        with (self.root / 'metadata.json').open('r', encoding='utf-8') as handle:
            self.metadata: Dict[str, object] = json.load(handle)
        if self.metadata.get('schema_version') != self.SCHEMA_VERSION:
            raise ValueError(f"EGCD requires schema {self.SCHEMA_VERSION!r}, got {self.metadata.get('schema_version')!r}")
        self.raw_embedding = np.load(self.root / 'raw_embedding.npy', mmap_mode='r')
        self.raw_available_mask = np.load(self.root / 'raw_available_mask.npy', mmap_mode='r')
        self.solution_embedding = np.load(self.root / 'solution_embedding.npy', mmap_mode='r')
        self.solution_reliability = np.load(self.root / 'solution_reliability.npy', mmap_mode='r')
        self.solution_available_mask = np.load(self.root / 'solution_available_mask.npy', mmap_mode='r')
        self.distractor_ptr = np.load(self.root / 'distractor_ptr.npy', mmap_mode='r')
        self.distractor_embedding = np.load(self.root / 'distractor_embedding.npy', mmap_mode='r')
        self.distractor_reliability = np.load(self.root / 'distractor_reliability.npy', mmap_mode='r')
        self.distractor_available_mask = np.load(self.root / 'distractor_available_mask.npy', mmap_mode='r')
        self.distractor_total_count = np.load(self.root / 'distractor_total_count.npy', mmap_mode='r')
        self._validate()

    def _validate(self) -> None:
        if self.solution_embedding.ndim != 2:
            raise ValueError('solution_embedding must have shape [num_items, dim]')
        num_items, embedding_dim = self.solution_embedding.shape
        if num_items <= 0 or embedding_dim <= 0:
            raise ValueError('EGCD evidence bank cannot be empty')
        if self.raw_embedding.shape != (num_items, embedding_dim):
            raise ValueError(f'raw_embedding must have shape [{num_items}, {embedding_dim}]')
        for name, array in (('raw_available_mask', self.raw_available_mask), ('solution_reliability', self.solution_reliability), ('solution_available_mask', self.solution_available_mask), ('distractor_total_count', self.distractor_total_count)):
            if array.ndim != 1 or array.shape[0] != num_items:
                raise ValueError(f'{name} must have shape [{num_items}]')
        if self.distractor_ptr.ndim != 1 or self.distractor_ptr.shape[0] != num_items + 1:
            raise ValueError(f'distractor_ptr must have shape [{num_items + 1}]')
        if int(self.distractor_ptr[0]) != 0 or np.any(np.diff(self.distractor_ptr) < 0):
            raise ValueError('distractor_ptr must start at zero and be monotonic')
        num_cards = int(self.distractor_ptr[-1])
        if self.distractor_embedding.ndim != 2 or self.distractor_embedding.shape != (num_cards, embedding_dim):
            raise ValueError(f'distractor_embedding must have shape [{num_cards}, {embedding_dim}]')
        for name, array in (('distractor_reliability', self.distractor_reliability), ('distractor_available_mask', self.distractor_available_mask)):
            if array.ndim != 1 or array.shape[0] != num_cards:
                raise ValueError(f'{name} must have shape [{num_cards}]')
        for name, embedding in (('raw', self.raw_embedding), ('solution', self.solution_embedding), ('distractor', self.distractor_embedding)):
            if not np.all(np.isfinite(embedding)):
                raise ValueError(f'{name} embedding contains non-finite values')
        for name, reliability in (('solution', self.solution_reliability), ('distractor', self.distractor_reliability)):
            if not np.all(np.isfinite(reliability)) or not np.all(np.isin(reliability, (0.0, 0.5, 1.0))):
                raise ValueError(f'{name} reliability must be categorical 0/0.5/1')
        if np.any((self.solution_available_mask == 0) & (self.solution_reliability != 0)):
            raise ValueError('solution reliability must be zero when its available mask is zero')
        if np.any((self.distractor_available_mask == 0) & (self.distractor_reliability != 0)):
            raise ValueError('distractor reliability must be zero when its available mask is zero')
        if np.any(self.distractor_total_count < 0):
            raise ValueError('distractor_total_count cannot be negative')
        card_counts = np.diff(self.distractor_ptr)
        if np.any(card_counts > self.distractor_total_count):
            raise ValueError('an item has more Distractor cards than wrong options')
        metadata_items = int(self.metadata.get('num_items', num_items))
        metadata_dim = int(self.metadata.get('embedding_dim', embedding_dim))
        if metadata_items != num_items or metadata_dim != embedding_dim:
            raise ValueError('metadata dimensions do not match EGCD evidence arrays')
        for key in ('id_mapping_sha256', 'q_matrix_sha256', 'bank_fingerprint', 'array_sha256'):
            if not self.metadata.get(key):
                raise ValueError(f'EGCD bank metadata is missing {key!r}')
        array_hashes = self.metadata['array_sha256']
        if not isinstance(array_hashes, dict):
            raise ValueError('EGCD array_sha256 metadata must be a mapping')
        expected_array_files = {name for name in self.REQUIRED_FILES if name.endswith('.npy')}
        if set(array_hashes) != expected_array_files:
            raise ValueError('EGCD array hashes do not cover the required arrays')
        for filename, expected_hash in array_hashes.items():
            if self._sha256_file(self.root / filename) != expected_hash:
                raise ValueError(f'EGCD bank array fingerprint mismatch: {filename}')
        expected_fingerprint = str(self.metadata['bank_fingerprint'])
        actual_fingerprint = self.compute_fingerprint(self.metadata)
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(f'EGCD bank metadata fingerprint mismatch: expected {expected_fingerprint}, got {actual_fingerprint}')

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def compute_fingerprint(metadata: Dict[str, object]) -> str:
        """Compute the stable identity of a bank from its source and arrays."""
        keys = ('schema_version', 'dataset', 'num_items', 'embedding_dim', 'num_distractor_cards', 'encoder_name', 'encoder_revision', 'max_length', 'candidates_sha256', 'evidence_records_sha256', 'q_matrix_sha256', 'id_mapping_sha256', 'array_sha256')
        payload = {key: metadata.get(key) for key in keys}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    @property
    def num_items(self) -> int:
        return int(self.solution_embedding.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.solution_embedding.shape[1])

    @property
    def num_distractor_cards(self) -> int:
        return int(self.distractor_ptr[-1])

    @property
    def fingerprint(self) -> str:
        return str(self.metadata['bank_fingerprint'])

    @classmethod
    def validate_dataset_binding(cls, root: Union[str, Path], data_dir: Union[str, Path], dataset: Optional[str]=None) -> Dict[str, object]:
        """Verify that a bank belongs to the requested dataset files."""
        root = Path(root).expanduser().resolve()
        data_dir = Path(data_dir).expanduser().resolve()
        metadata_path = root / 'metadata.json'
        if not metadata_path.is_file():
            raise FileNotFoundError(f'EGCD bank metadata not found: {metadata_path}')
        with metadata_path.open('r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        if metadata.get('schema_version') != cls.SCHEMA_VERSION:
            raise ValueError(f"EGCD requires schema {cls.SCHEMA_VERSION!r}, got {metadata.get('schema_version')!r}")
        if dataset and metadata.get('dataset') not in (None, dataset):
            raise ValueError(f"EGCD bank dataset {metadata['dataset']!r} does not match {dataset!r}")
        for filename, metadata_key in (('q_matrix.csv', 'q_matrix_sha256'), ('id_mapping.json', 'id_mapping_sha256')):
            path = data_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f'Dataset file not found: {path}')
            expected = metadata.get(metadata_key)
            if not expected or cls._sha256_file(path) != expected:
                raise ValueError(f'EGCD bank and dataset {filename} are not aligned')
        return metadata

    def gather(self, question_idx: torch.Tensor, device: Optional[Union[torch.device, str]]=None, dtype: Optional[torch.dtype]=None) -> Dict[str, torch.Tensor]:
        if question_idx.ndim != 1:
            raise ValueError(f'question_idx must be one-dimensional, got {question_idx.shape}')
        destination = torch.device(device) if device is not None else question_idx.device
        if not hasattr(self, '_device_tensor_cache'):
            self._device_tensor_cache = {}
        float_dtype = dtype if dtype is not None else torch.float32
        cache_key = (str(destination), str(float_dtype))
        if cache_key not in self._device_tensor_cache:
            raw_all = torch.from_numpy(np.asarray(self.raw_embedding, dtype=np.float32).copy())
            raw_mask_all = torch.from_numpy(np.asarray(self.raw_available_mask, dtype=np.bool_).copy())
            sol_all = torch.from_numpy(np.asarray(self.solution_embedding, dtype=np.float32).copy())
            sol_mask_all = torch.from_numpy(np.asarray(self.solution_available_mask, dtype=np.bool_).copy())
            count_all = torch.from_numpy(np.asarray(self.distractor_total_count, dtype=np.int64).copy())
            lengths_all = self.distractor_ptr[1:] - self.distractor_ptr[:-1]
            max_cards_all = int(lengths_all.max()) if lengths_all.size else 0
            dist_all = torch.zeros((self.num_items, max_cards_all, self.embedding_dim), dtype=torch.float32)
            dist_mask_all = torch.zeros((self.num_items, max_cards_all), dtype=torch.bool)
            for item_id, (start, end) in enumerate(zip(self.distractor_ptr[:-1].tolist(), self.distractor_ptr[1:].tolist())):
                length = int(end - start)
                if length:
                    dist_all[item_id, :length] = torch.from_numpy(np.asarray(self.distractor_embedding[start:end], dtype=np.float32).copy())
                    dist_mask_all[item_id, :length] = torch.from_numpy(np.asarray(self.distractor_available_mask[start:end], dtype=np.bool_).copy())
            self._device_tensor_cache[cache_key] = {'raw_embedding': raw_all.to(device=destination, dtype=float_dtype), 'raw_available_mask': raw_mask_all.to(device=destination), 'solution_embedding': sol_all.to(device=destination, dtype=float_dtype), 'solution_available_mask': sol_mask_all.to(device=destination), 'distractor_embedding': dist_all.to(device=destination, dtype=float_dtype), 'distractor_available_mask': dist_mask_all.to(device=destination), 'distractor_total_count': count_all.to(device=destination)}
        idx = question_idx.detach().to(device=destination, dtype=torch.long)
        if idx.numel() and idx.device.type == 'cpu':
            if idx.min().item() < 0 or idx.max().item() >= self.num_items:
                raise IndexError(f'Question ID outside EGCD bank range [0, {self.num_items})')
        return {name: tensor.index_select(0, idx) for name, tensor in self._device_tensor_cache[cache_key].items()}
