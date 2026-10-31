"""Read the released sparse item--KC evidence vectors."""
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch


class EvidenceBank:
    REQUIRED_FILES = (
        'item_ptr.npy', 'edge_kc_idx.npy', 'solution_embedding.npy',
        'distractor_embedding.npy', 'solution_reliability.npy',
        'distractor_reliability.npy', 'edge_reliability.npy',
    )

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser().resolve()
        missing = [name for name in self.REQUIRED_FILES
                   if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(f'Incomplete KC evidence bank: {missing}')
        for name in self.REQUIRED_FILES:
            setattr(self, name.removesuffix('.npy'),
                    np.load(self.root / name, mmap_mode='r'))
        self._validate()

    def _validate(self):
        if self.item_ptr.ndim != 1 or self.item_ptr.size < 2:
            raise ValueError('item_ptr must be one-dimensional')
        if self.item_ptr[0] != 0 or np.any(np.diff(self.item_ptr) < 0):
            raise ValueError('Invalid item_ptr')
        edges = int(self.item_ptr[-1])
        for name in ('edge_kc_idx', 'solution_reliability',
                     'distractor_reliability', 'edge_reliability'):
            if getattr(self, name).shape != (edges,):
                raise ValueError(f'Invalid {name} shape')
        if np.any(self.edge_kc_idx < 0):
            raise ValueError('Negative KC index')
        if self.solution_embedding.ndim != 2 or self.solution_embedding.shape[0] != edges:
            raise ValueError('Invalid solution_embedding shape')
        if self.distractor_embedding.shape != self.solution_embedding.shape:
            raise ValueError('Invalid distractor_embedding shape')
        for name in ('solution_embedding', 'distractor_embedding',
                     'solution_reliability', 'distractor_reliability',
                     'edge_reliability'):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f'Non-finite {name}')
        for name in ('solution_reliability', 'distractor_reliability'):
            values = getattr(self, name)
            if np.any(values < 0) or np.any(values > 1):
                raise ValueError(f'Invalid {name}')
        if np.any(self.edge_reliability <= 0) or np.any(self.edge_reliability > 1):
            raise ValueError('Invalid edge_reliability')
        if not np.allclose(self.edge_reliability,
                           np.maximum(self.solution_reliability,
                                      self.distractor_reliability), atol=1e-7, rtol=0):
            raise ValueError('Edge and source reliabilities disagree')
        for item in range(self.num_items):
            start, end = int(self.item_ptr[item]), int(self.item_ptr[item + 1])
            indices = self.edge_kc_idx[start:end]
            if indices.size != np.unique(indices).size:
                raise ValueError(f'Duplicate KC edge for item {item}')

    @property
    def num_items(self) -> int:
        return int(self.item_ptr.size - 1)

    @property
    def num_edges(self) -> int:
        return int(self.item_ptr[-1])

    @property
    def num_concepts(self) -> int:
        return int(self.edge_kc_idx.max()) + 1 if self.num_edges else 0

    @property
    def embedding_dim(self) -> int:
        return int(self.solution_embedding.shape[1])

    def gather(self, question_idx: torch.Tensor,
               device: Optional[Union[torch.device, str]] = None) -> Dict[str, torch.Tensor]:
        if question_idx.ndim != 1:
            raise ValueError('question_idx must be one-dimensional')
        destination = torch.device(device) if device is not None else question_idx.device
        q_cpu = question_idx.detach().to(device='cpu', dtype=torch.long).numpy()
        if q_cpu.size and (q_cpu.min() < 0 or q_cpu.max() >= self.num_items):
            raise IndexError('Question ID outside evidence bank')
        starts, ends = self.item_ptr[q_cpu], self.item_ptr[q_cpu + 1]
        lengths = ends - starts
        width = int(lengths.max()) if lengths.size else 0
        batch_size = int(q_cpu.size)
        kc_idx = torch.full((batch_size, width), -1, dtype=torch.long)
        solution_embedding = torch.zeros((batch_size, width, self.embedding_dim), dtype=torch.float32)
        distractor_embedding = torch.zeros_like(solution_embedding)
        solution_reliability = torch.zeros((batch_size, width), dtype=torch.float32)
        distractor_reliability = torch.zeros_like(solution_reliability)
        edge_reliability = torch.zeros_like(solution_reliability)
        valid = torch.zeros((batch_size, width), dtype=torch.bool)
        for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
            length = int(end - start)
            if not length:
                continue
            kc_idx[row, :length] = torch.from_numpy(np.array(
                self.edge_kc_idx[start:end], dtype=np.int64, copy=True))
            for name, output in (
                ('solution_embedding', solution_embedding),
                ('distractor_embedding', distractor_embedding),
                ('solution_reliability', solution_reliability),
                ('distractor_reliability', distractor_reliability),
                ('edge_reliability', edge_reliability),
            ):
                output[row, :length] = torch.from_numpy(np.array(
                    getattr(self, name)[start:end], dtype=np.float32, copy=True))
            valid[row, :length] = True
        result = dict(kc_idx=kc_idx, solution_embedding=solution_embedding,
                      distractor_embedding=distractor_embedding,
                      solution_reliability=solution_reliability,
                      distractor_reliability=distractor_reliability,
                      edge_reliability=edge_reliability, valid=valid)
        return {name: value.to(destination) for name, value in result.items()}
