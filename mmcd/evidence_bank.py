"""Sparse, CPU-resident evidence bank for VERA-CD.

The runtime bank stores only usable item--KC evidence edges.  VERA-CD keeps
source-specific Solution/Distractor reliabilities because they are not
interchangeable, and derives one final edge reliability as their maximum.
Legacy v1 and compact v2 banks remain readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch


class EvidenceBank:
    """Load and batch-gather a CSR-style item--KC evidence bank."""

    REQUIRED_FILES = (
        "item_ptr.npy",
        "edge_kc_idx.npy",
        "solution_embedding.npy",
        "distractor_embedding.npy",
    )

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Evidence bank directory not found: {self.root}")
        missing = [name for name in self.REQUIRED_FILES if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Evidence bank is incomplete ({self.root}); missing: {missing}"
            )

        self.item_ptr = np.load(self.root / "item_ptr.npy", mmap_mode="r")
        self.edge_kc_idx = np.load(self.root / "edge_kc_idx.npy", mmap_mode="r")
        self.solution_embedding = np.load(
            self.root / "solution_embedding.npy", mmap_mode="r"
        )
        self.distractor_embedding = np.load(
            self.root / "distractor_embedding.npy", mmap_mode="r"
        )

        compact_reliability_path = self.root / "edge_reliability.npy"
        self.is_compact = compact_reliability_path.is_file()
        self.has_source_reliability = False
        if self.is_compact:
            self.edge_reliability = np.load(compact_reliability_path, mmap_mode="r")
            source_paths = (
                self.root / "solution_reliability.npy",
                self.root / "distractor_reliability.npy",
            )
            if all(path.is_file() for path in source_paths):
                self.solution_reliability = np.load(source_paths[0], mmap_mode="r")
                self.distractor_reliability = np.load(source_paths[1], mmap_mode="r")
                self.has_source_reliability = True
            else:
                # Compact v2 compatibility. Presence is recoverable, but the
                # original per-source confidence is not; new runs should use v3.
                self.solution_reliability = None
                self.distractor_reliability = None
            self.solution_mask = None
            self.distractor_mask = None
        else:
            legacy_files = (
                "solution_reliability.npy",
                "distractor_reliability.npy",
                "solution_mask.npy",
                "distractor_mask.npy",
            )
            legacy_missing = [
                name for name in legacy_files if not (self.root / name).is_file()
            ]
            if legacy_missing:
                raise FileNotFoundError(
                    "Evidence bank has neither compact reliability nor complete "
                    f"legacy fields; missing: {legacy_missing}"
                )
            self.solution_reliability = np.load(
                self.root / "solution_reliability.npy", mmap_mode="r"
            )
            self.distractor_reliability = np.load(
                self.root / "distractor_reliability.npy", mmap_mode="r"
            )
            self.solution_mask = np.load(self.root / "solution_mask.npy", mmap_mode="r")
            self.distractor_mask = np.load(
                self.root / "distractor_mask.npy", mmap_mode="r"
            )
            self.edge_reliability = None
            self.has_source_reliability = True

        edge_item_path = self.root / "edge_item_idx.npy"
        self.edge_item_idx = (
            np.load(edge_item_path, mmap_mode="r") if edge_item_path.is_file() else None
        )
        self.metadata: Dict[str, object] = {}
        metadata_path = self.root / "metadata.json"
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        self._validate()

    def _validate(self) -> None:
        if self.item_ptr.ndim != 1 or self.item_ptr.size < 2:
            raise ValueError("item_ptr must be a one-dimensional CSR pointer array")
        if int(self.item_ptr[0]) != 0:
            raise ValueError("item_ptr must start at zero")
        if np.any(np.diff(self.item_ptr) < 0):
            raise ValueError("item_ptr must be monotonically non-decreasing")
        edge_count = int(self.item_ptr[-1])
        if self.edge_kc_idx.ndim != 1 or self.edge_kc_idx.shape[0] != edge_count:
            raise ValueError("edge_kc_idx is inconsistent with item_ptr")
        if np.any(self.edge_kc_idx < 0):
            raise ValueError("edge_kc_idx must contain non-negative KC indices")

        arrays = {}
        if self.is_compact:
            arrays["edge_reliability"] = self.edge_reliability
            if self.has_source_reliability:
                arrays["solution_reliability"] = self.solution_reliability
                arrays["distractor_reliability"] = self.distractor_reliability
        else:
            arrays.update(
                {
                    "solution_reliability": self.solution_reliability,
                    "distractor_reliability": self.distractor_reliability,
                    "solution_mask": self.solution_mask,
                    "distractor_mask": self.distractor_mask,
                }
            )
        if self.edge_item_idx is not None:
            arrays["edge_item_idx"] = self.edge_item_idx
        for name, array in arrays.items():
            if array.shape[0] != edge_count:
                raise ValueError(
                    f"{name} has {array.shape[0]} rows but item_ptr declares {edge_count}"
                )

        for name, array in {
            "solution_embedding": self.solution_embedding,
            "distractor_embedding": self.distractor_embedding,
        }.items():
            if array.ndim != 2 or array.shape[0] != edge_count:
                raise ValueError(
                    f"{name} must have shape [num_edges, embedding_dim], got {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
        if self.solution_embedding.shape[1] != self.distractor_embedding.shape[1]:
            raise ValueError("Solution and Distractor embedding dimensions differ")

        if self.edge_item_idx is not None and edge_count and not np.array_equal(
            self.edge_item_idx,
            np.repeat(np.arange(self.num_items), np.diff(self.item_ptr)),
        ):
            raise ValueError("edge_item_idx is inconsistent with item_ptr")
        for item_id in range(self.num_items):
            start, end = int(self.item_ptr[item_id]), int(self.item_ptr[item_id + 1])
            concepts = self.edge_kc_idx[start:end]
            if concepts.size != np.unique(concepts).size:
                raise ValueError(f"item {item_id} contains duplicate KC evidence edges")

        reliability_arrays = []
        if self.is_compact:
            reliability_arrays.append(("edge", self.edge_reliability))
            if self.has_source_reliability:
                reliability_arrays.extend(
                    (
                        ("solution", self.solution_reliability),
                        ("distractor", self.distractor_reliability),
                    )
                )
        else:
            reliability_arrays.extend(
                (
                    ("solution", self.solution_reliability),
                    ("distractor", self.distractor_reliability),
                )
            )
        for name, reliability in reliability_arrays:
            if not np.all(np.isfinite(reliability)):
                raise ValueError(f"{name} reliability contains non-finite values")
            lower = reliability <= 0 if name == "edge" else reliability < 0
            if np.any(lower) or np.any(reliability > 1):
                interval = "(0, 1]" if name == "edge" else "[0, 1]"
                raise ValueError(f"{name} reliability must lie in {interval}")
        if self.is_compact and self.has_source_reliability:
            expected = np.maximum(self.solution_reliability, self.distractor_reliability)
            if not np.allclose(expected, self.edge_reliability, atol=1e-7, rtol=0):
                raise ValueError("edge_reliability must equal max of source reliabilities")
        if not self.is_compact:
            for name, mask, reliability in (
                ("solution", self.solution_mask, self.solution_reliability),
                ("distractor", self.distractor_mask, self.distractor_reliability),
            ):
                if np.any((mask == 0) & (reliability != 0)):
                    raise ValueError(
                        f"{name} reliability must be zero whenever its mask is zero"
                    )

    @property
    def num_items(self) -> int:
        return int(self.item_ptr.size - 1)

    @property
    def num_edges(self) -> int:
        return int(self.item_ptr[-1])

    @property
    def num_concepts(self) -> int:
        observed = int(self.edge_kc_idx.max()) + 1 if self.num_edges else 0
        return max(observed, int(self.metadata.get("num_concepts", 0)))

    @property
    def embedding_dim(self) -> int:
        return int(self.solution_embedding.shape[1])

    def gather(
        self,
        question_idx: torch.Tensor,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if question_idx.ndim != 1:
            raise ValueError(
                f"question_idx must be one-dimensional, got {question_idx.shape}"
            )
        destination = torch.device(device) if device is not None else question_idx.device
        q_cpu = question_idx.detach().to(device="cpu", dtype=torch.long).numpy()
        if q_cpu.size and (q_cpu.min() < 0 or q_cpu.max() >= self.num_items):
            raise IndexError(
                f"Question ID outside evidence bank range [0, {self.num_items})"
            )

        starts = self.item_ptr[q_cpu]
        ends = self.item_ptr[q_cpu + 1]
        lengths = ends - starts
        max_edges = int(lengths.max()) if lengths.size else 0
        batch_size = int(q_cpu.size)
        kc_idx = torch.full((batch_size, max_edges), -1, dtype=torch.long)
        solution_embedding = torch.zeros(
            (batch_size, max_edges, self.embedding_dim), dtype=torch.float32
        )
        distractor_embedding = torch.zeros_like(solution_embedding)
        solution_reliability = torch.zeros((batch_size, max_edges), dtype=torch.float32)
        distractor_reliability = torch.zeros_like(solution_reliability)
        edge_reliability = torch.zeros_like(solution_reliability)
        valid = torch.zeros((batch_size, max_edges), dtype=torch.bool)

        for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
            length = int(end - start)
            if length == 0:
                continue
            sol = np.array(self.solution_embedding[start:end], dtype=np.float32, copy=True)
            dis = np.array(self.distractor_embedding[start:end], dtype=np.float32, copy=True)
            kc_idx[row, :length] = torch.from_numpy(
                np.array(self.edge_kc_idx[start:end], dtype=np.int64, copy=True)
            )
            solution_embedding[row, :length] = torch.from_numpy(sol)
            distractor_embedding[row, :length] = torch.from_numpy(dis)
            if self.is_compact:
                edge = np.array(self.edge_reliability[start:end], dtype=np.float32, copy=True)
                if self.has_source_reliability:
                    sol_rel = np.array(
                        self.solution_reliability[start:end], dtype=np.float32, copy=True
                    )
                    dis_rel = np.array(
                        self.distractor_reliability[start:end], dtype=np.float32, copy=True
                    )
                else:
                    sol_rel = edge * (np.linalg.norm(sol, axis=1) > 0)
                    dis_rel = edge * (np.linalg.norm(dis, axis=1) > 0)
            else:
                sol_rel = np.array(
                    self.solution_reliability[start:end], dtype=np.float32, copy=True
                )
                dis_rel = np.array(
                    self.distractor_reliability[start:end], dtype=np.float32, copy=True
                )
                edge = np.maximum(sol_rel, dis_rel)
            solution_reliability[row, :length] = torch.from_numpy(sol_rel)
            distractor_reliability[row, :length] = torch.from_numpy(dis_rel)
            edge_reliability[row, :length] = torch.from_numpy(edge)
            valid[row, :length] = True

        result = {
            "kc_idx": kc_idx,
            "solution_embedding": solution_embedding,
            "distractor_embedding": distractor_embedding,
            "solution_reliability": solution_reliability,
            "distractor_reliability": distractor_reliability,
            "edge_reliability": edge_reliability,
            "valid": valid,
        }
        return {name: value.to(destination) for name, value in result.items()}


