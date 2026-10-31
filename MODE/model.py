from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .item_evidence_bank import ItemEvidenceBank
from .kc_semantic_bank import KCSemanticBank

class PosLinear(nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class _Backbone(nn.Module):

    def __init__(self, n_concepts, n_exercises, n_students, emb_dim=20, hidden_dims=(256, 128), dropout=(0.5, 0.5)):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_exercises = n_exercises
        self.n_students = n_students
        self.emb_dim = emb_dim
        self.mf_type = 'ncf2'
        self.prednet_len1, self.prednet_len2 = hidden_dims
        self.student_emb = nn.Embedding(n_students, emb_dim)
        self.exercise_emb = nn.Embedding(n_exercises, emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(n_concepts, emb_dim))
        self.e_discrimination = nn.Embedding(n_exercises, 1)
        self.prednet_full1 = PosLinear(n_concepts, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=dropout[0])
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=dropout[1])
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        self.k_diff_full1 = nn.Linear(2 * emb_dim, emb_dim)
        self.k_diff_full2 = nn.Linear(emb_dim, 1)
        self.stat_full1 = nn.Linear(2 * emb_dim, emb_dim)
        self.stat_full2 = nn.Linear(emb_dim, 1)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

class _SemanticEncoder(_Backbone):

    def __init__(self, n_concepts: int, n_exercises: int, n_students: int, evidence_bank_path: Union[str, Path], kc_semantic_bank_path: Optional[Union[str, Path]]=None, emb_dim: int=20, hidden_dims=(256, 128), dropout=(0.5, 0.5), router_hidden_dim: int=32, gate_init: float=-2.0, raw_gate_init: float=-2.0, projection_init_std: float=0.001, semantic_scale_init: float=0.99):
        super().__init__(n_concepts=n_concepts, n_exercises=n_exercises, n_students=n_students, emb_dim=emb_dim, hidden_dims=hidden_dims, dropout=dropout)
        if router_hidden_dim <= 0:
            raise ValueError('router_hidden_dim must be positive')
        if projection_init_std <= 0:
            raise ValueError('projection_init_std must be positive')
        if not 0 < semantic_scale_init < 1:
            raise ValueError('semantic_scale_init must lie strictly between zero and one')
        self.router_hidden_dim = int(router_hidden_dim)
        self.semantic_scale_init = float(semantic_scale_init)
        self.evidence_bank = ItemEvidenceBank(evidence_bank_path)
        if self.evidence_bank.num_items != n_exercises:
            raise ValueError(f'Item bank has {self.evidence_bank.num_items} items, but model expects {n_exercises}')
        self.kc_semantic_bank = None
        self.kc_global_proj = None
        self.edge_solution_proj = None
        self.edge_distractor_proj = None
        if kc_semantic_bank_path is None:
            raise ValueError('kc_semantic_bank_path is required when KC semantics are enabled')
        self.kc_semantic_bank = KCSemanticBank(kc_semantic_bank_path)
        if self.kc_semantic_bank.num_items != n_exercises:
            raise ValueError(f'KC bank has {self.kc_semantic_bank.num_items} items, but model expects {n_exercises}')
        if self.kc_semantic_bank.num_concepts != n_concepts:
            raise ValueError(f'KC bank has {self.kc_semantic_bank.num_concepts} KCs, but model expects {n_concepts}')
        semantic_dim = self.evidence_bank.embedding_dim
        self.solution_proj = nn.Linear(semantic_dim, emb_dim)
        self.distractor_proj = nn.Linear(semantic_dim, emb_dim)
        for layer in (self.solution_proj, self.distractor_proj):
            nn.init.normal_(layer.weight, mean=0.0, std=projection_init_std)
            nn.init.zeros_(layer.bias)
        self.raw_proj = nn.Linear(semantic_dim, emb_dim)
        nn.init.normal_(self.raw_proj.weight, mean=0.0, std=projection_init_std)
        nn.init.zeros_(self.raw_proj.bias)
        self.solution_gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.distractor_gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.raw_gate_logit = nn.Parameter(torch.tensor(float(raw_gate_init)))
        self.semantic_router = nn.Sequential(nn.Linear(3 * emb_dim, self.router_hidden_dim), nn.Tanh(), nn.Linear(self.router_hidden_dim, 3, bias=False))
        nn.init.xavier_normal_(self.semantic_router[0].weight)
        nn.init.zeros_(self.semantic_router[0].bias)
        nn.init.normal_(self.semantic_router[2].weight, mean=0.0, std=projection_init_std)
        semantic_scale_logit = math.log(self.semantic_scale_init / (1.0 - self.semantic_scale_init))
        self.semantic_scale_logit = nn.Parameter(torch.tensor(semantic_scale_logit))
        kc_semantic_dim = self.kc_semantic_bank.embedding_dim
        self.register_buffer('kc_name_embedding', self.kc_semantic_bank.name_tensor(), persistent=False)
        self.kc_global_proj = nn.Linear(kc_semantic_dim, emb_dim, bias=False)
        nn.init.zeros_(self.kc_global_proj.weight)
        self.edge_solution_proj = nn.Linear(kc_semantic_dim, emb_dim, bias=False)
        self.edge_distractor_proj = nn.Linear(kc_semantic_dim, emb_dim, bias=False)
        nn.init.zeros_(self.edge_solution_proj.weight)
        nn.init.zeros_(self.edge_distractor_proj.weight)

    @property
    def semantic_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.semantic_scale_logit)

    @staticmethod
    def _validate_gathered_bank(bank: Dict[str, torch.Tensor], batch_size: int, semantic_dim: int) -> None:
        expected = {'raw_embedding': (batch_size, semantic_dim), 'raw_available_mask': (batch_size,), 'solution_embedding': (batch_size, semantic_dim), 'solution_available_mask': (batch_size,), 'distractor_total_count': (batch_size,)}
        for name, shape in expected.items():
            if bank[name].shape != shape:
                raise ValueError(f'Invalid item bank batch {name}: expected {shape}, got {tuple(bank[name].shape)}')
        distractor_shape = bank['distractor_embedding'].shape
        if len(distractor_shape) != 3 or distractor_shape[0] != batch_size or distractor_shape[2] != semantic_dim:
            raise ValueError('Invalid batched Distractor embedding shape')
        if bank['distractor_available_mask'].shape != distractor_shape[:2]:
            raise ValueError('Distractor availability shape does not match cards')
        for name in ('raw_embedding', 'solution_embedding', 'distractor_embedding'):
            if not torch.isfinite(bank[name]).all():
                raise ValueError(f'Non-finite values found in batched {name}')

    @staticmethod
    def _pool_distractors(distractor_semantic: torch.Tensor, available_mask: torch.Tensor, total_count: torch.Tensor) -> torch.Tensor:
        denominator = total_count.clamp_min(1).to(distractor_semantic.dtype)
        return (distractor_semantic * available_mask.to(distractor_semantic.dtype).unsqueeze(-1)).sum(dim=1) / denominator.unsqueeze(-1)

    def _route_views(self, raw_view: torch.Tensor, solution_view: torch.Tensor, distractor_view: torch.Tensor, raw_active: torch.Tensor, solution_active: torch.Tensor, distractor_active: torch.Tensor) -> torch.Tensor:
        prior_logits = torch.stack((self.raw_gate_logit, self.solution_gate_logit, self.distractor_gate_logit))
        router_input = torch.cat((raw_view, solution_view, distractor_view), dim=-1)
        logits = prior_logits.unsqueeze(0) + self.semantic_router(router_input)
        available = torch.stack((raw_active, solution_active, distractor_active), dim=-1)
        logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        return torch.where(available.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))

    def _global_knowledge(self) -> torch.Tensor:
        semantic = self.kc_name_embedding.to(device=self.knowledge_emb.device, dtype=self.knowledge_emb.dtype)
        residual = torch.tanh(self.kc_global_proj(semantic))
        return self.knowledge_emb + residual

    def _edge_residual(self, exercise_ids: torch.Tensor, q_vector: torch.Tensor) -> torch.Tensor:
        batch_size = exercise_ids.shape[0]
        dense_shape = (batch_size, self.n_concepts, self.emb_dim)
        unique_items, inverse = torch.unique(exercise_ids, sorted=True, return_inverse=True)
        bank = self.kc_semantic_bank.gather(unique_items, device=exercise_ids.device)
        kc_idx = bank['kc_idx']
        safe_kc_idx = kc_idx.clamp_min(0)
        valid = bank['valid']
        solution_active = valid & (bank['solution_reliability'] > 0)
        distractor_active = valid & (bank['distractor_reliability'] > 0)
        solution = bank['solution_embedding'].to(self.knowledge_emb.dtype)
        distractor = bank['distractor_embedding'].to(self.knowledge_emb.dtype)
        solution_delta = torch.tanh(self.edge_solution_proj(solution))
        distractor_delta = torch.tanh(self.edge_distractor_proj(distractor))
        edge_delta = solution_delta * solution_active.unsqueeze(-1).to(solution_delta.dtype) + distractor_delta * distractor_active.unsqueeze(-1).to(distractor_delta.dtype)
        edge_delta = edge_delta * valid.unsqueeze(-1).to(edge_delta.dtype)
        unique_dense = self.knowledge_emb.new_zeros((unique_items.shape[0], self.n_concepts, self.emb_dim))
        if edge_delta.shape[1] > 0:
            scatter_index = safe_kc_idx.unsqueeze(-1).expand_as(edge_delta)
            unique_dense.scatter_add_(1, scatter_index, edge_delta)
        residual = unique_dense[inverse] * q_vector.unsqueeze(-1).to(unique_dense.dtype)
        return residual

    def semantic_augmentation(self, item_embedding: torch.Tensor, exercise_ids: torch.Tensor) -> torch.Tensor:
        unique_items, inverse = torch.unique(exercise_ids, sorted=True, return_inverse=True)
        bank = self.evidence_bank.gather(unique_items, device=item_embedding.device, dtype=item_embedding.dtype)
        self._validate_gathered_bank(bank, batch_size=unique_items.shape[0], semantic_dim=self.evidence_bank.embedding_dim)
        shape = (unique_items.shape[0], self.emb_dim)
        raw_view = torch.zeros(shape, dtype=item_embedding.dtype, device=item_embedding.device)
        raw_active = torch.zeros(unique_items.shape[0], dtype=torch.bool, device=item_embedding.device)
        raw_active = bank['raw_available_mask']
        raw_view = raw_active.to(item_embedding.dtype).unsqueeze(-1) * torch.tanh(self.raw_proj(bank['raw_embedding']))
        solution_view = torch.zeros_like(raw_view)
        solution_active = torch.zeros_like(raw_active)
        solution_active = bank['solution_available_mask']
        solution_view = solution_active.to(item_embedding.dtype).unsqueeze(-1) * torch.tanh(self.solution_proj(bank['solution_embedding']))
        distractor_view = torch.zeros_like(raw_view)
        distractor_active = torch.zeros_like(raw_active)
        card_active = bank['distractor_available_mask']
        distractor_semantic = torch.tanh(self.distractor_proj(bank['distractor_embedding']))
        distractor_view = self._pool_distractors(distractor_semantic, card_active, bank['distractor_total_count'])
        distractor_active = card_active.any(dim=1)
        gates = self._route_views(raw_view, solution_view, distractor_view, raw_active, solution_active, distractor_active)
        scale = self.semantic_scale
        raw_residual = scale * gates[:, 0:1] * raw_view
        solution_residual = scale * gates[:, 1:2] * solution_view
        distractor_residual = scale * gates[:, 2:3] * distractor_view
        raw_batch = raw_residual[inverse]
        solution_batch = solution_residual[inverse]
        distractor_batch = distractor_residual[inverse]
        return item_embedding + raw_batch + solution_batch + distractor_batch

class MOCD(_SemanticEncoder):

    def __init__(self, *args, beta=13.0, semantic_temperature=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        rng_state = torch.get_rng_state()
        self.student_level_calibration = torch.nn.Linear(self.emb_dim, 1, bias=False)
        torch.nn.init.zeros_(self.student_level_calibration.weight)
        torch.set_rng_state(rng_state)
        rng_state = torch.get_rng_state()
        self.difficulty_kc_adapter = torch.nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        torch.nn.init.zeros_(self.difficulty_kc_adapter.weight)
        torch.set_rng_state(rng_state)
        self.beta = float(beta)
        self.semantic_temperature = float(semantic_temperature)

    def _frozen_semantic_profile_weights(self, labels, mastery):
        bank = self.evidence_bank.gather(self._batch_exercises, device=mastery.device, dtype=mastery.dtype)
        raw_active = bank['raw_available_mask']
        solution_active = bank['solution_available_mask']
        wrong_active = bank['distractor_available_mask'].any(dim=1)
        wrong = self._pool_distractors(bank['distractor_embedding'], bank['distractor_available_mask'], bank['distractor_total_count'])
        y = labels.view(-1, 1).to(mastery.dtype)
        path = y * bank['solution_embedding'] + (1.0 - y) * wrong
        path_active = (y.view(-1) > 0.5) * solution_active + (y.view(-1) <= 0.5) * wrong_active
        count = raw_active.to(mastery.dtype) + path_active.to(mastery.dtype)
        evidence = (raw_active.to(mastery.dtype).unsqueeze(-1) * bank['raw_embedding'] + path_active.to(mastery.dtype).unsqueeze(-1) * path) / count.clamp_min(1.0).unsqueeze(-1)
        students, inverse = torch.unique(self._batch_students, sorted=True, return_inverse=True)
        pooled = torch.zeros((students.numel(), evidence.shape[1]), device=evidence.device, dtype=evidence.dtype)
        pooled.index_add_(0, inverse, evidence)
        denom = torch.bincount(inverse, minlength=students.numel()).to(evidence.dtype)
        pooled = pooled / denom.clamp_min(1.0).unsqueeze(1)
        kc_text = self.kc_name_embedding.to(device=mastery.device, dtype=mastery.dtype)
        similarity = F.normalize(pooled, dim=-1) @ F.normalize(kc_text, dim=-1).T
        return torch.softmax(self.semantic_temperature * similarity, dim=-1)

    def forward(self, stu_id, exer_id, q_vector):
        self._batch_students = stu_id
        self._batch_exercises = exer_id
        exercise = self.semantic_augmentation(self.exercise_emb(exer_id), exer_id)
        knowledge = self._global_knowledge()
        edge = self._edge_residual(exer_id, q_vector)
        student = self.student_emb(stu_id)
        batch = student.shape[0]
        se = student[:, None, :].expand(-1, self.n_concepts, -1)
        ke = knowledge[None, :, :].expand(batch, -1, -1)
        mastery_ke = ke
        hidden = torch.sigmoid(self.stat_full1(torch.cat((se, mastery_ke), dim=-1)))
        mastery_logit = self.stat_full2(hidden).view(batch, -1)
        mastery_logit = mastery_logit + self.student_level_calibration(student)
        mastery = torch.sigmoid(mastery_logit)
        ee = exercise[:, None, :].expand(-1, self.n_concepts, -1)
        item_k = ke + edge
        item_k = item_k + self.difficulty_kc_adapter(ke)
        d_hidden = torch.sigmoid(self.k_diff_full1(torch.cat((ee, item_k), dim=-1)))
        difficulty = torch.sigmoid(self.k_diff_full2(d_hidden)).view(batch, -1)
        mastery_pred, difficulty_pred = (mastery, difficulty)
        raw_discrimination = self.e_discrimination(exer_id)
        discrimination = torch.sigmoid(raw_discrimination)
        x = discrimination * (mastery_pred - difficulty_pred) * q_vector
        x = self.drop_1(torch.tanh(self.prednet_full1(x)))
        x = self.drop_2(torch.tanh(self.prednet_full2(x)))
        logit = self.prednet_full3(x).view(-1)
        return torch.sigmoid(logit)

    def objectives(self, predictions, labels):
        students = torch.unique(self._batch_students)
        mastery = self._mastery_for_students(students)
        per_example = F.binary_cross_entropy(predictions, labels, reduction='none')
        p = predictions.clamp(1e-06, 1.0 - 1e-06)
        uncertainty = -(p * p.log() + (1.0 - p) * (1.0 - p).log()).detach()
        weights = uncertainty / uncertainty.mean().clamp_min(1e-06)
        bce = (weights * per_example).mean()
        weights = self._frozen_semantic_profile_weights(labels, mastery)
        spread_values = mastery
        mean = (weights * spread_values).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1e-06)
        variance = (weights * (spread_values - mean).square()).sum() / weights.sum().clamp_min(1e-06)
        beta = self.beta
        negative_variance = -variance + beta * variance.square()
        return (bce, negative_variance)

    def _mastery_for_students(self, students):
        student = self.student_emb(students)
        knowledge = self._global_knowledge()
        s = student[:, None, :].expand(-1, self.n_concepts, -1)
        k = knowledge[None, :, :].expand(student.shape[0], -1, -1)
        hidden = torch.sigmoid(self.stat_full1(torch.cat((s, k), -1)))
        logit = self.stat_full2(hidden).squeeze(-1)
        logit = logit + self.student_level_calibration(student)
        return torch.sigmoid(logit)

    def _mastery_tensor(self):
        students = torch.arange(self.n_students, device=self.student_emb.weight.device)
        return self._mastery_for_students(students)

    @torch.no_grad()
    def get_all_knowledge_emb(self):
        return self._mastery_tensor().cpu().numpy()

    @torch.no_grad()
    def predict(self, student_ids, exercise_ids, q_vector):
        self.eval()
        return self(student_ids, exercise_ids, q_vector)
