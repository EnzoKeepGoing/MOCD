"""Formal Full training: uncertainty-weighted BCE + semantic MPC, module routing."""
import copy
import json
from pathlib import Path
import torch


def parameter_group(name):
    """Parameter ownership, not selectable experimental methods."""
    if name.startswith('difficulty_kc_adapter'):
        return 'difficulty'
    if name.startswith('student_emb'):
        return 'student'
    if name == 'knowledge_emb':
        return 'knowledge'
    if name.startswith('stat_full'):
        return 'mastery_head'
    if name.startswith(('exercise_emb', 'k_diff')):
        return 'difficulty'
    if name.startswith('e_discrimination'):
        return 'discrimination'
    if name.startswith('prednet'):
        return 'prediction_head'
    if 'proj' in name or 'gate' in name or name.startswith('semantic_'):
        return 'semantic'
    return 'other'


def packed_grads(loss, params, retain_graph):
    raw = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(raw, params)]


def route_gradients(primary, auxiliary, groups):
    """One fixed modulewise nonnegative alignment rule; no method switch."""
    result = list(primary)
    for group in sorted(set(groups)):
        indices = [i for i, name in enumerate(groups) if name == group]
        auxiliary_norm2 = sum((auxiliary[i] * auxiliary[i]).sum() for i in indices).clamp_min(1e-16)
        alignment = sum((primary[i] * auxiliary[i]).sum() for i in indices)
        alpha = (alignment / auxiliary_norm2).clamp_min(0.0)
        for i in indices:
            result[i] = primary[i] + alpha * auxiliary[i]
    return result


def train_step(model, optimizer, student_ids, exercise_ids, q_vector, labels):
    model.train()
    optimizer.zero_grad()
    predictions = model(student_ids, exercise_ids, q_vector)
    response_loss, mpc_loss = model.objectives(predictions, labels)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    groups = [parameter_group(n) for n, _ in named]
    primary = packed_grads(response_loss, params, retain_graph=True)
    auxiliary = packed_grads(mpc_loss, params, retain_graph=False)
    routed = route_gradients(primary, auxiliary, groups)
    for parameter, gradient in zip(params, routed):
        if not torch.isfinite(gradient).all():
            raise ValueError('Non-finite MOCD gradient; refusing optimizer step')
        parameter.grad = gradient
    optimizer.step()
    return dict(response_loss=float(response_loss.detach()), mpc_loss=float(mpc_loss.detach()))


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
    model.eval()
    truth, predictions = [], []
    for student, exercise, q, labels in loader:
        output = model(student.to(device), exercise.to(device), q.to(device))
        truth.extend(labels.cpu().tolist())
        predictions.extend(output.cpu().tolist())
    if len(set(truth)) != 2:
        raise ValueError('AUC requires both response labels')
    return dict(auc=float(roc_auc_score(truth, predictions)),
                acc=float(accuracy_score(truth, [p >= .5 for p in predictions])),
                rmse=float(mean_squared_error(truth, predictions)**.5))


def fit(model, train_loader, valid_loader, test_loader, output, *, lr, epochs=30, device='cpu'):
    """Validation AUC early stopping (patience 5), restore best, then test once.

    This portable runner consumes pre-existing splits; it never creates a new
    split, searches hyperparameters, or resumes an unrelated checkpoint.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=.5)
    best_score, best_epoch, bad_epochs, best_state = float('-inf'), None, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        for batch in train_loader:
            losses.append(train_step(model, optimizer, *(x.to(device) for x in batch)))
        valid = evaluate(model, valid_loader, device)
        scheduler.step(valid['auc'])
        if valid['auc'] > best_score:
            best_score, best_epoch, bad_epochs = valid['auc'], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output/'best_model.pt')
        else:
            bad_epochs += 1
        row = dict(epoch=epoch, response_loss=sum(x['response_loss'] for x in losses)/len(losses),
                   mpc_loss=sum(x['mpc_loss'] for x in losses)/len(losses), valid=valid)
        history.append(row)
        print(json.dumps(row), flush=True)
        if bad_epochs >= 5:
            break
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    import numpy as np
    np.save(output/'mastery.npy', model.get_all_knowledge_emb())
    result = dict(best_epoch=best_epoch, best_valid_auc=best_score, test=test, history=history,
                  selection='validation only; test once after best restoration')
    (output/'results.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return result
