"""Train only full MOCD from the released five-fold mapped response data."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from mmcd import MOCD, fit
from mmcd.item_evidence_bank import ItemEvidenceBank
from mmcd.kc_semantic_bank import KCSemanticBank
from mmcd.data import load_q_matrix


def read_responses(path, *, expected_fold=None, excluded_fold=None):
    students, exercises, labels = [], [], []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream)
        if set(rows.fieldnames or []) != {'user_id', 'question_id', 'correct', 'fold'}:
            raise ValueError(f'Unexpected response columns in {path}')
        for row in rows:
            fold = int(row['fold'])
            if expected_fold is not None and fold != expected_fold:
                continue
            if excluded_fold is not None and fold == excluded_fold:
                continue
            students.append(int(row['user_id']))
            exercises.append(int(row['question_id']))
            labels.append(float(row['correct']))
    return (np.asarray(students, dtype=np.int64), np.asarray(exercises, dtype=np.int64),
            np.asarray(labels, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--item-bank', type=Path, required=True)
    ap.add_argument('--kc-bank', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dry-run', action='store_true', help='Validate inputs and one forward pass; do not train')
    args = ap.parse_args()
    c = json.loads(args.config.read_text(encoding='utf8'))
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    # Dataset ID/Q hashes must match the two frozen evidence banks.
    ItemEvidenceBank.validate_dataset_binding(args.item_bank, args.data_dir, c['dataset'])
    KCSemanticBank.validate_dataset_binding(args.kc_bank, args.data_dir, c['dataset'])
    mapping = json.loads((args.data_dir/'id_mapping.json').read_text(encoding='utf8'))
    # Preserve the original global IDs, not per-split remapped or inferred IDs.
    n_students = len(mapping['uid'])
    n_exercises, n_concepts = len(mapping['questions']),len(mapping['concepts'])
    for name in ['uid','questions','concepts']:
        if sorted(mapping[name].values()) != list(range(len(mapping[name]))):
            raise ValueError('ID mapping must contain consecutive fixed zero-based IDs')
    q = load_q_matrix(args.data_dir/'q_matrix.csv', n_exercises,n_concepts)
    model_keys = ['emb_dim','hidden_dims','dropout','router_hidden_dim','gate_init',
                  'raw_gate_init','projection_init_std','semantic_scale_init','beta','semantic_temperature']
    model = MOCD(n_concepts, n_exercises, n_students, args.item_bank, args.kc_bank,
                 **{k:c[k] for k in model_keys})
    loaders = []
    train_generator = torch.Generator().manual_seed(int(c['seed']))
    fold = int(c['fold'])
    if fold not in range(5):
        raise ValueError('The released protocol uses folds 0 through 4')
    sources = [
        read_responses(args.data_dir/'train_valid.csv', excluded_fold=fold),
        read_responses(args.data_dir/'train_valid.csv', expected_fold=fold),
        read_responses(args.data_dir/'test.csv', expected_fold=-1),
    ]
    for name, (s, e, y) in zip(['train', 'valid', 'test'], sources):
        # Fold f is validation; all other rows of train_valid.csv are training.
        if not (s.ndim == e.ndim == y.ndim == 1 and len(s)==len(e)==len(y) and len(s)>0):
            raise ValueError('Malformed or empty split')
        if s.min()<0 or s.max()>=n_students or e.min()<0 or e.max()>=n_exercises or not np.isin(y,[0,1]).all():
            raise ValueError('Split IDs or response labels do not match the fixed dataset')
        data = TensorDataset(torch.from_numpy(s),torch.from_numpy(e),torch.from_numpy(q[e]),torch.from_numpy(y))
        loaders.append(DataLoader(data,batch_size=c['batch_size'],shuffle=name=='train',
                                  generator=train_generator if name=='train' else None,num_workers=0))
    # Preserve the recorded separation between model initialization and the
    # training/dropout RNG stream; the train-order generator is independent.
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.dry_run:
        model.to(args.device)
        model.eval()
        with torch.no_grad():
            sample = next(iter(loaders[1]))
            output = model(*(x.to(args.device) for x in sample[:3]))
        if output.ndim != 1 or not torch.isfinite(output).all():
            raise ValueError('Non-finite or malformed real-model forward output')
        print(json.dumps({'dry_run': 'passed', 'dataset': c['dataset'], 'fold': fold,
                          'rows': {key: len(value.dataset) for key, value in zip(['train','valid','test'],loaders)}}))
        return
    result = fit(model,*loaders,args.output,lr=c['lr'],epochs=c['epochs'],device=args.device)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'dataset': c['dataset'], 'fold': fold, 'config_sha256': sha(args.config),
                'data_sha256': {name: sha(args.data_dir/name) for name in
                                ['train_valid.csv','test.csv','q_matrix.csv','id_mapping.json']},
                'item_bank_fingerprint': model.evidence_bank.fingerprint,
                'kc_bank_fingerprint': model.kc_semantic_bank.fingerprint,
                'result_test': result['test']}
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')

if __name__ == '__main__':
    main()
