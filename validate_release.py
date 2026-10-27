"""Check released mappings, Q matrices, folds and bank binding without PyTorch."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

DATASETS = ('jiuzhang_g3', 'jiuzhang_g4_g5', 'jiuzhang_g7')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_dataset(root, dataset, assets):
    data = root/'data'/dataset
    mapping = json.loads((data/'id_mapping.json').read_text(encoding='utf8'))
    counts = {name: len(mapping[name]) for name in ('uid','questions','concepts')}
    for name, count in counts.items():
        if sorted(mapping[name].values()) != list(range(count)):
            raise ValueError(f'{dataset}: non-consecutive {name} IDs')
    question_ids = set()
    with (data/'q_matrix.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            item = int(row['question_id'])
            if item in question_ids or not 0 <= item < counts['questions']:
                raise ValueError(f'{dataset}: invalid Q item ID')
            question_ids.add(item)
            kc_ids = ast.literal_eval(row['concept_ids'])
            if not isinstance(kc_ids, list):
                kc_ids = [kc_ids]
            if not kc_ids or any(not isinstance(k, int) or not 0 <= k < counts['concepts'] for k in kc_ids):
                raise ValueError(f'{dataset}: invalid Q KC IDs')
    if len(question_ids) != counts['questions']:
        raise ValueError(f'{dataset}: Q matrix does not cover all items')
    response_counts = {}
    for filename in ('train_valid.csv','test.csv'):
        folds = {i: 0 for i in range(5)} if filename == 'train_valid.csv' else {-1: 0}
        with (data/filename).open(encoding='utf-8-sig', newline='') as stream:
            rows = csv.DictReader(stream)
            if set(rows.fieldnames or []) != {'user_id','question_id','correct','fold'}:
                raise ValueError(f'{dataset}: invalid {filename} columns')
            for row in rows:
                fold = int(row['fold'])
                if (fold not in folds or not 0 <= int(row['user_id']) < counts['uid']
                    or not 0 <= int(row['question_id']) < counts['questions']
                    or float(row['correct']) not in (0.0, 1.0)):
                    raise ValueError(f'{dataset}: invalid {filename} response')
                folds[fold] += 1
        if any(value == 0 for value in folds.values()):
            raise ValueError(f'{dataset}: empty fold in {filename}')
        response_counts[filename] = folds
    for fold in range(5):
        cfg = json.loads((root/'configs'/f'{dataset}_fold{fold}.json').read_text(encoding='utf8'))
        if cfg['dataset'] != dataset or cfg['fold'] != fold or cfg['epochs'] != 30:
            raise ValueError(f'{dataset}: invalid fold {fold} config identity')
    bank_state = 'not checked'
    if assets is not None:
        for bank in ('item_bank','kc_bank'):
            metadata = json.loads((assets/dataset/bank/'metadata.json').read_text(encoding='utf8'))
            if metadata['dataset'] != dataset:
                raise ValueError(f'{dataset}: wrong {bank} dataset')
            for filename, key in (('q_matrix.csv','q_matrix_sha256'),
                                  ('id_mapping.json','id_mapping_sha256')):
                if sha256(data/filename) != metadata[key]:
                    raise ValueError(f'{dataset}: {bank} mismatches {filename}')
        bank_state = 'Q and ID mapping bindings pass'
    return {'dataset': dataset, 'counts': counts, 'responses': response_counts, 'banks': bank_state}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', type=Path, help='Optional extracted semantic asset directory')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    for dataset in DATASETS:
        print(json.dumps(check_dataset(root, dataset, args.assets), ensure_ascii=False))


if __name__ == '__main__':
    main()
