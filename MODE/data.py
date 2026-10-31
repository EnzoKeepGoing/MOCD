"""Fixed original Q IDs; no KC compression, filtering, or data splitting."""
import ast
import csv
import json
from pathlib import Path
import numpy as np


def load_q_matrix(path, n_exercises, n_concepts):
    """Read original PyCD question_id,concept_ids CSV with already-remapped IDs."""
    q = np.zeros((n_exercises,n_concepts),dtype=np.float32)
    seen = set()
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        rows = csv.DictReader(stream)
        if not {'question_id','concept_ids'} <= set(rows.fieldnames or []):
            raise ValueError('Q CSV requires question_id,concept_ids columns')
        for row in rows:
            item = int(row['question_id'])
            if item in seen or not 0 <= item < n_exercises:
                raise ValueError('Duplicate or out-of-range Q question ID')
            seen.add(item)
            try:
                concepts = json.loads(row['concept_ids'])
            except json.JSONDecodeError:
                concepts = ast.literal_eval(row['concept_ids'])
            if not isinstance(concepts,list):
                concepts = [concepts]
            for kc in concepts:
                if isinstance(kc,bool) or not isinstance(kc,int) or not 0 <= kc < n_concepts:
                    raise ValueError('Invalid Q KC ID')
                q[item,kc] = 1
    if seen != set(range(n_exercises)):
        raise ValueError('Q CSV does not cover every fixed question ID')
    return q
