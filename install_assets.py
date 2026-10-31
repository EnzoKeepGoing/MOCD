"""Install the six MOCD semantic banks from the separate asset archive."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ARCHIVE_SHA256 = '8d2f961e7ed708d00d4bfb2d795280031d2f6088165eff28c13a060f6d93cec8'
PREFIX = 'probes/failure_aware/'
MAPPING = {
    'artifacts_g3_full_v2_7/egcd_item_bank_mpnet_v3': 'Grade3/item_bank',
    'artifacts_g3_full_v2_7/egcd_kc_semantic_bank_mpnet_v1': 'Grade3/kc_bank',
    'artifacts_g4_full/egcd_item_bank_mpnet_v3': 'Math50k/item_bank',
    'artifacts_g4_full/egcd_kc_semantic_bank_mpnet_v1': 'Math50k/kc_bank',
    'artifacts_g7_full_v2_7/egcd_item_bank_mpnet_v3': 'MidMath/item_bank',
    'artifacts_g7_full_v2_7/egcd_kc_semantic_bank_mpnet_v1': 'MidMath/kc_bank',
}
FINGERPRINT_KEYS = {
    'item_bank': ('schema_version', 'dataset', 'num_items', 'embedding_dim',
                  'num_distractor_cards', 'encoder_name', 'encoder_revision',
                  'max_length', 'candidates_sha256', 'evidence_records_sha256',
                  'q_matrix_sha256', 'id_mapping_sha256', 'array_sha256'),
    'kc_bank': ('schema_version', 'dataset', 'num_items', 'num_concepts',
                'num_edges', 'embedding_dim', 'encoder_name', 'encoder_revision',
                'max_length', 'candidates_sha256', 'evidence_records_sha256',
                'q_matrix_sha256', 'id_mapping_sha256', 'kc_names_sha256',
                'text_template_version', 'array_sha256'),
}


def use_dataset_name(bank_dir: Path, dataset: str, bank_type: str) -> None:
    metadata_path = bank_dir / 'metadata.json'
    with metadata_path.open('r', encoding='utf-8') as stream:
        metadata = json.load(stream)
    data_dir = Path(__file__).resolve().parent / 'data' / dataset
    for filename, key in (('q_matrix.csv', 'q_matrix_sha256'),
                          ('id_mapping.json', 'id_mapping_sha256')):
        source = data_dir / filename
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != metadata.get(key):
            raise ValueError(f'Semantic bank does not match {dataset}: {filename}')
    source_name = metadata.get('dataset')
    metadata['dataset'] = dataset
    payload = {key: metadata.get(key) for key in FINGERPRINT_KEYS[bank_type]}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':')).encode('utf-8')
    metadata['bank_fingerprint'] = hashlib.sha256(encoded).hexdigest()
    with metadata_path.open('w', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, sort_keys=True,
                  separators=(',', ':'))
    coverage_path = bank_dir / 'coverage_report.json'
    if coverage_path.is_file():
        with coverage_path.open('r', encoding='utf-8') as stream:
            coverage = json.load(stream)
        if isinstance(coverage, dict) and coverage.get('dataset') == source_name:
            coverage['dataset'] = dataset
            with coverage_path.open('w', encoding='utf-8') as stream:
                json.dump(coverage, stream, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive', type=Path)
    ap.add_argument('--output', type=Path, default=Path('assets'))
    args = ap.parse_args()
    digest = hashlib.sha256()
    with args.archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Semantic asset archive SHA256 mismatch')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'{args.output} is not empty')
    installed = set()
    with tarfile.open(args.archive, 'r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or not member.name.startswith(PREFIX):
                raise ValueError(f'Unexpected archive member: {member.name}')
            relative = member.name[len(PREFIX):]
            matches = [(old, new) for old, new in MAPPING.items()
                       if relative.startswith(old + '/')]
            if len(matches) != 1:
                raise ValueError(f'Unexpected bank path: {member.name}')
            old, new = matches[0]
            filename = relative[len(old)+1:]
            if not filename or '/' in filename or filename in ('.','..'):
                raise ValueError(f'Unexpected bank filename: {member.name}')
            destination = args.output/new/filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            with archive.extractfile(member) as source, destination.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            installed.add(new)
    if installed != set(MAPPING.values()):
        raise ValueError('Asset archive is missing one or more banks')
    for bank in sorted(installed):
        dataset, bank_type = bank.split('/')
        use_dataset_name(args.output / bank, dataset, bank_type)
    print('Installed six semantic banks under', args.output)


if __name__ == '__main__':
    main()
