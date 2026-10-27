"""Install the six MOCD semantic banks from the separate asset archive."""
import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path

ARCHIVE_SHA256 = 'c940fb7e6d9d25eb4d40480596a38b2364708664767309177819be97b1e1c624'
PREFIX = 'probes/failure_aware/'
MAPPING = {
    'artifacts_g3_full_v2_7/egcd_item_bank_mpnet_v3': 'jiuzhang_g3/item_bank',
    'artifacts_g3_full_v2_7/egcd_kc_semantic_bank_mpnet_v1': 'jiuzhang_g3/kc_bank',
    'artifacts_g4_full/egcd_item_bank_mpnet_v3': 'jiuzhang_g4_g5/item_bank',
    'artifacts_g4_full/egcd_kc_semantic_bank_mpnet_v1': 'jiuzhang_g4_g5/kc_bank',
    'artifacts_g7_full_v2_7/egcd_item_bank_mpnet_v3': 'jiuzhang_g7/item_bank',
    'artifacts_g7_full_v2_7/egcd_kc_semantic_bank_mpnet_v1': 'jiuzhang_g7/kc_bank',
}


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
    print('Installed six semantic banks under', args.output)


if __name__ == '__main__':
    main()
