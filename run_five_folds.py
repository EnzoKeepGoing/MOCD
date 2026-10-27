"""Run one MOCD dataset's five recorded folds sequentially on one device."""
import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = {'Grade3': 'jiuzhang_g3', 'Math50k': 'jiuzhang_g4_g5',
            'MidMath': 'jiuzhang_g7'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    parser.add_argument('--output-root', type=Path, default=Path('runs'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    dataset_id = DATASETS[args.dataset]
    for fold in range(5):
        output = args.output_root / args.dataset / f'fold{fold}'
        if output.exists() and not args.dry_run:
            raise FileExistsError(f'{output} exists; inspect it before rerunning')
        command = [sys.executable, str(root/'train.py'), '--config',
                   str(root/'configs'/f'{dataset_id}_fold{fold}.json'),
                   '--data-dir', str(root/'data'/dataset_id),
                   '--item-bank', str(args.assets/dataset_id/'item_bank'),
                   '--kc-bank', str(args.assets/dataset_id/'kc_bank'),
                   '--output', str(output), '--device', args.device]
        if args.dry_run:
            command.append('--dry-run')
        print(f'{args.dataset} fold {fold}: starting', flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
