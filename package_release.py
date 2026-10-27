"""Create the MOCD GitHub source/data ZIP without caches or semantic assets."""
import hashlib
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    destination = root.parent/'MOCD_github_source_20261027_final.zip'
    if destination.exists():
        raise FileExistsError(destination)
    allowed = {'configs', 'data', 'mmcd', 'results'}
    files = [path for path in root.rglob('*') if path.is_file()
             and '__pycache__' not in path.parts
             and (path.relative_to(root).parts[0] in allowed
                  or len(path.relative_to(root).parts) == 1)]
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, arcname='MOCD/' + path.relative_to(root).as_posix())
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(destination, destination.stat().st_size, 'bytes', 'SHA256', digest)


if __name__ == '__main__':
    main()
