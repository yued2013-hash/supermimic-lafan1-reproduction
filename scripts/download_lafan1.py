"""Download the exact 40 source CSVs by pinned revision, verifying every SHA256."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REVISION='ce1572906efe6157840e8474d5a0d7aa87481e74'

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    manifest=json.loads((ROOT/'results/lafan1_seed0/data_manifest.json').read_text())
    if manifest['revision']!=REVISION or len(manifest['files'])!=40:raise ValueError('Unexpected source manifest')
    for i,entry in enumerate(manifest['files']):
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid manifest path')
        target=args.output/relative
        if target.exists():
            if digest(target)!=entry['sha256']:raise ValueError(f'Existing file differs: {target}')
            print(f'{i+1}/40 verified {relative}');continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.partial')
        url=f'https://huggingface.co/datasets/{manifest["repo"]}/resolve/{REVISION}/{relative.as_posix()}'
        with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as f:
            for block in iter(lambda:response.read(1024*1024),b''):f.write(block)
        if temporary.stat().st_size!=entry['bytes'] or digest(temporary)!=entry['sha256']:
            raise ValueError(f'Download checksum mismatch; partial retained: {temporary}')
        temporary.rename(target)
        print(f'{i+1}/40 downloaded and verified {relative}')

if __name__=='__main__':main()
