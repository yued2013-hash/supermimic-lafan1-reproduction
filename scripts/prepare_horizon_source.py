"""Create a separate, traceable upstream copy with a fixed policy-loss divisor."""
from pathlib import Path
import argparse,ast,hashlib,json,shutil

COMMIT='cb334ff674b2f57248f33f8c21b7fb526fbdf8cc'
TRAIN_SHA256='b422a8db1f7ba8899fa82b6aaa1eced937fadb18bc65e23dbd974957e1a14d0e'
ANCHOR='loss_pi = -J.mean() / H'
REPLACEMENT='loss_pi = -J.mean() / 32.0  # controlled horizon study: common numerical scale'

def transform(source):
    if source.count(ANCHOR)!=1:raise ValueError('Expected exactly one unmodified policy-loss expression')
    result=source.replace(ANCHOR,REPLACEMENT)
    ast.parse(result)
    return result

def prepare(upstream,output):
    upstream=upstream.resolve();output=output.resolve()
    if output==upstream or upstream in output.parents or output in upstream.parents:
        raise ValueError('Source/output must be separate, non-nested directories')
    path=upstream/'allmimic/train.py';before=path.read_bytes()
    if hashlib.sha256(before).hexdigest()!=TRAIN_SHA256:raise ValueError('Unexpected pinned upstream train.py hash')
    if output.exists():raise FileExistsError(output)
    patched=transform(before.decode('utf-8')).encode('utf-8')
    shutil.copytree(upstream,output,ignore=shutil.ignore_patterns('.git','data','runs','logs','__pycache__'))
    (output/'allmimic/train.py').write_bytes(patched)
    record={'upstream_commit':COMMIT,'original_train_sha256':TRAIN_SHA256,
            'patched_train_sha256':hashlib.sha256(patched).hexdigest(),
            'policy_loss_divisor':32,'policy_gradient_norm_clip':1/32,
            'change':'Use a common divisor32 for both H16/H32; all other source unchanged',
            'validation':'Source transform/AST checks only; this does not establish GPU gradient parity'}
    (output/'HORIZON_PATCH.json').write_text(json.dumps(record,indent=2))
    return record

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();print(json.dumps(prepare(args.upstream,args.output),indent=2))
