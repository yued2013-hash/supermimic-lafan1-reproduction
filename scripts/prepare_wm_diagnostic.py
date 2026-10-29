"""Freeze clip-balanced open-loop windows from archived physical rollouts. Requires numpy."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def quat_matrix(q):
    q=np.asarray(q,dtype=np.float64)
    if not np.allclose(np.linalg.norm(q,axis=-1),1,atol=1e-4):raise ValueError('Non-unit wxyz quaternion')
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    values=[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]
    return np.stack(values,-1).reshape(q.shape[:-1]+(3,3))

def to_state(qpos,qvel):
    if qpos.shape[-1]!=36 or qvel.shape[-1]!=35:raise ValueError('Expected G1 nq36/nv35')
    world_omega=np.einsum('...ij,...j->...i',quat_matrix(qpos[...,3:7]),qvel[...,3:6])
    return np.concatenate([qpos[...,:36],qvel[...,:3],world_omega,qvel[...,6:]],-1).astype('f4')

def build(source,output,windows_per_clip=16,seed=20261029):
    if output.exists() or output.with_suffix('.manifest.json').exists():raise FileExistsError(output)
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['simulation_hz']!=50 or manifest['precision']!='fp32' or not manifest['all_complete'] or manifest['failed']:
        raise ValueError('Expected complete deterministic fp32 evaluation rollouts')
    clips=sorted(manifest['clips'],key=lambda x:x['name'])
    if len(clips)!=40 or len({r['name'] for r in clips})!=40:raise ValueError('Need40 unique clips')
    rng=np.random.default_rng(seed);states=[];actions=[];ids=[];starts=[];provenance=[]
    for i,row in enumerate(clips):
        path=source/row['rollout']
        with np.load(path,allow_pickle=False) as d:
            qp=d['qpos'];qv=d['qvel'];a=d['absolute_setpoint_action']
            if len(qp)!=row['required_steps']+1 or len(a)!=len(qp)-1 or len(qv)!=len(qp):raise ValueError('Incomplete/unaligned trajectory')
            if float(d['fps'])!=50 or a.shape[1]!=29:raise ValueError('Action/fps mismatch')
            if not all(np.isfinite(v).all() for v in [qp,qv,a]):raise ValueError('Nonfinite rollout')
            available=len(a)-16+1
            if available<windows_per_clip:raise ValueError('Insufficient distinct windows')
            ix=np.sort(rng.choice(available,windows_per_clip,replace=False))
            for start in ix:
                states.append(to_state(qp[start:start+17],qv[start:start+17]))
                actions.append(a[start:start+16].astype('f4'));ids.append(i);starts.append(int(start))
        provenance.append({'name':row['name'],'sha256':sha(path),'starts':ix.tolist()})
    s=np.stack(states);a=np.stack(actions);xi=np.zeros((len(s),33),dtype='f4');xi[:,0]=1
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,states=s,actions=a,xi=xi,clip_id=np.array(ids),start_frame=np.array(starts),names=np.array([r['name'] for r in clips]))
    record={'dataset_sha256':sha(output),'seed':seed,'windows_per_clip':windows_per_clip,'windows':len(s),
            'horizon':16,'state_layout':'pos3|quat_wxyz4|q29|world_v3|world_w3|qd29',
            'action':'absolute_setpoint_action, not raw policy residual',
            'context':'xi=[friction1,29 zero encoder biases,3 zero COM offsets], reconstructed from fixed evaluation protocol',
            'source_precision':'fp32','source_policy_iteration':manifest['checkpoint_iteration'],
            'scope':'Fixed in-library diagnostic trajectories, not unseen-motion/held-out-policy performance',
            'source_manifest_sha256':sha(source/'manifest.json'),'sources':provenance}
    output.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2))
    return record

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rollouts',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--windows-per-clip',type=int,default=16);ap.add_argument('--seed',type=int,default=20261029)
    args=ap.parse_args();r=build(args.rollouts,args.output,args.windows_per_clip,args.seed)
    print(json.dumps({k:v for k,v in r.items() if k!='sources'},indent=2))
