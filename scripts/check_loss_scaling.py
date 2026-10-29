"""Autograd scaling/clipping checks through the actual frozen WM, using a probe objective.

This is NOT an actor-critic/analytic-task-reward or torch.compile equivalence test.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from wm_diagnostic_runtime import load_world_model,validate_pose_kinematics

def check(code,checkpoint,dataset,out):
    if out.exists():raise FileExistsError(out)
    manifest=json.loads(dataset.with_suffix('.manifest.json').read_text())
    with dataset.open('rb') as f:dataset_sha=hashlib.file_digest(f,'sha256').hexdigest()
    if dataset_sha!=manifest['dataset_sha256']:raise ValueError('Diagnostic dataset hash mismatch')
    wm,ck,info=load_world_model(code,checkpoint,'cpu')
    from allmimic.task import flat_to_state
    d=np.load(dataset,allow_pickle=False)
    initial=torch.tensor(d['states'][:2,0]);logged=torch.tensor(d['actions'][:2]);xi=torch.tensor(d['xi'][:2])
    info['pose_kinematics_check']=validate_pose_kinematics(wm,initial)
    records=[]
    for horizon in [16,32]:
        theta=torch.zeros(2,29,requires_grad=True)
        def objective(parameter):
            state=flat_to_state(initial);j=0
            for k in range(horizon):
                state=wm.step(state,logged[:,min(k,15)]+parameter,xi)
                # Smooth bounded state probe; deliberately not the training reward/J.
                j=j+(0.99**k)*torch.exp(-state['q'].square().mean(-1)).mean()
            return j
        j=objective(theta)
        raw=torch.autograd.grad(-j,theta,retain_graph=True)[0]
        native=torch.autograd.grad(-j/horizon,theta,retain_graph=True)[0]
        fixed=torch.autograd.grad(-j/32,theta)[0]
        torch.testing.assert_close(native,raw/horizon,rtol=1e-4,atol=1e-7)
        torch.testing.assert_close(fixed,raw/32,rtol=1e-4,atol=1e-7)
        if not torch.isfinite(raw).all() or raw.norm()<=0:raise ValueError('Invalid/non-informative WM probe gradient')
        # Exercise the actual PyTorch norm-clipping operator on the same gradient.
        clipped=torch.nn.Parameter(torch.zeros_like(theta));clipped.grad=fixed.detach().clone()
        norm=torch.nn.utils.clip_grad_norm_([clipped],1/32)
        if clipped.grad.norm()>1/32+1e-6:raise ValueError('Clipping cap exceeded')
        direction=raw.detach()/raw.detach().norm();eps=.001
        with torch.no_grad():
            fd=float((-objective(eps*direction)+objective(-eps*direction))/(2*eps))
        ad=float((raw*direction).sum());relative=abs(fd-ad)/max(abs(fd),abs(ad),1e-6)
        if relative>.05:raise ValueError(f'Directional finite difference mismatch: {relative}')
        records.append({'horizon':horizon,'raw_gradient_norm':float(raw.norm()),
                        'native_divisor_gradient_norm':float(native.norm()),'fixed32_gradient_norm':float(fixed.norm()),
                        'post_clip_norm':float(clipped.grad.norm()),'finite_difference_relative_error':relative,
                        'native_raw_objective_norm_cap':horizon/32,'fixed32_raw_objective_norm_cap':1})
    with checkpoint.open('rb') as f:checkpoint_sha=hashlib.file_digest(f,'sha256').hexdigest()
    report={'status':'CPU_EAGER_WM_PROBE_PASS','runtime':info,'checks':records,
            'diagnostic_sha256':dataset_sha,'checkpoint_sha256':checkpoint_sha,
            'checkpoint_iteration':ck['it'],'h32_action_extension':'Last logged action repeated beyond16, for synthetic gradient probe only',
            'limits':'Probe uses action-offset parameters and a smooth state objective. Not the actual policy parameterization, critic bootstrap, task reward, optimizer update, or compiled/GPU BPTT.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(4);print(json.dumps(check(args.code,args.checkpoint,args.dataset,args.out),indent=2))
