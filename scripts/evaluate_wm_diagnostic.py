"""Evaluate checkpoints on one frozen, open-loop, in-library diagnostic dataset."""
import argparse,hashlib,json,time,platform
from pathlib import Path
import numpy as np
import torch
from wm_diagnostic_runtime import load_world_model,validate_pose_kinematics

def file_sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def evaluate(code,checkpoint,dataset,output,device='cpu',batch=32,max_windows=None):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads(dataset.with_suffix('.manifest.json').read_text())
    if file_sha(dataset)!=manifest['dataset_sha256']:raise ValueError('Diagnostic hash mismatch')
    data=np.load(dataset,allow_pickle=False)
    states=torch.tensor(data['states'],dtype=torch.float32,device=device)
    actions=torch.tensor(data['actions'],dtype=torch.float32,device=device)
    xi=torch.tensor(data['xi'],dtype=torch.float32,device=device)
    if states.shape[1:]!=(17,71) or actions.shape[1:]!=(16,29) or xi.shape[1]!=33:raise ValueError('Dataset layout mismatch')
    if not torch.all(xi[:,0]==1) or not torch.all(xi[:,1:]==0):raise ValueError('Unexpected diagnostic context')
    wm,ck,info=load_world_model(code,checkpoint,device)
    from allmimic.task import flat_to_state
    from allmimic.networks import state_delta,state_error
    info['pose_kinematics_check']=validate_pose_kinematics(wm,states[:,0])
    # Common scales depend ONLY on the immutable diagnostic ground truth, never on a checkpoint normalizer.
    # Compute them on CPU even when model inference is on GPU, avoiding device-specific reductions.
    with torch.no_grad():
        ground_truth_cpu=torch.tensor(data['states'],dtype=torch.float32)
        deltas=state_delta(flat_to_state(ground_truth_cpu[:,:-1].reshape(-1,71)),flat_to_state(ground_truth_cpu[:,1:].reshape(-1,71)))
        scale=deltas.std(0,unbiased=False).clamp_min(1e-3)
    total=len(states);n=min(total,max_windows) if max_windows else total
    results={h:[] for h in [1,4,8,16]};baselines={h:[] for h in results}
    if device.startswith('cuda'):torch.cuda.synchronize()
    t0=time.perf_counter()
    with torch.no_grad():
        for start in range(0,n,batch):
            end=min(start+batch,n);initial=flat_to_state(states[start:end,0]);pred=initial
            for h in range(1,17):
                pred=wm.step(pred,actions[start:end,h-1],xi[start:end])
                if h in results:
                    true=flat_to_state(states[start:end,h])
                    results[h].append(state_error(pred,true).cpu().numpy())
                    baselines[h].append(state_error(initial,true).cpu().numpy())
            print(f'evaluated {end}/{n} windows',flush=True)
    if device.startswith('cuda'):torch.cuda.synchronize()
    prediction_seconds=time.perf_counter()-t0
    scale_np=scale.cpu().numpy()
    def summarize(error):
        if not np.isfinite(error).all():raise ValueError('Nonfinite model prediction')
        return {'root_position_rmse_m':float(np.sqrt(np.mean(error[:,:3]**2))),
                'root_rotation_mean_deg':float(np.rad2deg(np.linalg.norm(error[:,3:6],axis=1)).mean()),
                'root_linear_velocity_rmse_m_s':float(np.sqrt(np.mean(error[:,6:9]**2))),
                'root_local_angular_velocity_rmse_rad_s':float(np.sqrt(np.mean(error[:,9:12]**2))),
                'joint_position_rmse_rad':float(np.sqrt(np.mean(error[:,12:41]**2))),
                'joint_velocity_rmse_rad_s':float(np.sqrt(np.mean(error[:,41:]**2))),
                'common_scaled_mse':float(np.mean((error/scale_np)**2))}
    per_h={};per_clip={}
    for h in results:
        errors=np.concatenate(results[h]);reference=np.concatenate(baselines[h])
        per_h[str(h)]={'world_model':summarize(errors),'persistence':summarize(reference)}
        per_clip[str(h)]={}
        for cid in np.unique(data['clip_id'][:n]):
            mask=data['clip_id'][:n]==cid
            per_clip[str(h)][str(data['names'][cid])]={'windows':int(mask.sum()),'world_model':summarize(errors[mask]),'persistence':summarize(reference[mask])}
    report={'checkpoint_sha256':file_sha(checkpoint),'checkpoint_iteration':ck['it'],
            'checkpoint_training_horizon':ck['cfg']['horizon'],'diagnostic_sha256':manifest['dataset_sha256'],
            'windows_evaluated':n,'total_windows':total,'subset_smoke_only':n!=total,
            'common_scale_sha256':hashlib.sha256(scale_np.tobytes()).hexdigest(),'common_scale':scale_np.tolist(),
            'prediction_seconds':prediction_seconds,'scope':manifest['scope'],'runtime':info,
            'cpu_platform':platform.processor(),'metrics':per_h,'per_clip_metrics':per_clip,
            'protocol':'Open-loop logged actions, no teacher-forced state refresh; equal samples across checkpoints. CPU eager results are not GPU parity evidence.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--max-windows',type=int,help='Smoke subset only, not full-dataset benchmark')
    args=ap.parse_args();torch.set_num_threads(4)
    r=evaluate(args.code,args.checkpoint,args.dataset,args.out,args.device,args.batch,args.max_windows)
    print(json.dumps({'out':str(args.out),'windows':r['windows_evaluated'],'metrics':r['metrics']},indent=2))
