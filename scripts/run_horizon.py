"""Print, or explicitly execute, one bounded horizon study on a GPU Linux instance."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,time,zipfile,math

def validate_motion_container(path):
    with zipfile.ZipFile(path) as archive:names=set(archive.namelist())
    metadata={'names.npy','starts.npy','lengths.npy','fps.npy'}
    layouts=[{'qpos.npy','qvel.npy'},{'root_pos.npy','rest.npy'}]
    if not metadata<=names or not any(layout<=names for layout in layouts):
        raise ValueError('Expected a motion library NPZ, not a WM diagnostic or arbitrary NPZ')

def command_for(code,motions,output,horizon,seed=0,smoke=False):
    code=code.resolve();motions=motions.resolve();output=output.resolve()
    metadata=json.loads((code/'HORIZON_PATCH.json').read_text())
    actual=hashlib.sha256((code/'allmimic/train.py').read_bytes()).hexdigest()
    if actual!=metadata['patched_train_sha256'] or metadata['policy_loss_divisor']!=32:raise ValueError('Controlled source hash/divisor mismatch')
    if output.exists():raise FileExistsError(output)
    if not motions.is_file():raise FileNotFoundError(motions)
    validate_motion_container(motions)
    values={'preset':'recommended','motions':str(motions),'out':str(output),'seed':seed,
            'horizon':horizon,'pi_clip':1/32,'num_envs':64 if smoke else 4096,
            'pi_batch':128 if smoke else 4096,'wm_batch':128 if smoke else 2048,
            'buffer_steps':100 if smoke else 1000,'warmup_steps':16 if smoke else 64,
            'iterations':50 if smoke else 8000,'log_every':10 if smoke else 100,
            'wall_hours':1e9,'ckpt_every':50 if smoke else 2000,
            'eval_every':0 if smoke else 2000,'eval_clips':0,'eval_envs':40}
    command=[sys.executable,'-u','-m','allmimic.train']
    for k,v in values.items():command+=['--'+k.replace('_','-'),str(v)]
    return command,metadata,values

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code',type=Path,required=True);ap.add_argument('--motions',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--horizon',type=int,choices=[16,32],required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--execute',action='store_true',help='Without this flag no workload or output file is created')
    ap.add_argument('--max-seconds',type=int,default=5400)
    args=ap.parse_args();command,patch,config=command_for(args.code,args.motions,args.out,args.horizon,args.seed,args.smoke)
    print(json.dumps({'command':command,'source_patch':patch,'smoke_only':args.smoke},indent=2),flush=True)
    if not args.execute:raise SystemExit(0)
    if os.name!='posix':raise SystemExit('GPU execution is supported on Linux; local dry-run only')
    import torch
    if not torch.cuda.is_available():raise SystemExit('CUDA unavailable; no training started')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    record=args.out.with_name(args.out.name+'.launch.json');console=args.out.with_name(args.out.name+'.console.log')
    if record.exists() or console.exists():raise FileExistsError('Refusing to overwrite launcher records')
    state={'config':config,'patch':patch,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
           'status':'RUNNING','smoke_only':args.smoke,'max_seconds':args.max_seconds}
    with args.motions.open('rb') as f:state['motion_library_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    record.write_text(json.dumps(state,indent=2));env=os.environ.copy()
    env.setdefault('OMP_NUM_THREADS','4');env.setdefault('MKL_NUM_THREADS','4')
    env['PYTHONPATH']=str(args.code.resolve())+os.pathsep+env.get('PYTHONPATH','')
    t0=time.perf_counter()
    with console.open('w') as log:
        process=subprocess.Popen(command,cwd=args.code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:exit_code=process.wait(timeout=args.max_seconds);status='EXITED'
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            exit_code=124;status='TIMEOUT_NOT_EQUAL_BUDGET_COMPLETE'
    state.update(status=status,exit_code=exit_code,total_process_seconds=time.perf_counter()-t0)
    state['done_marker']=bool((args.out/'DONE').exists())
    if exit_code==0 and state['done_marker']:
        ck=torch.load(args.out/'last.pt',map_location='cpu',weights_only=False)
        state.update(checkpoint_iteration=ck['it'],env_steps=ck['env_steps'],accumulated_train_seconds=ck['train_time'])
        state['status']='TRAINING_BUDGET_COMPLETE' if ck['it']==config['iterations'] else 'INCOMPLETE_ITERATION_BUDGET'
        with (args.out/'last.pt').open('rb') as f:state['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        logs=[json.loads(line) for line in (args.out/'log.jsonl').read_text().splitlines() if line.strip()]
        state['training_logs_finite']=bool(logs) and all(math.isfinite(v) for row in logs for v in row.values() if isinstance(v,(int,float)))
        if not state['training_logs_finite']:state['status']='FAILED_NONFINITE_OR_MISSING_LOGS';exit_code=1;state['exit_code']=1
    record.write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
    raise SystemExit(exit_code)
