"""Load the pinned upstream WM eagerly, without constructing a simulator or importing mjlab."""
from pathlib import Path
import ast,hashlib,importlib.metadata,re,sys

def load_world_model(code,checkpoint,device='cpu'):
    import torch,mujoco
    import numpy as np
    sys.path.insert(0,str(Path(code).resolve()))
    from allmimic import networks,rotations
    from allmimic.kinematics import FastTreeKinematics
    from allmimic.task import Task,task_cfg
    networks.AMP['enabled']=False
    torch.backends.cuda.matmul.allow_tf32=False
    # Import-time compile wrappers are replaced by their upstream eager functions.
    for name in ['apply_delta','state_error','state_delta','apply_delta_tri','state_delta_tri']:
        setattr(networks,name+'_c',getattr(networks,name))
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    cfg=ck['cfg'];tc=task_cfg(cfg,train=False)
    dist=importlib.metadata.distribution('mjlab')
    asset=Path(dist.locate_file('mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'))
    constants=Path(dist.locate_file('mjlab/asset_zoo/robots/unitree_g1/g1_constants.py'))
    tree=ast.parse(constants.read_text())
    home=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KNEES_BENT_KEYFRAME' for t in n.targets))
    defaults=ast.literal_eval(next(k.value for k in home.keywords if k.arg=='joint_pos'))
    # Python reads Unicode paths; supply mesh bytes through the virtual filesystem.
    assets={p.relative_to(asset.parent).as_posix():p.read_bytes() for p in (asset.parent/'assets').rglob('*') if p.is_file()}
    model=mujoco.MjModel.from_xml_string(asset.read_text(),assets);root=model.body('pelvis').id
    kin=FastTreeKinematics(model,device,root_body=root)
    q0=[]
    for name in kin.joint_names:
        matches=[float(v) for pattern,v in defaults.items() if re.fullmatch(pattern,name)]
        if len(matches)>1:raise ValueError('Ambiguous default-joint pattern')
        q0.append(matches[0] if matches else 0.)
    if len(q0)!=29:raise ValueError('Unexpected G1 joint order')
    # Only kinematics and state_features are used: no reward/action decoding/physics calls.
    task=object.__new__(Task);task.kin=kin;task.cfg=tc;task.q0=torch.tensor(q0,dtype=torch.float32,device=device)
    task.diagnostic_mj_model=model
    task.compile_sizes=set();task._compiled={};rotations.HEADING['mode']=tc.heading
    weights=ck['wm'];layers=sorted((int(k.split('.')[1]),v.shape) for k,v in weights.items() if re.fullmatch(r'net\.\d+\.weight',k))
    hidden=tuple(shape[0] for _,shape in layers[:-1]);feat_dim=layers[0][1][1]-29-33
    wm=networks.WorldModel(task,feat_dim,33,hidden=hidden,norm_eps=cfg.get('wm_norm_eps',1e-3),rot_out=cfg.get('wm_rot_out','axis')).to(device)
    wm.load_state_dict(weights,strict=True);wm.eval()
    for p in wm.parameters():p.requires_grad_(False)
    info={'torch':torch.__version__,'mujoco':mujoco.__version__,'mjlab':dist.version,
          'robot_xml_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'q0':q0,'joint_names':kin.joint_names,
          'features':feat_dim,'checkpoint_iteration':ck['it'],'backend':'eager fp32','device':device,'tf32':False,
          'source_sha256':{name:hashlib.sha256((Path(code)/'allmimic'/name).read_bytes()).hexdigest() for name in ['networks.py','task.py','kinematics.py','rotations.py']}}
    return wm,ck,info

def validate_pose_kinematics(wm,flat):
    import torch,mujoco,numpy as np
    from allmimic.task import flat_to_state
    s=flat_to_state(flat[:8]);positions,rotations,_,_=wm.task.body_kin(s)
    model=wm.task.diagnostic_mj_model;data=mujoco.MjData(model)
    bodies=[model.body(name).id for name in wm.task.kin.body_names]
    pos_error=0.;rot_error=0.
    for i,state in enumerate(flat[:8].detach().cpu().numpy()):
        data.qpos[:]=state[:36];mujoco.mj_forward(model,data)
        pos_error=max(pos_error,float(np.max(np.abs(positions[i].detach().cpu().numpy()-data.xpos[bodies]))))
        rot_error=max(rot_error,float(np.max(np.abs(rotations[i].detach().cpu().numpy()-data.xmat[bodies].reshape(-1,3,3)))))
    if pos_error>2e-5 or rot_error>2e-5:raise ValueError('Independent MuJoCo/FK pose check failed')
    return {'states':min(8,len(flat)),'position_max_abs_m':pos_error,'rotation_matrix_max_abs':rot_error}
