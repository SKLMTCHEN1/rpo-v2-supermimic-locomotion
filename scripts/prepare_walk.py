"""Convert RoboLab's legacy RPO walk to named 50 Hz SuperMimic format."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
  'left_thigh_yaw_joint','right_thigh_yaw_joint','torso_joint',
  'left_thigh_roll_joint','right_thigh_roll_joint','left_arm_pitch_joint','right_arm_pitch_joint',
  'left_thigh_pitch_joint','right_thigh_pitch_joint','left_arm_roll_joint','right_arm_roll_joint',
  'left_knee_joint','right_knee_joint','left_arm_yaw_joint','right_arm_yaw_joint',
  'left_ankle_pitch_joint','right_ankle_pitch_joint','left_elbow_pitch_joint','right_elbow_pitch_joint',
  'left_ankle_roll_joint','right_ankle_roll_joint','left_elbow_yaw_joint','right_elbow_yaw_joint')

def main():
  p = argparse.ArgumentParser()
  p.add_argument('--source', type=Path, default=ROOT.parent/'ctc_rpo/roboparty_train/robolab/data/motions/rpo_amp/B14_-__Walk_turn_right_45_t2_stageii.pkl')
  p.add_argument('--output', type=Path, default=ROOT/'data/walk.npz')
  args=p.parse_args()
  d=pickle.load(open(args.source,'rb'))
  source_names=list(d.get('joint_names',SOURCE_NAMES))
  robot=ET.parse(ROOT/'assets/rpo/urdf/rpo.urdf').getroot()
  # URDF document order is this robot's depth-first MuJoCo order.
  names=[j.get('name') for j in robot.findall('joint') if j.get('type')!='fixed']
  fps=float(d['fps']); t=np.arange(len(d['root_pos']))/fps
  target=np.arange(int(t[-1]*50)+1)/50
  positions=np.array(d['dof_pos'])[:,[source_names.index(n) for n in names]]
  q=np.stack([np.interp(target,t,positions[:,i]) for i in range(len(names))],axis=1)
  root=np.stack([np.interp(target,t,d['root_pos'][:,i]) for i in range(3)],axis=1)
  quat=np.asarray(d['root_rot'])
  rotations=Slerp(t,Rotation.from_quat(quat[:,[1,2,3,0]]))(target)
  # Translate and rotate the world so the robot starts at (0,0), facing +X.
  yaw=rotations[0].as_euler('xyz')[2]
  align=Rotation.from_euler('z',-yaw)
  root[:,:2]-=root[0,:2]
  root=align.apply(root)
  rotations=align*rotations
  before=np.maximum(np.arange(len(target))-1,0); after=np.minimum(np.arange(len(target))+1,len(target)-1)
  omega=(rotations[after]*rotations[before].inv()).as_rotvec()/(target[after]-target[before])[:,None]
  out=dict(fps=np.array(50.),joint_names=np.array(names),joint_pos=q,joint_vel=np.gradient(q,.02,axis=0),
    root_pos_w=root,root_quat_w=rotations.as_quat()[:,[3,0,1,2]],
    root_lin_vel_w=np.gradient(root,.02,axis=0),root_ang_vel_w=omega)
  out={k:v.astype(np.float32) if isinstance(v,np.ndarray) and v.dtype.kind=='f' else v for k,v in out.items()}
  assert np.isfinite(np.concatenate([v.ravel() for v in out.values() if isinstance(v,np.ndarray) and v.dtype.kind=='f'])).all()
  args.output.parent.mkdir(parents=True,exist_ok=True)
  np.savez(args.output,**out)
  meta={'source':str(args.source),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
    'source_joint_order':source_names,'target_joint_order':names,'quaternion_order':'wxyz',
    'fps':50,'frames':len(target),'seconds':float(target[-1]),'displacement_m':root[-1].tolist()}
  args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2))
  print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
