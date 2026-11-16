"""Catch joint-order, quaternion and robot asset incompatibilities before training."""
import mujoco
import numpy as np
from mjlab.entity import Entity
from supermimic.tasks.rpo import ROOT, get_robot_cfg


def test_motion_matches_actuated_robot():
  robot=Entity(get_robot_cfg())
  model=robot.spec.compile()
  motion=np.load(ROOT/'data/walk.npz')
  assert motion['joint_names'].tolist()==list(robot.joint_names)
  assert model.nu==23 and model.nq==30 and model.nv==29
  np.testing.assert_allclose(np.linalg.norm(motion['root_quat_w'],axis=1),1,atol=1e-6)
  q=motion['joint_pos']
  assert np.all(q>=model.jnt_range[1:,0]-.01)
  assert np.all(q<=model.jnt_range[1:,1]+.01)
  assert set(model.actuator_trnid[:,0])==set(range(1,24))
  assert np.all(model.actuator_forcelimited)
  assert any(model.geom_group==2) and any(model.geom_group==3)


def test_reference_feet_and_velocity():
  model=Entity(get_robot_cfg()).spec.compile()
  data=mujoco.MjData(model)
  motion=np.load(ROOT/'data/walk.npz')
  np.testing.assert_allclose(motion['root_lin_vel_w'],np.gradient(motion['root_pos_w'],.02,axis=0),atol=3e-5)
  for i in range(0,len(motion['joint_pos']),20):
    data.qpos[:3]=motion['root_pos_w'][i]
    data.qpos[3:7]=motion['root_quat_w'][i]
    data.qpos[7:]=motion['joint_pos'][i]
    mujoco.mj_forward(model,data)
    z=[data.body(n).xpos[2] for n in ['left_ankle_roll_link','right_ankle_roll_link']]
    assert -.03<min(z)<.12, (i,z)
