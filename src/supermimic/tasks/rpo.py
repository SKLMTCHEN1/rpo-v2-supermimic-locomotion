"""RoboParty RPO (Roboto Origin v2), 23 actuated joints.

Robot inertias, limits and collision geometry come from RoboParty's RPO URDF.
PD gains follow robolab/assets/robots/roboparty.py. The ankle joints use the
published independent-joint simulation approximation, not a linkage model.
"""
from pathlib import Path
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from .env_cfgs import gencoord_env_cfg

ROOT = Path(__file__).resolve().parents[3]
URDF = ROOT / "assets/rpo/urdf/rpo.urdf"
# Legacy RoboLab AMP files use Isaac's breadth-first joint order.
SOURCE_JOINT_NAMES = (
  "left_thigh_yaw_joint", "right_thigh_yaw_joint", "torso_joint",
  "left_thigh_roll_joint", "right_thigh_roll_joint",
  "left_arm_pitch_joint", "right_arm_pitch_joint",
  "left_thigh_pitch_joint", "right_thigh_pitch_joint",
  "left_arm_roll_joint", "right_arm_roll_joint",
  "left_knee_joint", "right_knee_joint", "left_arm_yaw_joint", "right_arm_yaw_joint",
  "left_ankle_pitch_joint", "right_ankle_pitch_joint",
  "left_elbow_pitch_joint", "right_elbow_pitch_joint",
  "left_ankle_roll_joint", "right_ankle_roll_joint",
  "left_elbow_yaw_joint", "right_elbow_yaw_joint",
)

def get_spec():
  spec = mujoco.MjSpec.from_file(str(URDF))
  spec.compiler.fusestatic = False
  spec.compiler.discardvisual = False
  base = spec.body("base_link")
  base.add_freejoint(name="floating_base_joint")
  base.add_site(name="imu", pos=[0, 0, 0], size=[0.01, 0, 0])
  spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  for body in spec.bodies:
    for i, geom in enumerate(body.geoms):
      collision = geom.contype != 0 or geom.conaffinity != 0
      geom.name = f"{body.name}_{'collision' if collision else 'visual'}_{i}"
      geom.group = 3 if collision else 2
      if collision:
        geom.friction = [0.8, 0.005, 0.0001]
        geom.condim = 3
  return spec


def get_robot_cfg():
  groups = [
    ((".*_thigh_.*_joint",), 100., 3.3, 120.),
    ((".*_knee_joint", "torso_joint"), 150., 5., 120.),
    ((".*_ankle_.*_joint", ".*_arm_.*_joint"), 40., 2., 27.),
    ((".*_elbow_pitch_joint",), 30., 1.5, 27.),
    ((".*_elbow_yaw_joint",), 20., 1., 27.),
  ]
  actuators = tuple(BuiltinPositionActuatorCfg(target_names_expr=names,
    stiffness=kp, damping=kd, effort_limit=effort, armature=0.01)
    for names, kp, kd, effort in groups)
  return EntityCfg(spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0, 0, 0.75), joint_pos={
      ".*_thigh_pitch_joint": -0.1, ".*_knee_joint": 0.3,
      ".*_ankle_pitch_joint": -0.2, ".*_arm_pitch_joint": 0.18,
      "left_arm_roll_joint": 0.06, "right_arm_roll_joint": -0.06,
      ".*_elbow_pitch_joint": 0.78,
    }, joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.9))


def rpo_env_cfg(play=False):
  cfg = gencoord_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_robot_cfg()}
  cfg.scene.sensors = ()
  cfg.actions["joint_pos"].scale = {".*": 0.5}
  motion = cfg.commands["motion"]
  motion.anchor_body_name = "base_link"
  motion.body_names = ("base_link", "left_thigh_roll_link", "left_knee_link",
    "left_ankle_roll_link", "right_thigh_roll_link", "right_knee_link",
    "right_ankle_roll_link", "torso_link", "left_arm_roll_link",
    "left_elbow_pitch_link", "left_elbow_yaw_link", "right_arm_roll_link",
    "right_elbow_pitch_link", "right_elbow_yaw_link")
  motion.motion_file = str(ROOT / "data/walk.npz")
  motion.debug_vis = False
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = r".*ankle_roll_link_collision.*"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  # Moderate randomization for the first single-motion policy.
  cfg.events["foot_friction"].params["ranges"] = (0.65, 0.95)
  cfg.events["base_com"].params["ranges"] = {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.005, 0.005)}
  cfg.events["encoder_bias"].params["bias_range"] = (-0.002, 0.002)
  if play:
    motion.joint_position_range = (0., 0.)
  cfg.viewer.body_name = "base_link"
  cfg.sim.nconmax = 64
  cfg.sim.njmax = 400
  return cfg
