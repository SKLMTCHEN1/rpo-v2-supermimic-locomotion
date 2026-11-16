"""Evaluate a local policy in real MuJoCo-Warp physics; render its recorded states."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from supermimic.tasks.rpo import rpo_env_cfg
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_world_model, build_actor
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.representations import extract_actor_goal


def render(model, positions, output):
  import imageio.v2 as imageio
  from PIL import Image, ImageDraw, ImageFont
  model.vis.global_.offwidth=1280; model.vis.global_.offheight=720
  model.vis.headlight.ambient[:]=[.25,.25,.25]
  model.vis.headlight.diffuse[:]=[.4,.4,.4]
  model.vis.headlight.specular[:]=[.15,.15,.15]
  renderer=mujoco.Renderer(model,height=720,width=1280)
  data=mujoco.MjData(model)
  cam=mujoco.MjvCamera(); cam.distance=2.8;cam.azimuth=125;cam.elevation=-12
  opt=mujoco.MjvOption();opt.geomgroup[3]=0
  font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
  writer=imageio.get_writer(str(output),fps=50,codec='libx264',quality=8,macro_block_size=1)
  for i,qpos in enumerate(positions):
    data.qpos[:]=qpos;mujoco.mj_forward(model,data)
    cam.lookat[:]=[qpos[0],qpos[1],.65]
    renderer.update_scene(data,camera=cam,scene_option=opt)
    frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
    draw.rectangle((0,0,1280,92),fill=(15,22,32))
    draw.text((28,14),'RPO v2 | SuperMimic | Trained policy replay',font=font,fill=(235,242,250))
    draw.text((28,49),f'Physics: MuJoCo-Warp   |   t = {i*.02:.2f} s   |   No assistance / no resets',font=font,fill=(170,205,225))
    writer.append_data(np.asarray(frame))
    if i==min(120,len(positions)-1):frame.save(output.with_suffix('.png'))
  writer.close();renderer.close()


def main():
  p=argparse.ArgumentParser()
  p.add_argument('--run',type=Path)
  p.add_argument('--num-envs',type=int,default=8)
  p.add_argument('--video',action='store_true')
  p.add_argument('--output',type=Path,default=Path('outputs/rpo_v2_walk.mp4'))
  args=p.parse_args()
  torch.set_num_threads(8);torch.manual_seed(123)
  run=args.run or Path(Path('outputs/latest_run.txt').read_text().strip())
  cfg=load_runner_cfg(json.loads((run/'run_config.json').read_text()))
  ecfg=rpo_env_cfg(play=True);ecfg.seed=123
  env=create_env(ecfg,args.num_envs,setup_device('cuda:0'))
  wm=build_world_model(cfg,env,'cuda:0');rep=wm.representation
  actor=build_actor(cfg,wm,'cuda:0')
  actor.load_state_dict(torch.load(run/'actor.pt',map_location='cuda:0',weights_only=False));actor.eval()
  obs,_=env.reset();cmd=env.unwrapped.command_manager.get_term('motion')
  alive=torch.ones(args.num_envs,dtype=torch.bool,device='cuda:0')
  first_failure=torch.full((args.num_envs,),-1,device='cuda:0',dtype=torch.int64)
  positions=[];root_errors=[];heights=[]
  steps=cmd.motion.time_step_total-2
  with torch.inference_mode():
    for i in range(steps):
      states=obs['full_state']; aux=rep.extract_aux_state(states)
      sensor=rep.compute_actor_sensor(rep.represent(states),wm.sim_context.read(env.unwrapped),aux_state=aux,
        sim_actor_obs=obs['actor'] if cfg.actor_obs=='proprio' else None,noisy=False)
      td=build_actor_obs_td({'motion':extract_actor_goal(obs['goal'])},sensor.unsqueeze(1))
      action=actor(td)
      positions.append(env.unwrapped.sim.data.qpos[0].cpu().numpy().copy())
      root_errors.append(torch.linalg.vector_norm(states[:,:3]-cmd.motion.root_pos_w[cmd.time_steps],dim=-1).cpu().numpy())
      heights.append(states[:,2].cpu().numpy())
      obs,_,done,_=env.step(action)
      failed=env.unwrapped.termination_manager.terminated
      first_failure[failed & alive]=i+1
      alive &= ~failed
  qpos=np.asarray(positions)
  report={'run':str(run),'steps':steps,'duration_s':steps*.02,'trials':args.num_envs,
    'successful_trials':int(alive.sum()),'first_failure_steps':first_failure.cpu().tolist(),
    'root_position_rmse_m':float(np.sqrt(np.mean(np.asarray(root_errors)**2))),
    'root_height_min_m':float(np.min(heights)),
    'recorded_trial_displacement_xy_m':(qpos[-1,:2]-qpos[0,:2]).tolist(),
    'recorded_trial_success':bool(alive[0])}
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
  np.savez_compressed(args.output.with_suffix('.npz'),qpos=qpos,root_errors=np.asarray(root_errors))
  print(json.dumps(report,indent=2),flush=True)
  if args.video:
    if not alive[0]: raise RuntimeError('Recorded trial fell: refusing to label a reset sequence as successful walking.')
    render(env.unwrapped.sim.mj_model,qpos,args.output)
  env.close()

if __name__=='__main__':main()
