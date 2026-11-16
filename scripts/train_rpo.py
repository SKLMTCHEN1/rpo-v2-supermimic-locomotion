"""Local, reproducible RPO-v2 SuperMimic training (no W&B account required)."""
import os
os.environ.setdefault('WANDB_MODE','offline')
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
from supermimic.tasks.rpo import rpo_env_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env, setup_device

class LocalRunner(SuperMimicRunner):
  def _init_wandb(self, log_dir, name_suffix=''):
    self.output_dir=log_dir
    Path('outputs').mkdir(exist_ok=True)
    (log_dir/'run_config.json').write_text(json.dumps({'runner_cfg':asdict(self.cfg)},indent=2))
    (Path('outputs')/'latest_run.txt').write_text(str(log_dir.resolve()))
    super()._init_wandb(log_dir,name_suffix)
    original_log=self.logger.log
    def log(*args, **kwargs):
      original_log(*args, **kwargs)
      row={'iteration':kwargs['it']*self.cfg.log_interval, 'env_steps':self.logger.tot_timesteps}
      row.update({k:float(v) for k,v in kwargs['loss_dict'].items()})
      if self.logger.lenbuffer:
        row['mean_episode_length']=sum(self.logger.lenbuffer)/len(self.logger.lenbuffer)
      with (log_dir/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    self.logger.log=log


def main():
  p=argparse.ArgumentParser()
  p.add_argument('--steps',type=int,default=6001)
  p.add_argument('--num-envs',type=int,default=1024)
  p.add_argument('--seed',type=int,default=42)
  p.add_argument('--resume',type=Path)
  p.add_argument('--name',default='rpo_v2_walk')
  args=p.parse_args()
  torch.set_num_threads(8)
  torch.manual_seed(args.seed)
  cfg=SuperMimicRunnerCfg(max_iterations=args.steps,buffer_size=256,
    dynamics_hidden_dims=(1024,1024,1024),actor_hidden_dims=(512,256,128),
    log_interval=100,save_interval=1000,run_name=args.name)
  cfg.wm.num_batches=4;cfg.wm.batch_size=1024;cfg.wm.horizon=8
  cfg.policy.batch_size=1024;cfg.policy.horizon=16
  env_cfg=rpo_env_cfg();env_cfg.seed=args.seed
  env_cfg.commands['motion'].sampling_mode='external'
  env=create_env(env_cfg,args.num_envs,setup_device('cuda:0'))
  runner=LocalRunner(env,cfg,'SuperMimic-GenCoord-RPO-v2',device='cuda:0')
  if args.resume:
    runner._wm.load_bundle(torch.load(args.resume/'world_model.pt',weights_only=False))
    runner.policy_learner.actor.load_state_dict(torch.load(args.resume/'actor.pt',weights_only=False))
    runner.policy_learner.critic.load_state_dict(torch.load(args.resume/'critic.pt',weights_only=False))
  runner.run()

if __name__=='__main__': main()
