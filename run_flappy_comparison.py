"""Sequential, reproducible DQN V12 / PPO V2 experiment and common evaluation.

Default: 8 training jobs, validation on a shared 50k grid, then locked tests.
Run --smoke-test --wandb-mode disabled before the full experiment.
Resume completed phases with --resume <suite directory>; interrupted training
is deliberately not resumed with a mismatched replay/normalization state.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
SOURCES = ['run_flappy_comparison.py', 'train_dqn.py',
           'train_ppo.py', 'vector_prioritized_double_dqn.py',
           'prioritized_double_dqn.py', 'double_dqn.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    payload = json.dumps(value, indent=2, ensure_ascii=False)
    # Windows readers/scanners can briefly deny replacement. Keep the last
    # committed JSON intact; never fall back to overwriting it in place.
    for attempt in range(8):
        try:
            temporary.write_text(payload, encoding='utf-8')
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.1 * 2**attempt, 1.0))


def verify_sources(suite, plan):
    # Explicit per-suite audit permits a storage-only repair without changing
    # historical configs or silently accepting modified training algorithms.
    audit_path = suite/'io_patch.json'
    audit = read(audit_path) if audit_path.exists() else {}
    for name, expected in plan['source_hashes'].items():
        actual = digest(ROOT/name)
        if actual == expected:
            continue
        if (name == 'run_flappy_comparison.py'
                and audit.get('original_sha256') == expected
                and audit.get('patched_sha256') == actual):
            continue
        raise ValueError(f'Source changed since suite creation: {name}')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(episodes):
    scores = sorted(e['score'] for e in episodes)
    return dict(mean_score=statistics.mean(scores), median_score=statistics.median(scores),
                score_std=statistics.pstdev(scores), min_score=min(scores), max_score=max(scores),
                worst_decile_mean=statistics.mean(scores[:max(1, math.ceil(len(scores)*.1))]),
                early_failure_rate=sum(s < 10 for s in scores)/len(scores),
                success_50_rate=sum(s >= 50 for s in scores)/len(scores),
                success_100_rate=sum(s >= 100 for s in scores)/len(scores),
                mean_reward=statistics.mean(e['reward'] for e in episodes),
                mean_episode_length=statistics.mean(e['steps'] for e in episodes),
                truncated_episodes=sum(e['truncated'] for e in episodes), episodes=len(episodes))


def make_plan(smoke=False):
    seeds = [100] if smoke else [100, 200, 300, 400]
    jobs = []
    # Alternate order between seed pairs to reduce consistent ordering effects.
    for index, seed in enumerate(seeds):
        for algo in (['dqn', 'ppo'] if index % 2 == 0 else ['ppo', 'dqn']):
            jobs.append(dict(algorithm=algo, seed=seed, name=f'{algo}_seed_{seed}',
                             wandb_id=uuid4().hex[:12]))
    plan = dict(smoke=smoke, total_timesteps=512 if smoke else 1500000,
                checkpoint_freq=256 if smoke else 50000, n_envs=8,
                train_seeds=seeds, validation_seeds=list(range(2000, 2002 if smoke else 2020)),
                test_seeds=list(range(3000, 3003 if smoke else 3100)),
                eval_max_steps=64 if smoke else 100000,
                torch_threads=1, jobs=jobs,
                selection='highest validation mean_score among periodic checkpoints; earliest tie',
                test_models=['best', 'final'], videos=False,
                source_hashes={name: digest(ROOT/name) for name in SOURCES})
    train = {seed+i for seed in seeds for i in range(plan['n_envs'])}
    assert not train.intersection(plan['validation_seeds'] + plan['test_seeds'])
    assert not set(plan['validation_seeds']).intersection(plan['test_seeds'])
    return plan


def model_args(module, plan, job):
    argv = [module.__name__, '--total-timesteps', str(plan['total_timesteps']),
            '--checkpoint-freq', str(plan['checkpoint_freq']), '--seed', str(job['seed']),
            '--n-envs', '8', '--device', 'cpu', '--wandb-mode', 'disabled']
    if plan['smoke']:
        argv += ['--batch-size', '32']
        argv += (['--n-steps', '32', '--n-epochs', '1'] if job['algorithm']=='ppo'
                 else ['--learning-starts', '32', '--buffer-size', '512'])
    original = sys.argv
    try:
        sys.argv = argv
        return module.parse_args()
    finally:
        sys.argv = original


def evaluate(model, seeds, factory, max_steps, destination):
    import numpy as np
    episodes = read(destination)['episodes'] if destination.exists() else []
    if [e['seed'] for e in episodes] != seeds[:len(episodes)]:
        raise ValueError('Saved evaluation has different seeds')
    env = factory('FlappyBird-v0', render_mode=None, max_episode_steps=max_steps)
    try:
        for seed in seeds[len(episodes):]:
            obs, _ = env.reset(seed=seed)
            total, steps = 0., 0
            terminated = truncated = False
            while not (terminated or truncated):
                if not np.isfinite(obs).all():
                    raise ValueError('Non-finite observation')
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                steps += 1
            episodes.append(dict(seed=seed, score=int(info['score']), reward=total,
                                 steps=steps, terminated=bool(terminated), truncated=bool(truncated)))
            save(destination, dict(complete=len(episodes)==len(seeds), episodes=episodes,
                                   metrics=summarize(episodes)))
            if len(episodes) % 10 == 0 or len(episodes)==len(seeds):
                print(f'  {destination.stem}: {len(episodes)}/{len(seeds)} episodes', flush=True)
    finally:
        env.close()
    return summarize(episodes)


def worker(suite, job_name, phase):
    import numpy as np
    import torch
    import wandb
    from importlib.metadata import version
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    import train_dqn as dqn
    import train_ppo as ppo
    from collections import deque

    plan = read(suite/'manifest.json')
    verify_sources(suite, plan)
    job = next(j for j in plan['jobs'] if j['name']==job_name)
    module = dqn if job['algorithm']=='dqn' else ppo
    model_class = dqn.DoubleDQN if job['algorithm']=='dqn' else ppo.PPO
    args = model_args(module, plan, job)
    folder = suite/job_name
    folder.mkdir(exist_ok=True)
    checkpoints = folder/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    torch.set_num_threads(plan['torch_threads'])
    config = dict(algorithm=job['algorithm'], training_seed=job['seed'],
                  initial_environment_seeds=list(range(job['seed'], job['seed']+8)),
                  total_timesteps=plan['total_timesteps'], checkpoint_freq=plan['checkpoint_freq'],
                  validation_seeds=plan['validation_seeds'], test_seeds=plan['test_seeds'],
                  eval_max_steps=plan['eval_max_steps'], torch_threads=plan['torch_threads'],
                  hyperparameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  packages={p:version(p) for p in ['torch','numpy','stable-baselines3','gymnasium',
                                                 'flappy-bird-gymnasium','wandb']},
                  observation_features=module.FEATURE_NAMES, normalization='rewards only',
                  source_hashes=plan['source_hashes'], selection=plan['selection'])
    if (folder/'config.json').exists():
        if read(folder/'config.json') != config:
            raise ValueError('Runtime/config changed; use a new suite')
    else:
        save(folder/'config.json', config)
    run = None
    if plan['wandb_mode'] != 'disabled':
        run = wandb.init(project=plan['wandb_project'], entity=plan['wandb_entity'],
                         group=suite.name, name=job_name, id=job['wandb_id'], resume='allow',
                         config=config, mode=plan['wandb_mode'], dir=str(folder),
                         tags=[job['algorithm'], 'smoke' if plan['smoke'] else 'comparison'],
                         settings=wandb.Settings(console='off'))
        for axis, prefix in [('train/env_steps','train/*'),
                             ('validation/checkpoint_step','validation/*')]:
            run.define_metric(axis)
            run.define_metric(prefix, step_metric=axis)
        print(f'W&B: {run.url}', flush=True)

    def log(values):
        if run is not None:
            run.log(values)

    class CommonTrainingMetrics(BaseCallback):
        def __init__(self):
            super().__init__()
            self.recent = deque(maxlen=100)
            self.next_log = plan['checkpoint_freq']
            self.elapsed = {}

        def _on_training_start(self):
            self.started = time.perf_counter()

        def _on_step(self):
            for info in self.locals['infos']:
                ep = info.get('episode')
                if ep:
                    self.recent.append(dict(score=int(ep['pipes_passed']), reward=float(ep['r']),
                                            steps=int(ep['l']), truncated=bool(info.get('TimeLimit.truncated',False))))
            if self.num_timesteps >= self.next_log:
                elapsed = time.perf_counter()-self.started
                self.elapsed[str(self.num_timesteps)] = elapsed
                values = {'train/env_steps':self.num_timesteps, 'train/elapsed_seconds':elapsed,
                          'train/env_steps_per_second':self.num_timesteps/max(elapsed,1e-9),
                          'train/learning_rate':float(self.model.lr_schedule(max(0.,1-self.num_timesteps/plan['total_timesteps'])))}
                if self.recent:
                    values.update({f'train/{k}':v for k,v in summarize(list(self.recent)).items()})
                log(values)
                print(f'{job_name}: {self.num_timesteps} steps, {elapsed/60:.1f} training minutes', flush=True)
                self.next_log += plan['checkpoint_freq']
            return True

    success = False
    try:
        if phase=='train_validate':
            if not (folder/'training.json').exists():
                if list(checkpoints.glob('*.zip')):
                    raise RuntimeError(f'Interrupted training in {folder}. Start a new suite; partial training is not silently restarted.')
                env = VecNormalize(make_vec_env(lambda: module.make_flappy_env('FlappyBird-v0'),
                    n_envs=8, seed=job['seed'], monitor_dir=str(folder/'monitor'),
                    monitor_kwargs=dict(info_keywords=('pipes_passed',))),
                    norm_obs=False, norm_reward=True, clip_reward=10., gamma=args.gamma)
                try:
                    model = module.create_model(args, env, folder/'tensorboard')
                    metrics = CommonTrainingMetrics()
                    callback = CallbackList([metrics, CheckpointCallback(
                        save_freq=plan['checkpoint_freq']//8, save_path=str(checkpoints),
                        name_prefix='model', save_vecnormalize=True)])
                    start = time.perf_counter()
                    model.learn(plan['total_timesteps'], callback=callback, progress_bar=False)
                    duration = time.perf_counter()-start
                    model.save(checkpoints/'final_model')
                    env.save(str(checkpoints/'final_vecnormalize.pkl'))
                    if job['algorithm']=='dqn':
                        model.save_replay_buffer(checkpoints/'final_replay_buffer.pkl')
                    save(folder/'training.json', dict(training_seconds=duration,
                         actual_timesteps=model.num_timesteps, checkpoint_training_seconds=metrics.elapsed))
                    del model
                finally:
                    env.close()
            training = read(folder/'training.json')
            validation = []
            for step in range(plan['checkpoint_freq'],plan['total_timesteps']+1,plan['checkpoint_freq']):
                path = checkpoints/f'model_{step}_steps.zip'
                model = model_class.load(path, device='cpu')
                result = evaluate(model, plan['validation_seeds'], module.make_flappy_env,
                                  plan['eval_max_steps'], folder/f'validation_{step}.json')
                validation.append(dict(step=step, checkpoint=path.name, sha256=digest(path), metrics=result))
                log({'validation/checkpoint_step':step, **{f'validation/{k}':v for k,v in result.items()}})
                del model
            best = max(validation,key=lambda r:r['metrics']['mean_score'])
            save(folder/'selection.json',dict(best=best, final=dict(checkpoint='final_model.zip',
                 sha256=digest(checkpoints/'final_model.zip')), validation=validation,
                 rule=plan['selection']))
            if run:
                run.summary.update({'timing/training_seconds':training['training_seconds'],
                    'timing/actual_timesteps':training['actual_timesteps'],
                    'selection/checkpoint_step':best['step'],
                    'selection/validation_mean_score':best['metrics']['mean_score']})
        else:
            lock = read(suite/'selection_lock.json')
            if digest(folder/'selection.json') != lock[job_name]:
                raise ValueError('Selection changed after lock')
            selected = read(folder/'selection.json')
            results = {}
            for kind in plan['test_models']:
                chosen = selected[kind]
                path = checkpoints/chosen['checkpoint']
                if digest(path)!=chosen['sha256']:
                    raise ValueError('Selected checkpoint changed')
                model = model_class.load(path, device='cpu')
                results[kind] = evaluate(model,plan['test_seeds'],module.make_flappy_env,
                                        plan['eval_max_steps'],folder/f'test_{kind}.json')
                values = {f'test_{kind}/{k}':v for k,v in results[kind].items()}
                log(values)
                if run:
                    run.summary.update(values)
                del model
            save(folder/'complete.json',results)
        success = True
    finally:
        if run:
            run.finish(exit_code=0 if success else 1)


def write_summary(suite, plan):
    lines = ['# DQN / PPO: Vergleich', '',
             'Best: ausschließlich durch Validierung ausgewählt. Test: getrennte Seeds.', '',
             '| Verfahren | Trainingsseed | Test Ø best | Test Median best | Unter 10 | Test Ø final | Training min |',
             '|---|---:|---:|---:|---:|---:|---:|']
    aggregate = {}
    for job in plan['jobs']:
        folder=suite/job['name']; result=read(folder/'complete.json');best=result['best']
        minutes=read(folder/'training.json')['training_seconds']/60
        lines.append(f"| {job['algorithm']} | {job['seed']} | {best['mean_score']:.2f} | {best['median_score']:.2f} | {best['early_failure_rate']:.1%} | {result['final']['mean_score']:.2f} | {minutes:.1f} |")
        aggregate.setdefault(job['algorithm'],[]).append(result)
    summary={}
    for algo,results in aggregate.items():
        summary[algo]={kind:{metric:dict(mean=statistics.mean(r[kind][metric] for r in results),
                            std_across_training_seeds=statistics.stdev(r[kind][metric] for r in results) if len(results)>1 else None)
                         for metric in ['mean_score','median_score','early_failure_rate','worst_decile_mean']}
                       for kind in plan['test_models']}
    lines += ['', 'Aggregierte Mittelwerte und Streuung über Trainingsseeds stehen in summary.json.',
              'Trainingszeit enthält Checkpoint-/Logging-Aufwand, aber keine Auswertung oder Videos.',
              'PPO kann wegen voller Rollouts das Sollbudget leicht überschreiten.',
              'Getrennte Seeds garantieren keine vollständig verschiedenen Röhrenfolgen.']
    (suite/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(suite/'summary.json',summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--plan-only',action='store_true',help='Print protocol, without files, W&B or training')
    parser.add_argument('--wandb-project',default='flappy-bird-dqn-ppo-vergleich')
    parser.add_argument('--wandb-entity',default=None)
    parser.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online')
    parser.add_argument('--worker',choices=['train_validate','test'],help=argparse.SUPPRESS)
    parser.add_argument('--job',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(args.resume.resolve(),args.job,args.worker)
        return
    if args.resume:
        suite=args.resume.resolve();plan=read(suite/'manifest.json')
        verify_sources(suite, plan)
    else:
        plan=make_plan(args.smoke_test)
        plan.update(wandb_project=args.wandb_project,wandb_entity=args.wandb_entity,wandb_mode=args.wandb_mode)
        if args.plan_only:
            print(json.dumps(plan,indent=2));return
        label='SMOKE' if args.smoke_test else 'COMPARISON'
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        suite=ROOT/'outputs'/f'{label}_DQN_PPO_{stamp}_{uuid4().hex[:8]}'
        suite.mkdir(parents=True,exist_ok=False)
        save(suite/'manifest.json',plan)
        snapshot=suite/'source';snapshot.mkdir()
        for name in SOURCES:
            (snapshot/name).write_bytes((ROOT/name).read_bytes())
    print(f'Experiment: {suite}',flush=True)
    print(f'Resume: python run_flappy_comparison.py --resume "{suite}"',flush=True)
    for phase in ['train_validate','test']:
        if phase=='test':
            lock={j['name']:digest(suite/j['name']/'selection.json') for j in plan['jobs']}
            if (suite/'selection_lock.json').exists() and read(suite/'selection_lock.json')!=lock:
                raise ValueError('Locked selections changed')
            save(suite/'selection_lock.json',lock)
        for job in plan['jobs']:
            marker='selection.json' if phase=='train_validate' else 'complete.json'
            if (suite/job['name']/marker).exists():
                continue
            print(f"{phase}: {job['name']}",flush=True)
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--resume',str(suite),
                            '--worker',phase,'--job',job['name']],cwd=ROOT,check=True)
    write_summary(suite,plan)
    print(f'Complete: {suite / "summary.md"}',flush=True)


if __name__=='__main__':
    main()
