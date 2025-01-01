#!/usr/bin/env python3
# actor.py
#
# Actor for a legacy pretrained model setup.
# Includes:
#  - APM limiting for "bot"/"model" players
#  - RollingRewardHackingMonitor for spam detection
#  - ToxicStrategyMonitor for cheese/worker harass
#  - partial_reward_sum for final ratio (train mode)
#
# No changes to the model input shapes -> preserves old agent.py compatibility.

import os
import time
import traceback
import uuid
import random
import json
import platform
from collections import defaultdict, deque

import torch
import torch.multiprocessing as mp

from distar.agent.import_helper import import_module
from distar.ctools.utils import read_config, deep_merge_dicts
from distar.ctools.utils.log_helper import TextLogger, VariableRecord
from distar.ctools.worker.actor.actor_comm import ActorComm
from distar.ctools.utils.dist_helper import dist_init
from distar.envs.env import SC2Env
from distar.ctools.worker.league.player import FRAC_ID

# ------------------------------------------------------------------------------
# Default config merged with user config
# ------------------------------------------------------------------------------
default_config = read_config(os.path.join(os.path.dirname(__file__), 'actor_default_config.yaml'))

class RollingRewardHackingMonitor:
    """
    Detects short-window action spam. Logs warnings when a certain threshold
    is exceeded within a rolling time window. Does not modify agent input shape.
    """
    def __init__(self, loop_threshold=10, window_seconds=3.0, warn_interval=10):
        self.loop_threshold = loop_threshold
        self.window_seconds = window_seconds
        self.warn_interval = warn_interval
        self.action_history = defaultdict(deque)
        self.steps_since_warn = 0

    def record_action(self, action_id):
        now = time.time()
        self.action_history[action_id].append(now)
        # Remove times older than window_seconds
        while self.action_history[action_id] and (now - self.action_history[action_id][0]) > self.window_seconds:
            self.action_history[action_id].popleft()

    def detect_spam_loops(self, logger=None):
        self.steps_since_warn += 1
        if self.steps_since_warn < self.warn_interval:
            return
        suspicious = []
        for act_id, times in self.action_history.items():
            if len(times) >= self.loop_threshold:
                suspicious.append((act_id, len(times)))
        if suspicious and logger:
            for (act_id, count) in suspicious:
                logger.info(f"[RollingRewardHackingMonitor] Potential spam: action={act_id} repeated {count}x in {self.window_seconds:.1f}s.")
        self.steps_since_warn = 0

class ToxicStrategyMonitor:
    """
    Monitors worker harass and early expansions for possible 'toxic' gameplay.
    Remains outside the agent's forward pass, preserving shape.
    """
    def __init__(self, early_game_cutoff=300, max_worker_harass=5, cheese_threshold=2):
        self.early_game_cutoff = early_game_cutoff
        self.max_worker_harass = max_worker_harass
        self.cheese_threshold = cheese_threshold

        self.worker_harass_count = 0
        self.early_expansion_or_proxy_count = 0
        self.toxic_strategic_events = 0

    def update_toxic_strategies(self, raw_ob, current_game_time, logger=None):
        self._check_worker_harass(raw_ob)
        self._check_early_cheese_expansions(raw_ob, current_game_time)
        # If counts exceed threshold, we log a potential toxic strategy
        if (self.worker_harass_count > self.max_worker_harass
            or self.early_expansion_or_proxy_count > self.cheese_threshold):
            self.toxic_strategic_events += 1
            if logger:
                logger.info(
                    f"[ToxicStrategyMonitor] Potential toxic pattern: harass={self.worker_harass_count}, expansions={self.early_expansion_or_proxy_count}"
                )

    def _check_worker_harass(self, raw_ob):
        # Example worker unit types: SCV=45, Drone=104, Probe=84
        for u in raw_ob.observation.raw_data.units:
            if u.alliance == 4 and u.unit_type in [45, 104, 84]:
                if u.health < 20 and u.weapon_cooldown > 0:
                    self.worker_harass_count += 1

    def _check_early_cheese_expansions(self, raw_ob, current_game_time):
        if current_game_time <= self.early_game_cutoff:
            expansions = 0
            for u in raw_ob.observation.raw_data.units:
                # For expansions: Hatch=86, Nexus=59, CommandCenter=18
                if (u.alliance == 1 and u.unit_type in [86, 59, 18] and u.build_progress < 1.0):
                    if unit.tag not in self.seen_expansion_tags:
                        self.seen_expansion_tags.add(unit.tag)
                        self.early_expansion_or_proxy_count += 1
            if expansions > 0:
                self.early_expansion_or_proxy_count += expansions

    def summarize_toxic_strategies(self):
        return {
            "worker_harass_count": self.worker_harass_count,
            "early_expansion_or_proxy_count": self.early_expansion_or_proxy_count,
            "toxic_strategic_events": self.toxic_strategic_events
        }

class Actor:
    """
    Preserves old shape usage, adding APM limiting, spam detection, and toxic
    strategy logs via separate code paths. This ensures pretrained agent or model
    input shapes remain untouched.
    """
    def __init__(self, cfg):
        cfg = deep_merge_dicts(default_config, cfg)
        self._whole_cfg = cfg
        self._cfg = cfg.actor
        self._job_type = self._cfg.job_type
        self._league_job_type = self._cfg.get('league_job_type','train')
        self._actor_uid = str(uuid.uuid1())
        self._gpu_batch_inference = self._cfg.get('gpu_batch_inference', False)

        # Logging
        self._logger = TextLogger(
            path=os.path.join(
                os.getcwd(),
                'experiments',
                self._whole_cfg.common.experiment_name,
                'actor_log'
            ),
            name=self._actor_uid
        )

        # Actor comm for training
        if self._job_type == 'train':
            self._comm = ActorComm(self._whole_cfg, self._actor_uid, self._logger)
            interval = self._whole_cfg.communication.actor_ask_for_job_interval
            self.max_job_duration = interval * random.uniform(0.7, 1.3)

        # APM / spam / toxic side-channels
        self._bot_action_timestamps = {}
        self._reward_hacking_monitor = RollingRewardHackingMonitor(
            loop_threshold=self._cfg.get('loop_threshold', 10),
            window_seconds=self._cfg.get('spam_window_seconds', 3.0),
            warn_interval=self._cfg.get('warn_interval', 10)
        )
        self._toxic_strategy_monitor = ToxicStrategyMonitor()

        self._setup_agents()

    def _setup_agents(self):
        self.agents = []
        if self._job_type == 'train':
            self._comm.ask_for_job(self)
        else:
            self.models = {}
            map_names = []
            for idx, player_id in enumerate(self._cfg.player_ids):
                if 'bot' in player_id:
                    continue
                # Import the agent code you want
                AgentClass = import_module(self._cfg.agents.get(player_id, 'default'), 'Agent')
                agent = AgentClass(self._whole_cfg)
                agent.player_id = player_id
                agent.side_id = idx
                self.agents.append(agent)

                # If this agent has a model, we load it from old config
                if agent.HAS_MODEL:
                    if player_id not in self.models:
                        if self._cfg.use_cuda:
                            agent.model = agent.model.cuda()
                        else:
                            agent.model = agent.model.eval().share_memory()
                        if not self._cfg.fake_model:
                            loaded_ckpt = torch.load(self._cfg.model_paths[player_id], map_location='cpu')
                            if 'map_name' in loaded_ckpt:
                                map_names.append(loaded_ckpt['map_name'])
                                agent._fake_reward_prob = loaded_ckpt['fake_reward_prob']
                                agent._z_path = loaded_ckpt['z_path']
                                agent.z_idx = loaded_ckpt['z_idx']
                            net_state = {
                                k: v
                                for k, v in loaded_ckpt['model'].items()
                                if 'value_networks' not in k
                            }
                            agent.model.load_state_dict(net_state, strict=False)
                        self.models[player_id] = agent.model
                    else:
                        agent.model = self.models[player_id]

            # If we discovered exactly one map_name in the model ckpt, set env
            if len(map_names) == 1:
                self._whole_cfg.env.map_name = map_names[0]
            elif len(map_names) == 2:
                # If not random => pick a specific map
                if not (map_names[0] == 'random' and map_names[1] == 'random'):
                    self._whole_cfg.env.map_name = 'NewRepugnancy'

    def _inference_loop(self, env_id=0, job=None, result_queue=None, pipe_c=None):
        if job is None:
            job = {}
        torch.set_num_threads(1)

        # Possibly override environment races
        frac_ids = job.get('frac_ids', [])
        env_info = job.get('env_info', {})
        chosen_races = []
        for fid in frac_ids:
            chosen_races.append(random.choice(FRAC_ID[fid]))
        if chosen_races:
            env_info['races'] = chosen_races

        merged_cfg = deep_merge_dicts(self._whole_cfg, {'env': env_info})
        self._env = SC2Env(merged_cfg)

        iter_count = 0
        if env_id == 0:
            variable_record = VariableRecord(self._cfg.print_freq)
            variable_record.register_var('agent_time')
            variable_record.register_var('agent_time_per_agent')
            variable_record.register_var('env_time')
            if 'train' in self._job_type:
                variable_record.register_var('post_process_time')
                variable_record.register_var('post_process_per_agent')
                variable_record.register_var('send_data_time')
                variable_record.register_var('send_data_per_agent')
                variable_record.register_var('update_model_time')

        # Set up APM-limiting
        bot_target_apm = self._cfg.get('bot_target_apm', 900)
        action_cooldown = 60.0 / bot_target_apm
        last_bot_action_time = {}
        NO_OP_ACTION = [{
            'func_id': 0,
            'queued': 0,
            'skip_steps': 0,
            'unit_tags': [],
            'target_unit_tag': 0,
            'location': (0, 0)
        }]

        episode_count = 0
        with torch.no_grad():
            while episode_count < self._cfg.episode_num:
                try:
                    game_start = time.time()
                    game_iters = 0
                    observations, game_info, map_name = self._env.reset()

                    # Initialize partial reward sums, agent resets
                    for idx in observations:
                        ag = self.agents[idx]
                        ag.env_id = env_id
                        race_str = self._whole_cfg.env.races[idx]
                        ag.reset(map_name, race_str, game_info[idx], observations[idx])
                        setattr(ag, "partial_reward_sum", 0.0)

                        pid = ag.player_id
                        if ('bot' in pid) or ('model' in pid):
                            last_bot_action_time[pid] = 0.0
                            self._bot_action_timestamps[pid] = deque()

                    while True:
                        if pipe_c and pipe_c.poll():
                            cmd = pipe_c.recv()
                            if cmd == 'reset':
                                break
                            elif cmd == 'close':
                                self._env.close()
                                return

                        agent_start = time.time()
                        agent_count = 0
                        actions = {}
                        players_obs = observations

                        for player_idx, obs_data in players_obs.items():
                            ag = self.agents[player_idx]
                            pid = ag.player_id
                            if self._job_type == 'train':
                                ag._model_last_iter = self._comm.model_last_iter_dict[pid].item()

                            # APM limit for bots/models
                            if ('bot' in pid) or ('model' in pid):
                                now_t = time.time()
                                if (now_t - last_bot_action_time[pid]) < action_cooldown:
                                    actions[player_idx] = NO_OP_ACTION
                                else:
                                    real_act = ag.step(obs_data)
                                    actions[player_idx] = real_act
                                    last_bot_action_time[pid] = now_t
                                    self._bot_action_timestamps[pid].append(now_t)
                                    while (self._bot_action_timestamps[pid]
                                           and (now_t - self._bot_action_timestamps[pid][0]) > 60):
                                        self._bot_action_timestamps[pid].popleft()
                                    apm_now = len(self._bot_action_timestamps[pid])
                                    self._logger.info(f"[APM] Player {pid}: {apm_now} in last 60s")
                                    # Record spam action
                                    if isinstance(real_act, list):
                                        for subact in real_act:
                                            if 'func_id' in subact:
                                                self._reward_hacking_monitor.record_action(subact['func_id'])
                            else:
                                actions[player_idx] = ag.step(obs_data)

                            agent_count += 1

                        # Time the agent step
                        agent_time = time.time() - agent_start

                        # Environment step
                        env_start = time.time()
                        next_obs, reward, done = self._env.step(actions)
                        env_time = time.time() - env_start

                        # Spam detection
                        self._reward_hacking_monitor.detect_spam_loops(logger=self._logger)

                        # Toxic detection
                        for p_idx, nxt_data in next_obs.items():
                            if 'raw_obs' in nxt_data:
                                gl_val = nxt_data['raw_obs'].observation.game_loop
                                current_time_s = gl_val / 22.4
                                self._toxic_strategy_monitor.update_toxic_strategies(nxt_data['raw_obs'], current_time_s, logger=self._logger)

                        # If training, gather data or send
                        if 'train' in self._job_type:
                            post_t = 0
                            post_c = 0
                            send_t = 0
                            send_c = 0
                            for p_idx, nxt_data in next_obs.items():
                                store_data = (
                                    self._job_type == 'train_test'
                                    or self.agents[p_idx].player_id in self._comm.job['send_data_players']
                                )
                                self.agents[p_idx].partial_reward_sum += reward[p_idx]
                                if store_data:
                                    t0 = time.time()
                                    trajectory = self.agents[p_idx].collect_data(nxt_data, reward[p_idx], done, p_idx)
                                    post_t += (time.time() - t0)
                                    post_c += 1
                                    if trajectory is not None and self._job_type == 'train':
                                        t1 = time.time()
                                        self._comm.send_data(trajectory, self.agents[p_idx].player_id)
                                        send_t += (time.time() - t1)
                                        send_c += 1
                                else:
                                    self.agents[p_idx].update_fake_reward(nxt_data)

                        iter_count += 1
                        game_iters += 1

                        if env_id == 0:
                            # Logging performance
                            if 'train' in self._job_type:
                                rec_obj = {
                                    'agent_time': agent_time,
                                    'agent_time_per_agent': agent_time / (agent_count + 1e-6),
                                    'env_time': env_time
                                }
                                variable_record.update_var(rec_obj)
                                if post_c > 0:
                                    variable_record.update_var({
                                        'post_process_time': post_t,
                                        'post_process_per_agent': post_t / post_c
                                    })
                                if send_c > 0:
                                    variable_record.update_var({
                                        'send_data_time': send_t,
                                        'send_data_per_agent': send_t / send_c
                                    })
                            else:
                                variable_record.update_var({
                                    'agent_time': agent_time,
                                    'agent_time_per_agent': agent_time / (agent_count + 1e-6),
                                    'env_time': env_time
                                })
                            self.iter_after_hook(iter_count, variable_record)

                        if not done:
                            observations = next_obs
                            continue

                        # If done => finalize training stats
                        if self._job_type == 'train':
                            # pick random player's obs for final info
                            rand_pid = random.sample(observations.keys(), 1)[0]
                            game_steps_done = observations[rand_pid]['raw_obs'].observation.game_loop
                            results = defaultdict(dict)
                            for a_idx in range(len(self.agents)):
                                pid2 = self.agents[a_idx].player_id
                                s2 = self.agents[a_idx].side_id
                                r2 = self.agents[a_idx].race
                                it2 = self.agents[a_idx].iter_count
                                final_r = reward[a_idx]
                                partial_s = getattr(self.agents[a_idx], "partial_reward_sum", 0.0)
                                ratio_s = None
                                if abs(final_r) > 1e-6:
                                    ratio_s = partial_s / abs(final_r)

                                results[s2]['race'] = r2
                                results[s2]['player_id'] = pid2
                                results[s2]['opponent_id'] = self.agents[a_idx].opponent_id
                                results[s2]['winloss'] = final_r
                                results[s2]['agent_iters'] = it2
                                results[s2]['partial_reward_sum'] = partial_s
                                if ratio_s is not None:
                                    results[s2]['partial_reward_ratio'] = ratio_s
                                results[s2].update(self.agents[a_idx].get_unit_num_info())
                                results[s2].update(self.agents[a_idx].get_stat_data())

                            duration_s = time.time() - game_start
                            results['game_steps'] = game_steps_done
                            results['game_iters'] = game_iters
                            results['game_duration'] = duration_s
                            # Add toxic summary
                            tox_sum = self._toxic_strategy_monitor.summarize_toxic_strategies()
                            results.update(tox_sum)

                            # Send final results
                            self._comm.send_result(results)

                        break

                    episode_count += 1

                except Exception as e:
                    print('[EPISODE LOOP ERROR]', e, flush=True)
                    print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
                    episode_count += 1
                    self._env.close()

            self._env.close()
            if result_queue:
                print(os.getpid(), 'done')
                result_queue.put('done')
                # Keep process alive if needed
                time.sleep(9999999)
            else:
                return

    def _gpu_inference_loop(self):
        # If you rely on GPU batch mode
        _, _ = dist_init(method='single_node')
        torch.set_num_threads(1)
        for ag in self.agents:
            ag.model = ag.model.cuda()
            if 'train' in self._job_type:
                ag.teacher_model = ag.teacher_model.cuda()

        start_time = time.time()
        done_count = 0
        with torch.no_grad():
            while True:
                if self._job_type == 'train':
                    self._comm.async_update_model(self)
                    if time.time() - start_time > self.max_job_duration:
                        self.close()
                    if self._result_queue.qsize():
                        self._result_queue.get()
                        done_count += 1
                        if done_count == len(self._processes):
                            self.close()
                            break
                elif self._job_type == 'eval':
                    if self._result_queue.qsize():
                        self._result_queue.get()
                        done_count += 1
                        if done_count == len(self._processes):
                            self._close_processes()
                            break

                # GPU-batch inference call for each agent
                for ag in self.agents:
                    ag.gpu_batch_inference()
                    if 'train' in self._job_type:
                        ag.gpu_batch_inference(teacher=True)

    def _start_multi_inference_loop(self):
        self._close_processes()
        self._processes = []
        job = self._comm.job if hasattr(self, '_comm') else {}
        self.pipes = []
        ctx_str = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        mp_ctx = mp.get_context(ctx_str)
        self._result_queue = mp_ctx.Queue()
        for env_id in range(self._cfg.env_num):
            pipe_p, pipe_c = mp_ctx.Pipe()
            p = mp_ctx.Process(
                target=self._inference_loop,
                args=(env_id, job, self._result_queue, pipe_c),
                daemon=True
            )
            self.pipes.append(pipe_p)
            self._processes.append(p)
            p.start()

    def reset_env(self):
        for p in self.pipes:
            p.send('reset')

    def run(self):
        try:
            if 'test' in self._job_type:
                # single environment test
                self._inference_loop()
            else:
                if self._job_type == 'train':
                    self._start_multi_inference_loop()
                    if self._gpu_batch_inference:
                        self._gpu_inference_loop()
                    else:
                        st = time.time()
                        while True:
                            if time.time() - st > self.max_job_duration:
                                self.reset()
                            self._comm.update_model(self)
                            time.sleep(1)
                if self._job_type == 'eval':
                    self._start_multi_inference_loop()
                    if self._gpu_batch_inference:
                        self._gpu_inference_loop()
                    else:
                        for _ in range(len(self._processes)):
                            self._result_queue.get()
                        self._close_processes()

        except Exception as e:
            print('[MAIN LOOP ERROR]', e, flush=True)
            print(''.join(traceback.format_tb(e.__traceback__)), flush=True)

    def reset(self):
        self._logger.info('Actor reset multi-process.')
        self._close_processes()
        if hasattr(self, '_comm'):
            self._comm.ask_for_job(self)
        self._start_multi_inference_loop()

    def close(self):
        self._logger.info('Actor close.')
        time.sleep(2)
        if hasattr(self, '_comm'):
            self._comm.close()
        self._close_processes()
        time.sleep(1)
        os._exit(0)

    def _close_processes(self):
        if hasattr(self, '_processes'):
            for pipe_p in self.pipes:
                pipe_p.send('close')
            for proc in self._processes:
                proc.join()

    def iter_after_hook(self, iter_count, variable_record):
        if iter_count % self._cfg.print_freq == 0:
            if hasattr(self,'_comm'):
                variable_record.update_var({
                    'update_model_time': self._comm._avg_update_model_time.item()
                })
            self._logger.info(
                'ACTOR({}):\n{}TimeStep{}{} {}'.format(
                    self._actor_uid,
                    '=' * 35,
                    iter_count,
                    '=' * 35,
                    variable_record.get_vars_text()
                )
            )

    # Optionally, minimal parse_logs if your code outside calls it
    def parse_logs(self, log_file):
        if not os.path.exists(log_file):
            return [], []
        with open(log_file, 'r') as f:
            lines = f.readlines()
        spam_logs = [ln for ln in lines if 'RollingRewardHackingMonitor' in ln]
        toxic_logs = [ln for ln in lines if 'ToxicStrategyMonitor' in ln]
        return spam_logs, toxic_logs

    def summarize_results(self, result_file):
        if not os.path.isfile(result_file):
            self._logger.info(f"No such result file: {result_file}")
            return
        with open(result_file, 'r') as f:
            data = json.load(f)
        self._logger.info(f"Results => {list(data.keys())}")


if __name__ == '__main__':
    actor = Actor(cfg={})
    actor.run()