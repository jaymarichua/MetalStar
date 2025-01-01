#!/usr/bin/env python3
# coding: utf-8
# =============================================================================
# actor.py
#
# This module merges the old actor flow with new features:
#  - RollingRewardHackingMonitor
#  - ToxicStrategyMonitor
#  - partial reward ratio & APM limiting
#  - parse_logs method restored so it doesn't crash in play.py
# =============================================================================

import os
import time
import traceback
import uuid
import random
import json
import platform
import collections
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

# =============================================================================
# Default config
# =============================================================================
default_config = read_config(os.path.join(os.path.dirname(__file__), 'actor_default_config.yaml'))


class RollingRewardHackingMonitor:
    """
    Monitors short-term usage of specific actions, logs warnings if spammy.
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
        while self.action_history[action_id] and (now - self.action_history[action_id][0]) > self.window_seconds:
            self.action_history[action_id].popleft()

    def detect_spam_loops(self, logger=None):
        self.steps_since_warn += 1
        if self.steps_since_warn < self.warn_interval:
            return

        suspicious_actions = []
        for act_id, timestamps in self.action_history.items():
            if len(timestamps) >= self.loop_threshold:
                suspicious_actions.append((act_id, len(timestamps)))

        if suspicious_actions and logger:
            for act_id, count in suspicious_actions:
                logger.info(
                    f"[RollingRewardHackingMonitor] Potential spam: Action '{act_id}' repeated "
                    f"{count} times in {self.window_seconds:.1f}s."
                )
        self.steps_since_warn = 0


class ToxicStrategyMonitor:
    """
    Tracks 'toxic' or hyper-aggressive strategies (e.g., worker harass, cheese expansions).
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

        if (self.worker_harass_count > self.max_worker_harass
                or self.early_expansion_or_proxy_count > self.cheese_threshold):
            self.toxic_strategic_events += 1
            if logger:
                logger.info(
                    f"[ToxicStrategyMonitor] Potential toxic strategy event: "
                    f"harass={self.worker_harass_count}, cheese={self.early_expansion_or_proxy_count}"
                )

    def _check_worker_harass(self, raw_ob):
        for u in raw_ob.observation.raw_data.units:
            # Example worker IDs: SCV=45, Drone=104, Probe=84
            if u.alliance == 4 and u.unit_type in [45, 104, 84]:
                if u.health < 10 and u.weapon_cooldown > 0:
                    self.worker_harass_count += 1

    def _check_early_cheese_expansions(self, raw_ob, current_game_time):
        if current_game_time <= self.early_game_cutoff:
            expansions_built = 0
            for unit in raw_ob.observation.raw_data.units:
                # expansions: Hatch=86, Nexus=59, CC=18
                if (unit.alliance == 1 and
                        unit.unit_type in [86, 59, 18] and
                        unit.build_progress < 1.0):
                    expansions_built += 1
            if expansions_built > 0:
                self.early_expansion_or_proxy_count += expansions_built

    def summarize_toxic_strategies(self):
        return {
            "worker_harass_count": self.worker_harass_count,
            "early_expansion_or_proxy_count": self.early_expansion_or_proxy_count,
            "toxic_strategic_events": self.toxic_strategic_events
        }


class Actor:
    """
    Combined actor code that includes:
     - RollingRewardHackingMonitor
     - ToxicStrategyMonitor
     - partial reward ratio
     - multi-env processes
     - parse_logs method to avoid missing method errors
    """

    def __init__(self, cfg):
        cfg = deep_merge_dicts(default_config, cfg)
        self._whole_cfg = cfg
        self._cfg = cfg.actor
        self._job_type = self._cfg.job_type
        self._league_job_type = self._cfg.get('league_job_type', 'train')
        self._actor_uid = str(uuid.uuid1())
        self._gpu_batch_inference = self._cfg.get('gpu_batch_inference', False)

        # Logger
        self._logger = TextLogger(
            path=os.path.join(
                os.getcwd(),
                self._whole_cfg.common.experiment_name,
                'actor_log'
            ),
            name=self._actor_uid
        )

        # Comm if train
        if self._job_type == 'train':
            self._comm = ActorComm(self._whole_cfg, self._actor_uid, self._logger)
            interval = self._whole_cfg.communication.actor_ask_for_job_interval
            self.max_job_duration = interval * random.uniform(0.7, 1.3)

        # APM tracking
        self._bot_action_timestamps = {}

        # Rolling spam check
        self._reward_hacking_monitor = RollingRewardHackingMonitor(
            loop_threshold=self._cfg.get('loop_threshold', 10),
            window_seconds=self._cfg.get('spam_window_seconds', 3.0),
            warn_interval=self._cfg.get('warn_interval', 10)
        )

        # Toxic strategies
        self._toxic_monitor = ToxicStrategyMonitor()

        # Setup agents
        self._setup_agents()

    def _setup_agents(self):
        """
        If train -> ask comm, else load model states if needed.
        """
        self.agents = []
        if self._job_type == 'train':
            self._comm.ask_for_job(self)
        else:
            self.models = {}
            map_names = []
            for idx, player_id in enumerate(self._cfg.player_ids):
                if 'bot' in player_id:
                    continue
                AgentClass = import_module(self._cfg.agents.get(player_id, 'default'), 'Agent')
                agent = AgentClass(self._whole_cfg)
                agent.player_id = player_id
                agent.side_id = idx
                self.agents.append(agent)

                if agent.HAS_MODEL:
                    if player_id not in self.models:
                        if self._cfg.use_cuda:
                            agent.model = agent.model.cuda()
                        else:
                            agent.model = agent.model.eval().share_memory()

                        if not self._cfg.fake_model:
                            loaded_state = torch.load(self._cfg.model_paths[player_id], map_location='cpu')
                            if 'map_name' in loaded_state:
                                map_names.append(loaded_state['map_name'])
                                agent._fake_reward_prob = loaded_state['fake_reward_prob']
                                agent._z_path = loaded_state['z_path']
                                agent.z_idx = loaded_state['z_idx']
                            net_state = {
                                k: v for k, v in loaded_state['model'].items()
                                if 'value_networks' not in k
                            }
                            agent.model.load_state_dict(net_state, strict=False)
                        self.models[player_id] = agent.model
                    else:
                        agent.model = self.models[player_id]

            if len(map_names) == 1:
                self._whole_cfg.env.map_name = map_names[0]
            elif len(map_names) == 2:
                if not (map_names[0] == 'random' and map_names[1] == 'random'):
                    self._whole_cfg.env.map_name = 'NewRepugnancy'

    def _inference_loop(self, env_id=0, job=None, result_queue=None, pipe_c=None):
        """
        Our main environment loop with partial ratio, toxic strategy, spam, APM, etc.
        """
        if job is None:
            job = {}
        torch.set_num_threads(1)

        frac_ids = job.get('frac_ids', [])
        env_info = job.get('env_info', {})
        chosen_races = []
        for frac_id in frac_ids:
            chosen_races.append(random.choice(FRAC_ID[frac_id]))
        if chosen_races:
            env_info['races'] = chosen_races

        merged_cfg = deep_merge_dicts(self._whole_cfg, {'env': env_info})
        self._env = SC2Env(merged_cfg)

        iter_count = 0
        # For logging
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

                    for idx in observations.keys():
                        self.agents[idx].env_id = env_id
                        race_str = self._whole_cfg.env.races[idx]
                        self.agents[idx].reset(map_name, race_str, game_info[idx], observations[idx])
                        setattr(self.agents[idx], "partial_reward_sum", 0.0)

                        pid = self.agents[idx].player_id
                        if 'bot' in pid or 'model' in pid:
                            last_bot_action_time[pid] = 0.0
                            self._bot_action_timestamps[pid] = deque()

                    while True:
                        # Check for pipe commands (reset, close)
                        if pipe_c is not None and pipe_c.poll():
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

                        for player_index, obs_data in players_obs.items():
                            agent = self.agents[player_index]
                            pid = agent.player_id

                            if self._job_type == 'train':
                                agent._model_last_iter = self._comm.model_last_iter_dict[pid].item()

                            # APM limiting if 'bot' or 'model'
                            if 'bot' in pid or 'model' in pid:
                                now_time = time.time()
                                if (now_time - last_bot_action_time[pid]) < action_cooldown:
                                    actions[player_index] = NO_OP_ACTION
                                else:
                                    real_action = agent.step(obs_data)
                                    actions[player_index] = real_action
                                    last_bot_action_time[pid] = now_time
                                    self._bot_action_timestamps[pid].append(now_time)
                                    # remove old timestamps over 60s
                                    while (self._bot_action_timestamps[pid] and
                                           (now_time - self._bot_action_timestamps[pid][0]) > 60):
                                        self._bot_action_timestamps[pid].popleft()
                                    apm_now = len(self._bot_action_timestamps[pid])
                                    self._logger.info(f"[APM] Player {pid}: {apm_now} (last 60s)")

                                    # record spam
                                    if isinstance(real_action, list):
                                        for a_dict in real_action:
                                            if 'func_id' in a_dict:
                                                self._reward_hacking_monitor.record_action(a_dict['func_id'])
                            else:
                                actions[player_index] = agent.step(obs_data)
                            agent_count += 1

                        agent_time = time.time() - agent_start

                        env_start = time.time()
                        next_obs, reward, done = self._env.step(actions)
                        env_time = time.time() - env_start

                        # spam detection
                        self._reward_hacking_monitor.detect_spam_loops(logger=self._logger)

                        # toxic detection
                        for p_idx, next_data in next_obs.items():
                            gl_val = next_data['raw_obs'].observation.game_loop
                            current_game_time = gl_val / 22.4
                            self._toxic_monitor.update_toxic_strategies(next_data['raw_obs'], current_game_time, logger=self._logger)

                        # if train -> gather data
                        if 'train' in self._job_type:
                            post_process_time = 0
                            post_process_count = 0
                            send_data_time = 0
                            send_data_count = 0

                            for p_idx, next_data in next_obs.items():
                                store_data = (
                                    self._job_type == 'train_test'
                                    or (self.agents[p_idx].player_id in self._comm.job['send_data_players'])
                                )
                                # partial reward accumulation
                                self.agents[p_idx].partial_reward_sum += reward[p_idx]

                                if store_data:
                                    t0 = time.time()
                                    traj_data = self.agents[p_idx].collect_data(
                                        next_obs[p_idx], reward[p_idx], done, p_idx
                                    )
                                    post_process_time += (time.time() - t0)
                                    post_process_count += 1

                                    if traj_data is not None and self._job_type == 'train':
                                        t1 = time.time()
                                        self._comm.send_data(traj_data, self.agents[p_idx].player_id)
                                        send_data_time += (time.time() - t1)
                                        send_data_count += 1
                                else:
                                    self.agents[p_idx].update_fake_reward(next_obs[p_idx])

                        iter_count += 1
                        game_iters += 1

                        if env_id == 0:
                            if 'train' in self._job_type:
                                # update variable record
                                variable_record.update_var({
                                    'agent_time': agent_time,
                                    'agent_time_per_agent': agent_time / (agent_count + 1e-6),
                                    'env_time': env_time,
                                })
                                if post_process_count > 0:
                                    variable_record.update_var({
                                        'post_process_time': post_process_time,
                                        'post_process_per_agent': post_process_time / post_process_count
                                    })
                                if send_data_count > 0:
                                    variable_record.update_var({
                                        'send_data_time': send_data_time,
                                        'send_data_per_agent': send_data_time / send_data_count
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

                        # If done
                        if self._job_type == 'train':
                            rand_pid = random.sample(observations.keys(), 1)[0]
                            final_game_steps = observations[rand_pid]['raw_obs'].observation.game_loop
                            result_info = defaultdict(dict)
                            for idx2 in range(len(self.agents)):
                                pid2 = self.agents[idx2].player_id
                                side_id2 = self.agents[idx2].side_id
                                race2 = self.agents[idx2].race
                                agent_iters2 = self.agents[idx2].iter_count
                                final_reward2 = reward[idx2]

                                partial_sum = getattr(self.agents[idx2], "partial_reward_sum", 0.0)
                                ratio = None
                                if abs(final_reward2) > 1e-6:
                                    ratio = partial_sum / abs(final_reward2)

                                result_info[side_id2]['race'] = race2
                                result_info[side_id2]['player_id'] = pid2
                                result_info[side_id2]['opponent_id'] = self.agents[idx2].opponent_id
                                result_info[side_id2]['winloss'] = final_reward2
                                result_info[side_id2]['agent_iters'] = agent_iters2
                                result_info[side_id2]['partial_reward_sum'] = partial_sum
                                if ratio is not None:
                                    result_info[side_id2]['partial_reward_ratio'] = ratio
                                result_info[side_id2].update(self.agents[idx2].get_unit_num_info())
                                result_info[side_id2].update(self.agents[idx2].get_stat_data())

                            game_duration = time.time() - game_start
                            result_info['game_steps'] = final_game_steps
                            result_info['game_iters'] = game_iters
                            result_info['game_duration'] = game_duration
                            # toxic summary
                            toxic_summary = self._toxic_monitor.summarize_toxic_strategies()
                            result_info.update(toxic_summary)

                            self._comm.send_result(result_info)

                        break

                    episode_count += 1

                except Exception as e:
                    print('[EPISODE LOOP ERROR]', e, flush=True)
                    print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
                    episode_count += 1
                    self._env.close()

            self._env.close()
            if result_queue is not None:
                print(os.getpid(), 'done')
                result_queue.put('done')
                # block to keep process alive if needed
                time.sleep(1)
            else:
                return

    def parse_logs(self, log_file):
        """
        RESTORED: parse_logs method, so we don't crash if play.py calls it.
        You can customize or remove if you don't need it.
        """
        if not os.path.exists(log_file):
            print(f"[parse_logs] No such file: {log_file}. Returning empty lists.")
            return [], []
        with open(log_file, 'r') as f:
            lines = f.readlines()

        spam_events = [ln for ln in lines if 'RollingRewardHackingMonitor' in ln]
        toxic_events = [ln for ln in lines if 'ToxicStrategyMonitor' in ln]
        return spam_events, toxic_events

    def summarize_results(self, result_file):
        """
        If your code or play.py calls it, implement logic here. Otherwise, safe to remove.
        """
        if not os.path.exists(result_file):
            print(f"[summarize_results] No such file: {result_file}")
            return
        with open(result_file, 'r') as f:
            data = json.load(f)
        print("Partial Reward Ratios:", data.get('partial_reward_ratio', 'N/A'))
        print("Toxic Strategy Summary:", data.get('toxic_strategy_summary', 'N/A'))

    def _gpu_inference_loop(self):
        """
        For GPU-batch usage, adapted from old code.
        """
        _, _ = dist_init(method='single_node')
        torch.set_num_threads(1)

        for agent in self.agents:
            agent.model = agent.model.cuda()
            if 'train' in self._job_type:
                agent.teacher_model = agent.teacher_model.cuda()

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

                for agent in self.agents:
                    agent.gpu_batch_inference()
                    if 'train' in self._job_type:
                        agent.gpu_batch_inference(teacher=True)

    def _start_multi_inference_loop(self):
        """
        Multi-env spawning.
        """
        self._close_processes()
        self._processes = []
        if hasattr(self, '_comm'):
            job = self._comm.job
        else:
            job = {}

        self.pipes = []
        context_str = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        mp_context = mp.get_context(context_str)
        self._result_queue = mp_context.Queue()

        for env_id in range(self._cfg.env_num):
            pipe_p, pipe_c = mp_context.Pipe()
            p = mp_context.Process(
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
                self._inference_loop()
            else:
                if self._job_type == 'train':
                    self._start_multi_inference_loop()
                    if self._gpu_batch_inference:
                        self._gpu_inference_loop()
                    else:
                        start_time = time.time()
                        while True:
                            if time.time() - start_time > self.max_job_duration:
                                self.reset()
                            self._comm.update_model(self)
                            time.sleep(1)

                if self._job_type == 'eval':
                    self._start_multi_inference_loop()
                    if self._gpu_batch_inference:
                        self._gpu_inference_loop()
                    else:
                        # Wait for all processes to finish
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
            for p in self.pipes:
                p.send('close')
            for p in self._processes:
                p.join()

    def iter_after_hook(self, iter_count, variable_record):
        if iter_count % self._cfg.print_freq == 0:
            if hasattr(self, '_comm'):
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


if __name__ == '__main__':
    actor = Actor(cfg={})
    actor.run()