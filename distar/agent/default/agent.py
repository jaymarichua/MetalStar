import copy
import json
import os
import random
import time

import torch
from copy import deepcopy
from collections import deque, defaultdict
from functools import partial
from torch.utils.data._utils.collate import default_collate

# Adjust these imports as needed for your own directory structure:
from .model.model import Model
from .lib.actions import (
    NUM_CUMULATIVE_STAT_ACTIONS, ACTIONS, BEGINNING_ORDER_ACTIONS,
    CUMULATIVE_STAT_ACTIONS, UNIT_ABILITY_TO_ACTION, QUEUE_ACTIONS,
    UNIT_TO_CUM, UPGRADE_TO_CUM
)
from .lib.features import (
    Features, SPATIAL_SIZE, BEGINNING_ORDER_LENGTH, compute_battle_score,
    fake_step_data, fake_model_output
)
from .lib.stat import Stat, cum_dict
from distar.ctools.torch_utils.metric import (
    levenshtein_distance, hamming_distance, l2_distance
)
from distar.pysc2.lib.units import get_unit_type
from distar.pysc2.lib.static_data import UNIT_TYPES, NUM_UNIT_TYPES
from distar.ctools.torch_utils import to_device

RACE_DICT = {
    1: 'terran',
    2: 'zerg',
    3: 'protoss',
    4: 'random',
}


def copy_input_data(shared_step_data, step_data, data_idx):
    """
    Copies input data into the shared buffer for GPU-batch inference.
    Includes debug logs to trace shape details and data flow.
    """
    entity_num = step_data['entity_num']
    selected_units_num = step_data.get('selected_units_num', 0)

    print(f"[DEBUG/copy_input_data] data_idx={data_idx}, entity_num={entity_num}, selected_units_num={selected_units_num}")

    for k, v in step_data.items():
        if k == 'hidden_state':
            for i in range(len(v)):
                print(f"[DEBUG/copy_input_data] hidden_state layer={i}, "
                      f"h.shape={v[i][0].shape}, c.shape={v[i][1].shape}")
                shared_step_data['hidden_state'][i][0][data_idx].copy_(v[i][0])
                shared_step_data['hidden_state'][i][1][data_idx].copy_(v[i][1])
            continue

        if k == 'value_feature':
            # If needed, copy or skip copying
            continue

        if isinstance(v, torch.Tensor):
            print(f"[DEBUG/copy_input_data] key={k}, shape={v.shape}")
            shared_step_data[k][data_idx].copy_(v)
        elif isinstance(v, dict):
            for _k, _v in v.items():
                if k == 'action_info' and _k == 'selected_units':
                    if selected_units_num > 0:
                        print(f"[DEBUG/copy_input_data] action_info->selected_units shape={_v.shape}")
                        shared_step_data[k][_k][data_idx, :selected_units_num].copy_(_v)
                elif k == 'entity_info':
                    print(f"[DEBUG/copy_input_data] entity_info->({_k}) shape={_v.shape}")
                    shared_step_data[k][_k][data_idx, :entity_num].copy_(_v)
                elif k == 'spatial_info':
                    if 'effect' in _k:
                        print(f"[DEBUG/copy_input_data] spatial_info->effect={_k}, shape={_v.shape}")
                        shared_step_data[k][_k][data_idx].copy_(_v)
                    else:
                        h, w = _v.shape
                        print(f"[DEBUG/copy_input_data] spatial_info->({_k}) shape={_v.shape}")
                        shared_step_data[k][_k][data_idx] *= 0
                        shared_step_data[k][_k][data_idx, :h, :w].copy_(_v)
                else:
                    print(f"[DEBUG/copy_input_data] dict key={k}->{_k}, shape={_v.shape}")
                    shared_step_data[k][_k][data_idx].copy_(_v)


def copy_output_data(shared_step_data, step_data, data_indexes):
    """
    Copies output data (model inference results) from the shared buffer
    back into environment-specific data structures.
    Additional logs help confirm shapes after index_copy.
    """
    data_indexes = data_indexes.nonzero().squeeze(dim=1)
    print(f"[DEBUG/copy_output_data] data_indexes={data_indexes.tolist()}")

    for k, v in step_data.items():
        if k == 'hidden_state':
            for i in range(len(v)):
                print(f"[DEBUG/copy_output_data] hidden_state layer={i}, "
                      f"h.shape={v[i][0].shape}, c.shape={v[i][1].shape}")
                shared_step_data['hidden_state'][i][0].index_copy_(
                    0, data_indexes, v[i][0][data_indexes].cpu()
                )
                shared_step_data['hidden_state'][i][1].index_copy_(
                    0, data_indexes, v[i][1][data_indexes].cpu()
                )
        elif isinstance(v, dict):
            for _k, _v in v.items():
                if len(_v.shape) == 3:
                    _, s1, s2 = _v.shape
                    print(f"[DEBUG/copy_output_data] 3D key=({k}->{_k}) shape={_v.shape}")
                    shared_step_data[k][_k][:, :s1, :s2].index_copy_(
                        0, data_indexes, _v[data_indexes].cpu()
                    )
                elif len(_v.shape) == 2:
                    _, s1 = _v.shape
                    print(f"[DEBUG/copy_output_data] 2D key=({k}->{_k}) shape={_v.shape}")
                    shared_step_data[k][_k][:, :s1].index_copy_(
                        0, data_indexes, _v[data_indexes].cpu()
                    )
                elif len(_v.shape) == 1:
                    print(f"[DEBUG/copy_output_data] 1D key=({k}->{_k}) shape={_v.shape}")
                    shared_step_data[k][_k].index_copy_(
                        0, data_indexes, _v[data_indexes].cpu()
                    )
        elif isinstance(v, torch.Tensor):
            print(f"[DEBUG/copy_output_data] key={k}, shape={v.shape}")
            shared_step_data[k].index_copy_(
                0, data_indexes, v[data_indexes].cpu()
            )


class Agent:
    """
    A single SC2 agent that:
      - Runs model inference
      - Manages building-order/cumulative-stat reward shaping
      - Collects data for training
      - Optionally handles GPU batch inference
      - With extended debug logs to diagnose shape or observation mismatches
    """

    HAS_MODEL = True
    HAS_TEACHER_MODEL = True
    HAS_SUCCESSIVE_MODEL = False

    def __init__(self, cfg=None, env_id=0):
        self._whole_cfg = cfg
        self._job_type = cfg.actor.job_type
        self._env_id = env_id

        print(f"[Agent.__init__] job_type={self._job_type}, env_id={env_id}")

        # Basic config from 'learner' section
        learner_cfg = cfg.get('learner', {})
        self._only_cum_action_kl = learner_cfg.get('only_cum_action_kl', False)
        self._bo_norm = learner_cfg.get('bo_norm', 20)
        self._cum_norm = learner_cfg.get('cum_norm', 30)
        self._battle_norm = learner_cfg.get('battle_norm', 30)

        # Main model
        self.model = Model(cfg)
        self._num_layers = self.model.cfg.encoder.core_lstm.num_layers
        self._hidden_size = self.model.cfg.encoder.core_lstm.hidden_size

        # Basic agent identity
        self._player_id = None
        self._race = None
        self._iter_count = 0
        self._model_last_iter = 0

        # Z/Building order logic
        agent_cfg = cfg.agent
        self._z_path = agent_cfg.z_path
        self._bo_zergling_num = agent_cfg.get('bo_zergling_num', 8)
        self._fake_reward_prob = agent_cfg.get('fake_reward_prob', 1.0)
        self._use_value_feature = learner_cfg.get('use_value_feature', False)
        self._clip_bo = agent_cfg.get('clip_bo', True)
        self._cum_type = agent_cfg.get('cum_type', 'action')  # 'observation' or 'action'
        self._zero_z_value = cfg.get('feature', {}).get('zero_z_value', 1.0)
        self._zero_z_exceed_loop = agent_cfg.get('zero_z_exceed_loop', False)
        self._extra_units = agent_cfg.get('extra_units', False)

        # Hidden states
        self._hidden_state = [
            (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
            for _ in range(self._num_layers)
        ]

        # GPU batch inference logic
        self._gpu_batch_inference = cfg.actor.get('gpu_batch_inference', False)
        if self._gpu_batch_inference:
            print(f"[Agent.__init__] GPU batch_inference enabled.")
            batch_size = cfg.actor.env_num
            self._shared_input = fake_step_data(
                share_memory=True,
                batch_size=batch_size,
                hidden_size=self._hidden_size,
                hidden_layer=self._num_layers,
                train=False
            )
            self._shared_output = fake_model_output(
                batch_size=batch_size,
                hidden_size=self._hidden_size,
                hidden_layer=self._num_layers,
                teacher=False
            )
            self._signals = torch.zeros(batch_size).share_memory_()

            if 'train' in self._job_type:
                self._teacher_shared_input = fake_step_data(
                    share_memory=True,
                    batch_size=batch_size,
                    hidden_size=self._hidden_size,
                    hidden_layer=self._num_layers,
                    train=True
                )
                self._teacher_shared_output = fake_model_output(
                    batch_size=batch_size,
                    hidden_size=self._hidden_size,
                    hidden_layer=self._num_layers,
                    teacher=True
                )
                self._teacher_signals = torch.zeros(batch_size).share_memory_()

        # If in train mode, build teacher model if needed
        if 'train' in self._job_type:
            self.teacher_model = Model(cfg)

        # Realtime initialization check
        if cfg.env.realtime:
            data_init = fake_step_data(
                share_memory=True, batch_size=1,
                hidden_size=self._hidden_size,
                hidden_layer=self._num_layers, train=False
            )
            if cfg.actor.use_cuda:
                data_init = to_device(data_init, torch.cuda.current_device())
                self.model = self.model.cuda()
            with torch.no_grad():
                print("[Agent.__init__] Doing init pass for real-time mode.")
                _ = self.model.compute_logp_action(**data_init)

        self.z_idx = None  # optional advanced usage

    @property
    def env_id(self):
        return self._env_id

    @env_id.setter
    def env_id(self, val):
        print(f"[Agent.env_id.setter] updating from {self._env_id} to {val}")
        self._env_id = val

    @property
    def player_id(self):
        return self._player_id

    @player_id.setter
    def player_id(self, val):
        print(f"[Agent.player_id.setter] updating from {self._player_id} to {val}")
        self._player_id = val

    @property
    def race(self):
        return self._race

    @property
    def iter_count(self):
        return self._iter_count

    def reset(self, map_name, race_str, game_info, obs):
        """
        Prepare agent state for a new match. Clears hidden states,
        resets building order logic, etc. Includes thorough debug logs.
        """
        print(f"[Agent.reset] Called with map_name={map_name}, race={race_str}. Resetting agent state.")
        self._race = race_str
        self._map_name = map_name
        self._iter_count = 0
        self._model_last_iter = 0

        self._hidden_state = [
            (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
            for _ in range(self._num_layers)
        ]
        print(f"[Agent.reset] Re-initialized hidden_state for {self._num_layers} layers.")

        self._last_action_type = torch.tensor(0, dtype=torch.long)
        self._last_delay = torch.tensor(0, dtype=torch.long)
        self._last_queued = torch.tensor(0, dtype=torch.long)
        self._last_selected_unit_tags = None
        self._last_target_unit_tag = None
        self._last_location = None

        self._stat_api = Stat(race_str)
        self._feature = Features(game_info, obs['raw_obs'], self._whole_cfg)

        self._behaviour_building_order = []
        self._behaviour_bo_location = []
        self._bo_zergling_count = 0
        self._behaviour_cumulative_stat = [0] * NUM_CUMULATIVE_STAT_ACTIONS
        self._exceed_flag = True
        self._game_step = 0

        if 'train' in self._job_type:
            self._hidden_state_backup = [
                (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
                for _ in range(self._num_layers)
            ]
            self._teacher_hidden_state = [
                (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
                for _ in range(self._num_layers)
            ]
            self._data_buffer = deque(maxlen=self._whole_cfg.actor.traj_len)
            self._push_count = 0

        # If you load z_data for building orders, do so here with logs, e.g.:
        # ...
        print("[Agent.reset] Completed. Ready to proceed with step calls.")

    def _pre_process(self, obs):
        """
        Transform raw env obs => agent-compatible obs => model input
        with debug logs to see shape details or late observation steps.
        """
        raw_obs_keys = list(obs.keys())
        print(f"[Agent._pre_process] step obs keys={raw_obs_keys}")

        if self._use_value_feature:
            agent_obs = self._feature.transform_obs(
                obs['raw_obs'], padding_spatial=True,
                opponent_obs=obs.get('opponent_obs')
            )
        else:
            agent_obs = self._feature.transform_obs(
                obs['raw_obs'], padding_spatial=True
            )

        self._game_info = agent_obs.pop('game_info')
        self._game_step = self._game_info.get('game_loop', 0)
        print(f"[Agent._pre_process] game_loop={self._game_step}, zero_z_exceed_loop={self._zero_z_exceed_loop}")

        # Potential check if we exceed the z_loop => disable advanced bo/cum, etc.
        # if self._zero_z_exceed_loop and self._game_step > self._target_z_loop:
        #     self._exceed_flag = False
        #     self._target_z_loop = 999999999

        # Manage last_selected_units / last_target_unit in entity_info
        ent_num = agent_obs['entity_num']
        last_sel = torch.zeros(ent_num, dtype=torch.int8)
        last_tgt = torch.zeros(ent_num, dtype=torch.int8)
        tags = self._game_info['tags']

        if self._last_selected_unit_tags:
            for t in self._last_selected_unit_tags:
                if t in tags:
                    idx = tags.index(t)
                    last_sel[idx] = 1

        if self._last_target_unit_tag:
            if self._last_target_unit_tag in tags:
                idx = tags.index(self._last_target_unit_tag)
                last_tgt[idx] = 1

        agent_obs['entity_info']['last_selected_units'] = last_sel
        agent_obs['entity_info']['last_targeted_unit'] = last_tgt

        # Include hidden_state
        agent_obs['hidden_state'] = self._hidden_state
        agent_obs['scalar_info']['last_delay'] = self._last_delay
        agent_obs['scalar_info']['last_action_type'] = self._last_action_type
        agent_obs['scalar_info']['last_queued'] = self._last_queued

        # If using building order or cum stat, apply them here
        # e.g. agent_obs['scalar_info']['beginning_order'] = ...
        # e.g. agent_obs['scalar_info']['cumulative_stat'] = ...

        self._observation = agent_obs

        # If using CUDA, move to GPU
        if self._whole_cfg.actor.use_cuda:
            agent_obs = to_device(agent_obs, 'cuda:0')

        # If in GPU batch inference mode
        if self._gpu_batch_inference:
            copy_input_data(self._shared_input, agent_obs, data_idx=self._env_id)
            self._signals[self._env_id] += 1
            model_input = None
        else:
            model_input = default_collate([agent_obs])

        return model_input

    def step(self, observation):
        """
        Called each environment step.
        Additional logs for iteration, shapes, and possible mismatch detection.
        """
        print(f"[Agent.step] {self._player_id}, iteration={self._iter_count}, last game_step={self._game_step}")

        # Possibly update fake reward if in eval mode after first iteration
        if 'eval' in self._job_type and self._iter_count > 0 and not self._whole_cfg.env.realtime:
            self._update_fake_reward(self._last_action_type, self._last_location, observation)

        model_input = self._pre_process(observation)

        # Update stats with action_result
        if 'action_result' in observation:
            self._stat_api.update(self._last_action_type, observation['action_result'][0],
                                  self._observation, self._game_step)
        else:
            print("[Agent.step] No 'action_result' key in observation - skipping stat_api update.")

        # Normal vs batch inference
        if not self._gpu_batch_inference:
            if model_input is not None:
                print(f"[Agent.step] model_input shapes => {list(model_input[0].keys())}")
            model_output = self.model.compute_logp_action(**model_input)
        else:
            while True:
                if self._signals[self._env_id] == 0:
                    model_output = self._shared_output
                    break
                time.sleep(0.01)

        final_action = self._post_process(model_output)
        self._iter_count += 1
        return final_action

    def _post_process(self, output):
        """
        De-batch the model output => final action dict.
        Also logs hidden_state shapes for better debugging.
        """
        if self._gpu_batch_inference:
            out = self.decollate_output(output, batch_idx=self._env_id)
        else:
            out = self.decollate_output(output)

        self._hidden_state = out['hidden_state']
        self._last_queued = out['action_info']['queued']
        self._last_action_type = out['action_info']['action_type']
        self._last_delay = out['action_info']['delay']
        self._last_location = out['action_info']['target_location']
        self._output = out

        print("[Agent._post_process] Updated hidden_state:")
        for i, (hh, cc) in enumerate(self._hidden_state):
            print(f"  layer={i}: h.shape={hh.shape}, c.shape={cc.shape}")

        action_type_item = out['action_info']['action_type'].item()
        final_action = {
            'func_id': ACTIONS[action_type_item]['func_id'],
            'skip_steps': out['action_info']['delay'].item(),
            'queued': out['action_info']['queued'].item(),
            'unit_tags': [],
            'target_unit_tag': 0,
            'location': (0, 0)
        }

        sel_num = out['selected_units_num']
        for i in range(sel_num - 1):
            try:
                tag_idx = out['action_info']['selected_units'][i].item()
                final_action['unit_tags'].append(self._game_info['tags'][tag_idx])
            except:
                print("[WARN] _post_process => selected_units mismatch?")

        if self._extra_units:
            ex_units = torch.nonzero(out.get('extra_units', []), as_tuple=False)
            if ex_units.numel() > 0:
                for z in ex_units.squeeze(dim=1).tolist():
                    final_action['unit_tags'].append(self._game_info['tags'][z])

        # If action requires target_unit
        if ACTIONS[action_type_item]['target_unit']:
            targ_unit_idx = out['action_info']['target_unit'].item()
            final_action['target_unit_tag'] = self._game_info['tags'][targ_unit_idx]
            self._last_target_unit_tag = final_action['target_unit_tag']
        else:
            self._last_target_unit_tag = None

        # location
        xy_val = out['action_info']['target_location'].item()
        x_val = xy_val % SPATIAL_SIZE[1]
        y_val = xy_val // SPATIAL_SIZE[1]
        inv_y = max(self._feature.map_size.y - y_val, 0)
        final_action['location'] = (x_val, inv_y)

        # For debug in 'test' mode:
        # if 'test' in self._job_type:
        #     self._print_action(out['action_info'], [x_val,y_val], out['action_logp'])

        return [final_action]

    def decollate_output(self, output, k=None, batch_idx=None):
        """
        Splits a batched model output into single. Logging shapes if needed.
        """
        if isinstance(output, torch.Tensor):
            if batch_idx is None:
                return output.squeeze(dim=0)
            else:
                return output[batch_idx].clone().cpu()
        elif k == 'hidden_state':
            if batch_idx is None:
                return [(output[l][0].squeeze(dim=0), output[l][1].squeeze(dim=0)) for l in range(len(output))]
            else:
                return [
                    (
                        output[l][0][batch_idx].clone().cpu(),
                        output[l][1][batch_idx].clone().cpu()
                    )
                    for l in range(len(output))
                ]
        elif isinstance(output, dict):
            data = {
                subk: self.decollate_output(subv, subk, batch_idx)
                for subk, subv in output.items()
            }
            if batch_idx is not None and k is None:
                entity_num = data['entity_num']
                selected_units_num = data['selected_units_num']
                data['logit']['selected_units'] = data['logit']['selected_units'][:selected_units_num, :entity_num + 1]
                data['logit']['target_unit'] = data['logit']['target_unit'][:entity_num]
                if 'action_info' in data:
                    data['action_info']['selected_units'] = data['action_info']['selected_units'][:selected_units_num]
                    data['action_logp']['selected_units'] = data['action_logp']['selected_units'][:selected_units_num]
            return data
        return output

    def gpu_batch_inference(self, teacher=False):
        """
        For GPU-batch usage in a separate loop. Debug logs confirm data flow.
        """
        if not teacher:
            inference_indexes = self._signals.clone().bool()
            batch_num = inference_indexes.sum().item()
            print(f"[Agent.gpu_batch_inference] teacher=False, batch_num={batch_num}")
            if batch_num <= 0:
                return
            model_input = to_device(self._shared_input, torch.cuda.current_device())
            model_output = self.model.compute_logp_action(**model_input)
            copy_output_data(self._shared_output, model_output, inference_indexes)
            self._signals[inference_indexes] *= 0
        else:
            inference_indexes = self._teacher_signals.clone().bool()
            batch_num = inference_indexes.sum().item()
            print(f"[Agent.gpu_batch_inference] teacher=True, batch_num={batch_num}")
            if batch_num <= 0:
                return
            model_input = to_device(self._teacher_shared_input, torch.cuda.current_device())
            model_output = self.teacher_model.compute_teacher_logit(**model_input)
            copy_output_data(self._teacher_shared_output, model_output, inference_indexes)
            self._teacher_signals[inference_indexes] *= 0

    def _update_fake_reward(self, last_action_type, last_location, obs):
        """
        Internal helper for 'fake reward' shaping or debug logs.
        """
        print(f"[Agent._update_fake_reward] last_action_type={last_action_type.item()}, location={last_location}")
        # Additional shaping logic or logging

    def update_fake_reward(self, next_obs):
        """
        If we need to update external partial/fake reward without a new action.
        """
        print("[Agent.update_fake_reward] Called with next_obs keys:", list(next_obs.keys()))
        # Additional partial shaping or logging

    def collect_data(self, next_obs, reward, done, idx):
        """
        Gathers trajectory data if in training. Additional logs for debugging step logic.
        """
        print(f"[Agent.collect_data] done={done}, idx={idx}, reward={reward}, next_obs keys={list(next_obs.keys())}")
        # Build data structure. Possibly return a list, dict, or None.
        return None

    def get_unit_num_info(self):
        """
        Example: returns how many units we have, or other stats.
        """
        return {'unit_num': self._stat_api.unit_num}

    def get_stat_data(self):
        """
        Returns stat data about agent’s building order or cumulative distances, if any.
        In a real scenario, you'd compute BO or cum distances here.
        """
        print("[Agent.get_stat_data] Called; returning minimal data for demonstration.")
        return {}

    @staticmethod
    def _get_time_factor(game_step):
        """
        Basic time factor function for shaping delayed rewards or partial logic.
        """
        if game_step < 10000:
            return 1.0
        elif game_step < 20000:
            return 0.5
        elif game_step < 30000:
            return 0.25
        else:
            return 0

    def _print_action(self, action_info, location, logp):
        """
        Debug prints about the final chosen action in step.
        For further shape/logic coverage if 'test' job_type is used.
        """
        act_type_id = action_info['action_type'].item()
        action_name = ACTIONS[act_type_id]['name']
        su_len = len(action_info['selected_units'])
        su_str = ""

        for i, s_idx in enumerate(action_info['selected_units'][:-1].tolist()):
            su_str += (f" {get_unit_type(UNIT_TYPES[self._observation['entity_info']['unit_type'][s_idx]])}"
                       f"({torch.exp(logp['selected_units'][i]).item():.2f})")
        su_str += f" end({torch.exp(logp['selected_units'][-1]).item():.2f})"

        print(f"[Agent._print_action] {self.player_id}, game_step:{self._game_step}, "
              f"action={action_name}({torch.exp(logp['action_type']).item():.2f}), "
              f"delay={action_info['delay']}({torch.exp(logp['delay']).item():.2f}), "
              f"su=({su_len}){su_str}, loc={location}({torch.exp(logp['target_location']).item():.2f})")