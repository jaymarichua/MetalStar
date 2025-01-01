#!/usr/bin/env python3
# coding: utf-8
#
# MetalStar play.py
#   - Demonstrates launching StarCraft II on macOS with MPS or CPU/CUDA fallback
#   - Creates an Actor instance, runs a single game in "eval_test" mode,
#     and then tries to parse the correct log file for RollingRewardHackingMonitor
#     or ToxicStrategyMonitor lines.

import os
import sys
import argparse
import torch
import warnings

# If your "actor" module is located differently, adjust the import path.
from distar.actor import Actor
from distar.ctools.utils import read_config

warnings.filterwarnings(
    "ignore",
    message="Setting attributes on ParameterList is not supported."
)

def load_default_config():
    """
    Provides a baseline config if you want a fallback
    and don't have a user_config.yaml present.
    """
    return {
        "actor": {
            "model_paths": {
                "model1": "default",
                "model2": "default",
            },
            "use_mps": True,
            "device": "mps",
            "player_ids": [],
        },
        "env": {
            "player_ids": [],
            "realtime": True
        },
        "common": {
            "type": "play",
            "experiment_name": "test"
        }
    }

print("DEBUG: Running the updated play.py with --race support on macOS (MetalStar)!")

def main():
    parser = argparse.ArgumentParser(
        description="MetalStar script for SC2 on macOS, focusing on MPS or CPU/CUDA fallback."
    )
    parser.add_argument(
        "--model1",
        type=str,
        default=None,
        help="First model's name minus '.pth'. E.g. 'sl_model' => sl_model.pth"
    )
    parser.add_argument(
        "--model2",
        type=str,
        default=None,
        help="Second model's name minus '.pth'."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forces CPU usage, ignoring MPS or CUDA."
    )
    parser.add_argument(
        "--game_type",
        type=str,
        default="human_vs_agent",
        choices=["agent_vs_agent", "agent_vs_bot", "human_vs_agent"],
        help="Match style. Default is 'human_vs_agent'."
    )
    parser.add_argument(
        "--race",
        type=str,
        default="zerg",
        choices=["zerg", "protoss", "terran"],
        help="Which SC2 race to use for the first player. Defaults to 'zerg'."
    )
    args = parser.parse_args()

    # Ensure SC2PATH is valid
    sc2path = os.environ.get("SC2PATH", "")
    if not sc2path:
        if sys.platform == "darwin":
            mac_default_sc2 = "/Applications/StarCraft II"
            if os.path.isdir(mac_default_sc2):
                os.environ["SC2PATH"] = mac_default_sc2
                print(f"[INFO] SC2PATH wasn't set, so using {mac_default_sc2} by default on macOS.")
            else:
                raise EnvironmentError(
                    "SC2PATH isn't set, and /Applications/StarCraft II not found. "
                    "Please install SC2 or export SC2PATH manually."
                )
        else:
            raise EnvironmentError(
                "SC2PATH is not set. Please specify your StarCraft II installation."
            )
    else:
        if not os.path.isdir(sc2path):
            raise NotADirectoryError(
                f"SC2PATH is '{sc2path}', but no such directory exists."
            )

    # Try to load user_config.yaml from the same folder
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'user_config.yaml')
    if os.path.isfile(config_path):
        user_config = read_config(config_path)
    else:
        # Fallback if no user_config.yaml found
        user_config = load_default_config()

    # Adjust job_type to an eval/test style, realtime on
    user_config["actor"]["job_type"] = "eval_test"
    user_config["common"]["type"] = "play"
    user_config["actor"]["episode_num"] = 1
    user_config["env"]["realtime"] = True

    # Start with no CUDA by default
    user_config["actor"]["use_cuda"] = False

    # Decide device preference
    if args.cpu:
        print("[INFO] CPU mode only. Disabling MPS/CUDA.")
        user_config["actor"]["device"] = "cpu"
        user_config["actor"]["use_mps"] = False
        user_config["actor"]["use_cuda"] = False
    else:
        # Try MPS, then CUDA, then CPU
        if torch.backends.mps.is_available():
            print("[INFO] MPS detected, using Metal for acceleration!")
            user_config["actor"]["device"] = "mps"
            user_config["actor"]["use_mps"] = True
            user_config["actor"]["use_cuda"] = False
        elif torch.cuda.is_available():
            print("[WARNING] MPS not available, falling back to CUDA.")
            user_config["actor"]["device"] = "cuda"
            user_config["actor"]["use_mps"] = False
            user_config["actor"]["use_cuda"] = True
        else:
            print("[WARNING] No MPS or CUDA found, falling back to CPU.")
            user_config["actor"]["device"] = "cpu"
            user_config["actor"]["use_mps"] = False
            user_config["actor"]["use_cuda"] = False

    default_model_path = os.path.join(script_dir, "rl_model.pth")

    # Model1 path
    if args.model1:
        user_config["actor"]["model_paths"]["model1"] = os.path.join(
            script_dir, args.model1 + ".pth"
        )
    model1_path = user_config["actor"]["model_paths"]["model1"]
    if model1_path == "default":
        model1_path = default_model_path
        user_config["actor"]["model_paths"]["model1"] = model1_path

    # Model2 path
    if args.model2:
        user_config["actor"]["model_paths"]["model2"] = os.path.join(
            script_dir, args.model2 + ".pth"
        )
    model2_path = user_config["actor"]["model_paths"]["model2"]
    if model2_path == "default":
        model2_path = default_model_path
        user_config["actor"]["model_paths"]["model2"] = model2_path

    # Validate model files exist
    if not os.path.exists(model1_path):
        raise FileNotFoundError(f"[ERROR] Model1 not found at {model1_path}")
    if not os.path.exists(model2_path):
        raise FileNotFoundError(f"[ERROR] Model2 not found at {model2_path}")

    # Override race for first slot in env.races if that field exists
    if "races" in user_config["env"] and len(user_config["env"]["races"]) > 0:
        user_config["env"]["races"][0] = args.race

    # Decide game type
    if args.game_type == "agent_vs_agent":
        user_config["env"]["player_ids"] = [
            os.path.basename(model1_path).split(".")[0],
            os.path.basename(model2_path).split(".")[0],
        ]
    elif args.game_type == "agent_vs_bot":
        user_config["actor"]["player_ids"] = ["model1"]
        bot_level = "bot10"
        if args.model2 and "bot" in args.model2:
            bot_level = args.model2
        user_config["env"]["player_ids"] = [
            os.path.basename(model1_path).split(".")[0],
            bot_level
        ]
    elif args.game_type == "human_vs_agent":
        user_config["actor"]["player_ids"] = ["model1"]
        user_config["env"]["player_ids"] = [
            os.path.basename(model1_path).split(".")[0],
            "human"
        ]

    # Create Actor and run
    actor = Actor(user_config)
    actor.run()

    # Return the Actor instance so we can parse logs or summarize results
    return actor

if __name__ == "__main__":
    actor_instance = main()

    # The actor writes logs under:
    #   experiments/<experiment_name>/actor_log/<actor_uid>.log
    # So let's reconstruct that path:
    experiment_name = actor_instance._whole_cfg["common"]["experiment_name"]
    log_dir = os.path.join(
        os.getcwd(),
        "experiments",
        experiment_name,
        "actor_log"
    )
    log_filename = f"{actor_instance._actor_uid}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # Attempt to parse logs:
    if hasattr(actor_instance, "parse_logs"):
        if os.path.isfile(log_file_path):
            spam_events, toxic_events = actor_instance.parse_logs(log_file_path)
            print("Spam Events:", spam_events)
            print("Toxic Events:", toxic_events)
        else:
            print(f"[WARNING] Log file not found at: {log_file_path}")
    else:
        print("[INFO] parse_logs method not found in actor. Skipping spam/toxic log parse.")