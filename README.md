# MetalStar: High-Performance Metal Performance Shaders (MPS) for StarCraft II AI

**License**  
This project is licensed under the Apache 2.0 License.  
The original DI-star is (c) OpenDILab, inspired by DeepMind's AlphaStar.  
All work in this fork is (c) 2024 Jaymari Chua.

> **Note**  
> “Applestar” or “MetalStar” refers to the same MPS-accelerated fork of DI-star. This fork adds Apple Metal (MPS) support for StarCraft II AI on macOS.
---
## Versions
- **Python** 3.10.0
- **PyTorch** 2.5.1
- **torchaudio** 2.5.1
---
## Command Line Usage Examples
### Apple MPS on Apple Silicon (MPS-Capable PyTorch)
MetalStar tries to use MPS first on Apple Silicon macOS; if MPS isn’t available, it prints a warning and falls back to CPU. If you really want CPU only, you can pass `--cpu`.
#### Human vs Agent (Default) with MPS
```bash
python play.py
```
Runs a “human_vs_agent” match using rl_model.pth (the default if you pass no model1). It uses MPS as long as your system supports it, else it warns and runs on CPU.
Human vs Agent with a Custom Model
Suppose you have my_rl_model.pth next to play.py:
```bash
python play.py --model1 my_rl_model
```
This looks for my_rl_model.pth and uses “human_vs_agent” by default. You can see if you can beat your own custom-trained model.
Agent vs Bot on MPS
```bash
python play.py --game_type agent_vs_bot
```
This sets up model1 (usually rl_model.pth) vs. the built-in bot at difficulty bot10. If you want a lower-level bot, e.g. bot7, run:
```bash
python play.py --model2 bot7 --game_type agent_vs_bot
```
In that case, "bot7" is interpreted as a built-in bot difficulty.
Agent vs Agent
```bash
python play.py --game_type agent_vs_agent --model1 rl_model --model2 sl_model
```
Now the reinforcement-learning model fights the supervised-learning model. Both run on MPS or CPU fallback. No humans involved.
Forcing CPU Mode
```bash
python play.py --cpu
```
This ignores MPS even if available, letting you test CPU performance.
Anodther Human vs Agent Example
```bash
python play.py --model1 rl_model
```
If you have rl_model.pth, it’ll attempt MPS first, use “human_vs_agent,” and let you face off against this advanced RL model.
macOS Installation
macOS Prerequisites
```bash
brew install python
brew install pip
brew install micromamba
```
```bash
micromamba create -n pytorch python=3.10
micromamba activate pytorch
micromamba install pytorch torchvision torchaudio -c pytorch -c conda-forge
```
DI-star Prerequisites
Clone or download the Applestar/MetalStar folder, then change directory into it:
```bash
pip install -e .
```
macOS Troubleshooting (VideoCard Errors)
Occasionally, if a game session doesn’t terminate gracefully, you might get video card–related errors. To fix this, open Battle.net settings and choose Restore In-Game Options.
Rolling Updates
	•	MPS support for model inference at play.py (updated on 2024-12-28)
	•	Latest PyTorch that supports MPS (torch._six fix → import math.inf, etc.) (2024-12-28)
	•	Tested on Python 3.10.0, torch 2.5.1, torchaudio 2.5.1 (updated on 2024-12-29)
	•	StarCraft II version pinned at 4.10.0 for model compatibility (updated on 2024-12-29)
	•	Plan to add MPS-based distributed training (WIP)
License and Attribution
	•	Apache 2.0 License
	•	Original DI-star: (c) OpenDILab
	•	MetalStar (this fork): (c) 2024 Jaymari Chua
DI-Star Overview
DI-star is a large-scale game AI distributed training platform for StarCraft II, originally by OpenDILab:
	•	Demo and test code (play with our agent!)
	•	Pre-trained SL and RL agent (Zerg vs Zerg)
	•	Training code (Supervised and Reinforcement Learning), updated 2022-01-31
	•	Training baseline on limited resources, plus guidance (2022-04-24)
	•	Agents fought [Harstem on YouTube] (2022-04-01)
	•	More robust pre-trained RL agents (WIP)
Usage
	•	Testing software on Windows
	•	对战软件下载
Please star the DI-star project to support the community’s growth.
Installation Requirements
	•	Python 3.6–3.8
1. Install StarCraftII
	•	Download retail SC2 from Blizzard’s official site.
	•	For Linux, see Blizzard’s instructions.
	•	Add SC2PATH to your environment if not installed in the default location. On macOS, typically /Applications/StarCraft II.
2. Install distar
git clone https://github.com/opendilab/DI-star.git
cd DI-star
pip install -e .
3. Install PyTorch
PyTorch official site for instructions.
Note: a GPU is recommended for real-time agent tests, or MPS on Apple Silicon if you’re using MetalStar.
Play with Pre-Trained Agent
1. Download SC2 version 4.10.0
Double-click data/replays/replay_4.10.0.SC2Replay to auto-download SC2 4.10.0.
2. Download Models
`python -m distar.bin.download_model --name rl_model`
	•	rl_model: reinforcement learning (Master/Grandmaster level)
	•	sl_model: supervised from human replays (Diamond level)
	•	Others: Abathur, Brakk, Dehaka, Zagara for different Zerg styles.
3. Agent Test
Play vs Agent
`python -m distar.bin.play`
	•	Default uses rl_model on GPU or CPU. If you want MPS (Metal), see the MetalStar instructions.
Agent vs Agent
`python -m distar.bin.play --game_type agent_vs_agent`
Agent vs Bot
`python -m distar.bin.play --game_type agent_vs_bot`
Building Your Own Agent
Instructions can be found in docs/agent.md. The design is modular so multiple agents can share the same environment.
Training (SL and RL)
See docs/guidance_to_small_scale_training.md for small-scale training pipelines.
Community
Slack: link
Discord: link
Recommended Citation
```latex
@misc{distar,
    title={DI-star: An Open-sourse Reinforcement Learning Framework for StarCraftII},
    author={DI-star Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/opendilab/DI-star}},
    year={2021},
}
@misc{metalstar,
    title={MetalStar: High-Performance Metal Performance Shaders MPS for StarCraft II AI},
    author={Jaymari Chua},
    publisher={GitHub},
    howpublished={\url{https://github.com/jaymarichua/MetalStar}},
    year={2024},
}
```
More Info
This project is forked from opendilab/DI-star. MetalStar focuses on macOS Metal (MPS) acceleration. The original DI-star is released under the Apache 2.0 license.
