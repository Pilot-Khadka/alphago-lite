# alphago-lite
![Tests](https://github.com/Pilot-Khadka/alphago-lite/actions/workflows/test.yml/badge.svg)

A lightweight implementation of Alphago style training pipeline based on the book
- Deep Learning and the game of Go, and the paper
- Mastering the game of Go with Deep Neural Networks and Tree Search

## implementation
- [x] fast policy network
- [x] strong policy network
- [x] self-play loop -q learning, policy learning
  - [x] policy gradient learning
  - [x] Q-learning
- [x] Bot vs Bot gameplay
- [x] web interface (fastapi + canvas)
- [x] train strong policy network (49.8% val top-1 on pro games)
- [ ] alphago style self-play
- [ ] benchmark against known bots

## layout
```
src/alphago/        library: go engine, encoders, networks, agents, training, web app
  paths.py          all data / checkpoint / log locations
  web/              fastapi server + single canvas page
scripts/
  data/             download, build_games, visualize_game
  train/            policy_sl, policy_gradient, q_learning(_parallel)
  eval/             list_runs, bot_vs_bot, model_v_model, evaluate_self_play, benchmark_engine
config/             sl_alphago.yaml, sl_fast.yaml (training), data.yaml (download)
test/
data/               (ignored) raw/, processed/games/<archive>/, experience/
checkpoints/        (ignored) <stage>/<run_id>/ per training run, rl/<method>/
runs/               (ignored) tensorboard logs
```

## usage
```
uv run scripts/data/download.py
uv run scripts/data/build_games.py
uv run scripts/train/policy_sl.py --config config/sl_alphago.yaml
uv run python -m alphago.web.app      # http://127.0.0.1:8000
```

## supervised training
1. `scripts/data/build_games.py` parses the raw SGF files once into a small game archive
   `data/processed/games/pro_v1/{train,val,test}.npz` (moves only, split by game).
2. `scripts/train/policy_sl.py` encodes positions on the fly in the data loader workers
   (8 board symmetries) and trains with SGD + warmup/cosine lr, AMP and a step budget.
   - `--config config/sl_alphago.yaml`: AlphaGo 13-layer policy net, 48-plane encoder
   - `--config config/sl_fast.yaml`: small residual net, 7-plane encoder
   - `--set train.total_steps=100000 train.lr=0.02`: override any config value
   - `--max-minutes 600`: stop cleanly at a time limit (status `paused`)
   - `--resume checkpoints/supervised/<run_id>`: continue a paused run
   - `--init-from checkpoints/supervised/<run_id>/best.pth`: start a new run from those
     weights with a fresh lr schedule (to train a finished run longer)
3. Each run gets `checkpoints/supervised/<run_id>/` with `run.json` (config, git commit,
   sessions), `metrics.jsonl`, `best.pth`, `last.pth` and `step_*.pth` snapshots.
   `uv run scripts/eval/list_runs.py -v` lists runs; the web app loads the `best.pth`
   with the highest val top-1 (or `--checkpoint PATH`).

## implementation overview

The project roughly follows the pipeline described in *Deep Learning and the Game of Go*.

### 1. supervised learning
Train a **policy network** to imitate human moves from professional Go games.

### 2. policy improvement
Improve the policy using **reinforcement learning through self-play**.

### 3. evaluation
Evaluate trained agents by running **automated matches between models**.


## dataset
This project uses the **Computer Go Dataset**:

https://github.com/yenw/computer-go-dataset.git


## references
- *Deep Learning and the Game of Go*
- Silver et al.,
  *Mastering the Game of Go with Deep Neural Networks and Tree Search*
  https://www.nature.com/articles/nature16961
