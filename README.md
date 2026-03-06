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
- [x] web interface(next.js frontend)
- [ ] train strong policy network
- [ ] alphago style self-play
- [ ] benchmark against known bots

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
