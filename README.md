## Neural Heuristic Search

### Introduction
This repository is based on the paper: Modular Construction Planning using Graph Neural Network Heuristic Search

[PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) are used to code the heuristic network.

<img src="images/SHNet.png"
     alt="Search heuristic network"
     style="float: margin-bottom: 50px;" />


### Environment Setup
To create the environment with conda...

for CPU only
```
conda env create -f environment_cpu.yml
```
or with GPU support

```
conda env create -f environment_gpu.yml
```

Activate the environment

```
conda activate neural_heuristic_search
```
### Running Examplar Searches

For the experiments in the paper we train against the results of A* searches for trusses spanning 1-3 struts/spokes

```
mkdir data
python build_training.py --target_dist 1 --train_example_path "./data/train_d1.pkl"
python build_training.py --target_dist 2 --train_example_path "./data/train_d2.pkl"
python build_training.py --target_dist 3 --train_example_path "./data/train_d3.pkl"

```

### Training the Heuristic Network

Training the GIN based network:
```
python train.py --config GIN --epochs 2000 --batch_size 32 --lr 1e-3 --lr_patience 100 --train_data_search './data/train_d*.pkl'
```

Training the GAT based network:
```
python train.py --config GAT --epochs 2000 --batch_size 32 --lr 1e-3 --lr_patience 100 --train_data_search './data/train_d*.pkl'
```

### Visualising Truss Construction Planning

<img src="images/truss.png"
     alt="Generated truss"
     style="float: margin-bottom: 50px;" />

The planning search and final construction plan can be viewed for a scenario


using a minimum Manhattan distance heuristic:
```
python scenario.py --target_dist 3 --heuristic Manhattan --show_search
```

using a mean Manhattan distance heuristic:
```
python scenario.py --target_dist 5 --heuristic Mean --show_search
```

using the GIN heuristic network:
```
python scenario.py --target_dist 5 --model_config GIN --show_search
```


### Evaluation

Planning construction for 25 random scenarios with 5 span trusses:

```
python evaluate.py --planner_file demo_eval_planners.json --target_dist 5 --eps 1000 --max_scenarios 25 --batch_size 64 
```

Using demo_eval_planners.json this evaluates the following four agent heuristics for each scenario:
- GIN heuristic network
- GAT heuristic netork
- Minimum Manhattan distance heuristic
- Mean Manhattan distance heuristic

Results are output to ```./logs/results.json``` by default