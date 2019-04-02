# Autonomous Driving using Deep Deterministic Policy Gradients.
Based on [Kendall, et. al. 2018](1).

[1] Alex Kendall, Jeffrey Hawke, David Janz, Przemyslaw Mazur, Daniele Reda, John-Mark Allen, Vinh-Dieu Lam, Alex Bewley: “Learning to Drive in a Day”, 2018; [http://arxiv.org/abs/1807.00412 arXiv:1807.00412].

## Installation
1. Navigate to __code/autodrive__
2. Run
``` pip install -r requirements.txt ```
3. Navigate to __code/autodrive/keras-rl__
4. Run 
```pip install .```
5. Navigate to __code/autodrive/carla-client__
6. Run ```pip install .```

## Set Carla environment variable
1.```$ export CARLA_ROOT=path/to/carla/directory```  
2. Copy __mysettings.ini__ to CARLA_ROOT directory

## Run experiment
```
python run_experiment.py
```
