#! /bin/bash
#$ -S bash


#  Task name (for logging and display)
#$ -N 2021-00-00-rl-change-this

#  Number of tasks (should match the possible parameter sets)
#$ -t 1-33

## Reserved Memory and CPU slots
#$ -l h_vmem=4G
#$ -pe smp 2

## Logfile configuration
#$ -j y
#$ -o /data/scc/robert.loeffler/Logs/

## Send email on abort
#$ -m a
#$ -M robert.loeffler@uni-konstanz.de

## Queue (don't use old)
#$ -q scc,long

## Exclude node scc131 (to old for math instruction set)
#$ -l h='!=scc131'


make

python3 <<ENDOFPYTHON

# folder to which data gets saved
job_name = '$JOB_NAME'

# parameter ranges that are used
job_parameters = {
    'food_rew': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'touch_penalty': [0.5, 1.0, 2.0],
}

# --- Impose Thread limitations to tensorflow ---

import tensorflow
tensorflow.config.threading.set_inter_op_parallelism_threads(2)
tensorflow.config.threading.set_intra_op_parallelism_threads(2)

# --- Setup parameters and start simulation ---

import os
import numpy as np
import learning_robert_food

# get array job index (SGE starts counting at 1!)
task_id = int($SGE_TASK_ID)

# choose one set out of all possible parameter combinations
selected_parameters = dict(zip(
    job_parameters.keys(),
    [vals.flat[task_id - 1] for vals in np.meshgrid(*job_parameters.values())]
))

# construct folder name from relevant parameters
data_dir = os.path.join(
    '/data/scc/robert.loeffler', 
    job_name, 
    '_'.join([key + str(val) for key, val in selected_parameters.items()])
)

# start simulation
learning_robert_food.do_task(selected_parameters, data_dir)

ENDOFPYTHON