#! /bin/bash

## Reserved Memory and CPU slots
## (Specify )
#$ -l h_vmem=4G
#$ -pe smp 2

## Logfile configuration
#$ -j y
#$ -o /data/scc/robert.loeffler/Logs/$JOB_NAME.$JOB_ID-$TASK_ID.log

## Send email on abort
#$ -m a
#$ -M robert.loeffler@uni-konstanz.de

## Queue 
## (don't use old, our libaries are compiled with modern math instructions)
#$ -q scc,long

## Exclude node scc131 (to old for math instruction set)
#$ -l h='!=scc131'

date --iso-8601=seconds

make

python3 <<ENDOFPYTHON

# --- Impose Thread limitations to tensorflow ---

import tensorflow
tensorflow.config.threading.set_inter_op_parallelism_threads(2)
tensorflow.config.threading.set_intra_op_parallelism_threads(2)

# --- Parse parameters and start simulation ---

import os
import learning_food

task_id = int($SGE_TASK_ID)
job_dir = os.path.abspath('$JOB_DIR')

learning_food.do_array_task(task_id, job_dir)

ENDOFPYTHON

date --iso-8601=seconds

