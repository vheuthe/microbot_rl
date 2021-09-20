#! /bin/bash

## Reserved Memory and CPU slots
## (Specify )
#$ -l h_vmem=4G
#$ -pe smp 2

## Logfile configuration
#$ -j y
#$ -o /data/scc/$USER/Logs/$JOB_NAME.$JOB_ID-$TASK_ID.log

## Queue
## (don't use old, our libaries are compiled with modern math instructions)
#$ -q scc,long

## Exclude node scc131 (to old for math instruction set)
#$ -l h='!=scc131'

PROJECT=$(git status --branch --porcelain=v2 | grep '^# branch.head' | cut -d ' ' -f 3)

date --iso-8601=seconds

make -C fortran

python3 <<ENDOFPYTHON

# --- Impose Thread limitations to tensorflow ---

import tensorflow
tensorflow.config.threading.set_inter_op_parallelism_threads(2)
tensorflow.config.threading.set_intra_op_parallelism_threads(2)

# --- Parse parameters and start simulation ---

import os
import learning_$PROJECT as learning

task_id = int($SGE_TASK_ID)

job_dir = os.path.abspath('$JOB_DIR')

learning.do_array_task(task_id, job_dir)

ENDOFPYTHON

date --iso-8601=seconds
