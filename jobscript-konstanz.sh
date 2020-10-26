#! /bin/bash

## Name (for logging and display)
#$ -N rl-food-ratio-test

## Maximum Memory
# #$ -l h_vmem=no idea ...

## Logfile configuration (folder must exist!)
#$ -j y
#$ -o /data/scc/robert.loeffler/Logs/

## Number of tasks
#$ -t 1-3

## Send email
## few large jobs, so success emails are also welcome
#$ -m eas
#$ -M robert.loeffler@uni-konstanz.de,epanizon@gmail.com

#-------------------

# The ratios to simulate in the different tasks
# Hack: the first one will never get used as tasks have to start at >=1
ratios=(placeholder 0.0 0.95 1.0)

code=`pwd`
data=/data/scc/robert.loeffler/2020-10-26-rl-food-ratio-test/ratio-${ratios[$SGE_TASK_ID]}

mkdir -p $data
cd $data

python3 $code/learning_GROUP_FOOD_dOBS.py ${ratios[$SGE_TASK_ID]} > results.dat

cd $code

