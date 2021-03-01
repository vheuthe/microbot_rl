#! /bin/bash

## Name (for logging and display)
#$ -N rl-food

## Maximum Memory
#$ -l h_vmem=8G

## Threads to use
#$ -pe smp 8

## Logfile configuration (folder must exist!)
#$ -j y
#$ -o /data/scc/robert.loeffler/Logs/

## Number of tasks
#$ -t 1-12

## Send email
## few large jobs, so success emails are also welcome
#$ -m eas
#$ -M robert.loeffler@uni-konstanz.de

#-------------------

python3 learning_robert_food.py $SGE_TASK_ID 

