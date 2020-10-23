#!/bin/bash

for food_ratio in 0.0 0.5 0.75 0.85 0.9 0.95 0.99 1.0;

    name_folder = 'FoodRatio_'$food_ratio
    mkdir name_folder
    cd name_folder

    ## Name (? does it understand variables here ?)
    #$ -N Food_$name_folder

    ## Maximum Memory
    #$ -l h_vmem=8G

    ## Configure Logfiles
    #$ -j y
    #$ -o ./

    ## Number of tasks (6?)
    #$ -t 1-6

    ## Send information email
    #$ -m a
    #$ -M emanuele.panizon@uni-konstanz.de

    module load python

    cp ../learning_GROUP_FOOD_dOBS.py .
    python ./learning_GROUP_FOOD_dOBS.py $food_ratio > 'results_FoodRatio_'$food_ratio'.dat'
    cd ../
