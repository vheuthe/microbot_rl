#! /bin/bash

for food_ratio in 0.0 0.95 1.0; do
name_folder='FoodRatio_'$food_ratio
mkdir $name_folder
cd $name_folder
module load python

cp ../learning_GROUP_FOOD_dOBS.py .
python ./learning_GROUP_FOOD_dOBS.py $food_ratio > 'results_FoodRatio_'$food_ratio'.dat'

cd ../
done
