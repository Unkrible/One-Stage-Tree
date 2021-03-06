#!/bin/bash

device=$2
script=$1
log_folder=$3

if [ ! $5 ]; then
  max_depth=6
else
  max_depth=$4
fi

#datasets=("bikeshare" "housing" "airfoil" "618" "589" "616" "607" "620" "637" "586")
datasets=("bikeshare" "housing" "airfoil" "589" "620" "586")

for dataset in ${datasets[@]}
do
  eval "sh ./run_$script.sh $dataset $device mse $log_folder $max_depth 70"
done
