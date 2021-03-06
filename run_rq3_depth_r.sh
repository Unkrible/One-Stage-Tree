#!/bin/bash

depths=$1
device=$2
log_folder=$3

datasets=("bikeshare" "housing" "airfoil" "589" "620" "586")

for dataset in ${datasets[@]}
do
  for i in `seq  1  $depths`
  do
    eval "sh ./run_main.sh $dataset $device mse $log_folder $i"
  done
done
