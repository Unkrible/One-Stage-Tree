#!/bin/bash

device=$1
log_folder=$2

datasets=("bikeshare" "housing" "airfoil" "589" "620")

val_sizes=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4)

for dataset in ${datasets[@]}
do
  for val_size in ${val_sizes[@]}
  do
    ls $log_folder || mkdir $log_folder
    cur_time="`date +%Y-%m-%d-%H-%M-%S`"
    log_file="$log_folder/$dataset#$val_size#$cur_time.log"
    eval "python ./main.py --dataset=$dataset --devices=$device --loss=mse --max_depth=5 --max_epochs=400 --val_size=$val_size --test_size=0.2 | tee -i $log_file"
  done
done

datasets=("pima" "spectf" "german" "ionosphere" "messidor_features" "winequality-red" "winequality-white" "spambase" "credit_a" "fertility" "hepatitis" "megawatt1" "credit_default")
for dataset in ${datasets[@]}
do
  for val_size in ${val_sizes[@]}
  do
    ls $log_folder || mkdir $log_folder
    cur_time="`date +%Y-%m-%d-%H-%M-%S`"
    log_file="$log_folder/$dataset#$val_size#$cur_time.log"
    eval "python ./main.py --dataset=$dataset --devices=$device --loss=ce --max_depth=5 --max_epochs=400 --val_size=$val_size --test_size=0.2 | tee -i $log_file"
  done
done
