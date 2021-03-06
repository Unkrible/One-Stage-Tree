#!/bin/bash

device=$1
log_folder=$2
max_depth=5

datasets=("german" "pima" "spambase" "magic04" "yeast" "ecoli" "glass")

ls $log_folder || mkdir $log_folder
dataset="housing"
cur_time="`date +%Y-%m-%d-%H-%M-%S`"
log_file="$log_folder/$dataset#$max_depth#$cur_time.log"
eval "python ./main.py --dataset=$dataset --devices=$device --loss=mse --max_depth=$max_depth --patience=50 --max_epochs=400 --val_size=0.1 --test_size=0.3333333 | tee -i $log_file"

for dataset in ${datasets[@]}
do
  ls $log_folder || mkdir $log_folder
  cur_time="`date +%Y-%m-%d-%H-%M-%S`"
  log_file="$log_folder/$dataset#$max_depth#$cur_time.log"
  eval "python ./main.py --dataset=$dataset --devices=$device --loss=ce --max_depth=$max_depth --max_epochs=400 --val_size=0.1 --test_size=0.3333333 | tee -i $log_file"
done
