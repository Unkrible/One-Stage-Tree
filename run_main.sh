#!/bin/bash

if [ ! $5 ]; then
  max_depth=6
else
  max_depth=$5
fi

if [ ! $6 ]; then
  patience=10
else
  patience=$6
fi

export PYTHONPATH=$PYTHONPATH:./

dataset=$1
device=$2
loss=$3
log_folder=$4

ls $log_folder || mkdir $log_folder


cur_time="`date +%Y-%m-%d-%H-%M-%S`"
log_file="$log_folder/$dataset#$max_depth#$cur_time.log"
eval "python ./main.py --dataset=$dataset --devices=$device --loss=$loss --max_depth=$max_depth --max_epochs=400 --patience=$patience --val_size=0.2 | tee -i $log_file"
