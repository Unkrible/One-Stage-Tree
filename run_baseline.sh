#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./

dataset=$1
device=$2
loss=$3
log_folder=$4

ls $log_folder || mkdir $log_folder


cur_time="`date +%Y-%m-%d-%H-%M-%S`"
log_file="$log_folder/$dataset#$cur_time.log"
eval "python ./baseline.py --dataset=$dataset --devices=$device --loss=$loss --max_epochs=400 --val_size=0.0 | tee -i $log_file"
