#!/bin/bash

depths=$1
device=$2
log_folder=$3

datasets=("pima" "spectf" "german" "ionosphere" "messidor_features" "winequality-red" "winequality-white" "spambase" "credit_a" "fertility" "hepatitis" "megawatt1" "credit_default")

for dataset in ${datasets[@]}
do
  for i in `seq  1  $depths`
  do
    eval "sh ./run_main.sh $dataset $device ce $log_folder $i"
  done
done
