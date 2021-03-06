#!/bin/bash

device=$2
script=$1
log_folder=$3

datasets=("pima" "spectf" "german" "ionosphere" "messidor_features" "winequality-red" "winequality-white" "spambase" "credit_a" "fertility" "hepatitis" "megawatt1" "credit_default")

for dataset in ${datasets[@]}
do
  eval "sh ./run_$script.sh $dataset $device ce $log_folder"
done
