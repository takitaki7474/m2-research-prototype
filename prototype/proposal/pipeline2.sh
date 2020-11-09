#!/bin/bash

LOOPS=5
MODEL_VERS=("v21" "v22" "v23" "v24" "v25")
BASE_RESULT_VERS=("v20" "v21" "v22" "v23" "v24")
INI_TRAIN_NUM=40
ADD_TRAIN_NUM=40
INI_TEST_NUM=10
ADD_TEST_NUM=10
LEARNING_RATE=0.01
SEED=0
EPOCHS=100

for ((i=0; i<LOOPS; i++))
do
  echo -e "\nstating ..."


  if [ $i -eq 0 ]; then
    params="-mn ${MODEL_VERS[$i]} -tr $INI_TRAIN_NUM -te $INI_TEST_NUM -e $EPOCHS -lr $LEARNING_RATE -s $SEED -eval 0"
  else
    eval_result=$(python comp_loss.py -mv1 ${BASE_RESULT_VERS[$i]})
    params="-mn ${MODEL_VERS[$i]} -bv ${BASE_RESULT_VERS[$i]} -tr $ADD_TRAIN_NUM -te $ADD_TEST_NUM -e $EPOCHS -lr $LEARNING_RATE -s $SEED -eval $eval_result"
  fi

  
  echo ${params}
  python re_learning.py ${params}
  echo -e "\n"
done
