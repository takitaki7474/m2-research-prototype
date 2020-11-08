#!/bin/bash

LOOPS=20
MODEL_VERS=("v1" "v2" "v3" "v4" "v5" "v6" "v7" "v8" "v9" "v10" "v11" "v12" "v13" "v14" "v15" "v16" "v17" "v18" "v19" "v20")
BASE_RESULT_VERS=("None" "v1" "v2" "v3" "v4" "v5" "v6" "v7" "v8" "v9" "v10" "v11" "v12" "v13" "v14" "v15" "v16" "v17" "v18" "v19")
INI_TRAIN_NUM=40
ADD_TRAIN_NUM=40
INI_TEST_NUM=10
ADD_TEST_NUM=10
LEARNING_RATE=0.01
SEED=4
EPOCHS=100

eval_result=0
for ((i=0; i<LOOPS; i++))
do
  echo -e "\nstating ..."


  if [ $i -eq 0 ]; then
    params="-mn ${MODEL_VERS[$i]} -tr $INI_TRAIN_NUM -te $INI_TEST_NUM -e $EPOCHS -lr $LEARNING_RATE -s $SEED"
    echo ${params}
    python re_learning.py ${params}


  elif [ $i -eq 1 ]; then
    params="-mn ${MODEL_VERS[$i]} -bv ${BASE_RESULT_VERS[$i]} -tr $ADD_TRAIN_NUM -te $ADD_TEST_NUM -e $EPOCHS -lr $LEARNING_RATE -s $SEED"
    echo ${params}
    python re_learning.py ${params}


  else
    if [ $eval_result -eq 1 ]; then
      eval_result=0
    else
      eval_result=$(python comp_loss.py -mv1 ${BASE_RESULT_VERS[`expr $i - 1`]} -mv2 ${MODEL_VERS[`expr $i - 1`]})
    fi
    params="-mn ${MODEL_VERS[$i]} -bv ${BASE_RESULT_VERS[$i]} -tr $ADD_TRAIN_NUM -te $ADD_TEST_NUM -e $EPOCHS -lr $LEARNING_RATE -s $SEED -eval $eval_result"
    echo ${params}
    python re_learning.py ${params}
  fi


  echo -e "\n"
done
