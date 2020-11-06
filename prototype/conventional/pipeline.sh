#!/bin/bash

LOOPS=15
MODEL_NAMES=("v1" "v2" "v3" "v4" "v5" "v6" "v7" "v8" "v9" "v10" "v11" "v12" "v13" "v14" "v15")
DT_INDEXES_VERS=("None" "v1" "v2" "v3" "v4" "v5" "v6" "v7" "v8" "v9" "v10" "v11" "v12" "v13" "v14")
INI_TRAIN_NUM=40
ADD_TRAIN_NUM=40
INI_TEST_NUM=10
ADD_TEST_NUM=10
LEARNING_RATE=0.01
SEED=1
EPOCHS=100

for ((i=0; i<LOOPS; i++))
do
  echo -e "\nstating ..."
  if [ $i -eq 0 ]
  then
    params="-mn ${MODEL_NAMES[$i]} -tr ${INI_TRAIN_NUM} -te ${INI_TEST_NUM} -e $EPOCHS -lr $LEARNING_RATE -s $SEED"
    echo ${params}
    python re_learning.py ${params}
  else
    params="-mn ${MODEL_NAMES[$i]} -bv ${DT_INDEXES_VERS[$i]} -tr ${ADD_TRAIN_NUM} -te ${ADD_TEST_NUM} -e $EPOCHS -lr $LEARNING_RATE -s $SEED"
    echo ${params}
    python re_learning.py ${params}
  fi
  echo -e "\n"
done
