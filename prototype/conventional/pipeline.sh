#!/bin/bash

LOOPS=8
MODEL_NAMES=("v9" "v10" "v11" "v12" "v13" "v14" "v15" "v16")
DT_INDEXES_VERS=("None" "v9" "v10" "v11" "v12" "v13" "v14" "v15")
INI_TRAIN_NUM=300
ADD_TRAIN_NUM=300
INI_TEST_NUM=30
ADD_TEST_NUM=30
EPOCHS=200

for ((i=0; i<LOOPS; i++))
do
  echo -e "\nstating ..."
  if [ $i -eq 0 ]
  then
    params="-mn ${MODEL_NAMES[$i]} -tr ${INI_TRAIN_NUM} -te ${INI_TEST_NUM} -e $EPOCHS"
    echo ${params}
    python main.py ${params}
  else
    params="-mn ${MODEL_NAMES[$i]} -dt ${DT_INDEXES_VERS[$i]} -tr ${ADD_TRAIN_NUM} -te ${ADD_TEST_NUM} -e $EPOCHS"
    echo ${params}
    python main.py ${params}
  fi
  echo -e "\n"
done
