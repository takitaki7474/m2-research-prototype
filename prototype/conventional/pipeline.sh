#!/bin/bash

LOOPS=2
MODEL_NAMES=("v1" "v2")
DT_INDEXES_VERS=("None" "v1")
TRAINS=(10 10)
TESTS=(10 10)
EPOCHS=10

for ((i=0; i<LOOPS; i++))
do
  echo -e "\nstating ..."
  if [ $i -eq 0 ]
  then
    params="-mn ${MODEL_NAMES[$i]} -tr ${TRAINS[$i]} -te ${TESTS[$i]} -e $EPOCHS"
    echo ${params}
    python main.py ${params}
  else
    params="-mn ${MODEL_NAMES[$i]} -dt ${DT_INDEXES_VERS[$i]} -tr ${TRAINS[$i]} -te ${TESTS[$i]} -e $EPOCHS"
    echo ${params}
    python main.py ${params}
  fi
  echo -e "\n"
done
