#!/bin/bash

# sizes=(1000 5000 10000 50000 100000 500000)
sizes=(100000)
# ps=(0.3 0.5 0.7 1.0)
# ps=(1.0 0.7 0.5 0.3)
ps=(1.0)
for p in "${ps[@]}"
do
    for size in "${sizes[@]}"
    do
        ./train_and_run.sh rare_history $size $p 5000
    done
done