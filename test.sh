iters=(250 500 750 1000 1250 1500 1750 2000)
ps=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
for p in "${ps[@]}"
do
    for iter in "${iters[@]}"
    do
        if [ $iter -eq 250 ]; then
            ./train_and_run.sh rare_difficulties 1000 $p $iter
        else
            ./train_and_run.sh rare_difficulties 1000 $p $iter resume
        fi
    done
done