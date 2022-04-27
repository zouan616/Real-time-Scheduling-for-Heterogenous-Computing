#!/bin/bash
# run in sudo

echo > trace

nvcc data_generator.cu -o data_generator
nvcc scheduling_experiment.cu -o scheduling_experiment

level=1
scale=2
echo level "$level" scale "$scale" >> trace
for total_util in {130..190..10}
do
    echo total_util "$total_util" >> trace
    success_count=0
    for iteration in {1..100}
    do
        echo iteration "$iteration" >> trace
        ./data_generator "$total_util" "$level" "$scale"
        for prio in {0..119}
        do
            echo "$prio" >> trace
            ./scheduling_experiment "$prio" >> trace
            if [ "$?" == "0" ];
            then
                let success_count+=1
                echo success >> trace
                break
            fi
        done
    done
    echo "$success_count" >> result12
done
