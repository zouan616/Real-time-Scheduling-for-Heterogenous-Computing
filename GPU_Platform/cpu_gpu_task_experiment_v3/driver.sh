#!/bin/bash
echo > trace
echo u0 d0 c00 c01 c02 ct0 g00 g01 gt0 prio0 u1 d1 c10 c11 c12 ct1 g10 g11 gt1 prio1 u2 d2 c20 c21 c02 ct2 g20 g21 gt2 prio2 u3 d3 c30 c31 c32 ct3 g30 g31 gt3 prio3 u4 d4 c40 c41 c42 ct4 g40 g41 gt4 prio4 lv ut > result_verbose
echo > result1
echo > result2
echo > result3

nvcc data_generator.cu -o data_generator
nvcc scheduling_experiment.cu -o scheduling_experiment

level=1
echo level "$level" >> trace
for total_util in {10..200..10} 
do
    echo total_util "$total_util" >> trace
    success_count=0
    for iteration in {1..75}
    do
        echo iteration "$iteration" >> trace
        ./data_generator "$total_util" "$level"
        for prio in {0..119}
        do
            ./scheduling_experiment "$prio" >> result_verbose
            if [ "$?" == "0" ];
            then
                let success_count+=1
                echo "$level" "$total_util" >> result_verbose
                break
            fi
        done
    done
    echo "$success_count" >> result"$level"
done

level=2
echo level "$level" >> trace
for total_util in {10..200..10} 
do
    echo total_util "$total_util" >> trace
    success_count=0
    for iteration in {1..75}
    do
        echo iteration "$iteration" >> trace
        ./data_generator "$total_util" "$level"
        for prio in {0..119}
        do
            ./scheduling_experiment "$prio" >> result_verbose
            if [ "$?" == "0" ];
            then
                let success_count+=1
                echo "$level" "$total_util" >> result_verbose
                break
            fi
        done
    done
    echo "$success_count" >> result"$level"2
done

level=3
echo level "$level" >> trace
for total_util in {10..200..10} 
do
    echo total_util "$total_util" >> trace
    success_count=0
    for iteration in {1..75}
    do
        echo iteration "$iteration" >> trace
        ./data_generator "$total_util" "$level"
        for prio in {0..119}
        do
            ./scheduling_experiment "$prio" >> result_verbose
            if [ "$?" == "0" ];
            then
                let success_count+=1
                echo "$level" "$total_util" >> result_verbose
                break
            fi
        done
    done
    echo "$success_count" >> result"$level"
done
