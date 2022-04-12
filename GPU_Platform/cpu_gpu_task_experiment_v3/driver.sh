#!/bin/bash
echo > trace

nvcc data_generator.cu -o data_generator
nvcc scheduling_experiment.cu -o scheduling_experiment

level=3
scale=1
echo level "$level" scale "$scale" >> trace
for total_util in {10..50..10} 
do
    echo total_util "$total_util" >> trace
    success_count=0
    for iteration in {1..50}
    do
        echo iteration "$iteration" >> trace
        ./data_generator "$total_util" "$level" "$scale"
        for prio in {0..119}
        do
            sudo ./scheduling_experiment "$prio" >> trace
            if [ "$?" == "0" ];
            then
                let success_count+=1
                echo success >> trace
                break
            fi
        done
    done
    echo "$success_count" >> result31
done

# level=2
# scale=1
# echo level "$level" scale "$scale" >> trace
# for total_util in {200..200..10} 
# do
#     echo total_util "$total_util" >> trace
#     success_count=0
#     for iteration in {1..50}
#     do
#         echo iteration "$iteration" >> trace
#         ./data_generator "$total_util" "$level" "$scale"
#         for prio in {0..119}
#         do
#             sudo ./scheduling_experiment "$prio" >> /dev/null
#             if [ "$?" == "0" ];
#             then
#                 let success_count+=1
#                 echo success "$success_count" >> trace
#                 break
#             fi
#         done
#     done
#     echo "$success_count" >> result21
# done

# level=3
# scale=1
# echo level "$level" scale "$scale" >> trace
# for total_util in {40..50..10} 
# do
#     echo total_util "$total_util" >> trace
#     success_count=0
#     for iteration in {1..38}
#     do
#         echo iteration "$iteration" >> trace
#         ./data_generator "$total_util" "$level" "$scale"
#         for prio in {0..119}
#         do
#             sudo ./scheduling_experiment "$prio" >> sth
#             if [ "$?" == "0" ];
#             then
#                 let success_count+=1
#                 echo success >> trace
#                 break
#             fi
#         done
#     done
#     echo "$success_count" >> result31
# done

# level=3
# scale=2
# echo level "$level" scale "$scale" >> trace
# for total_util in {10..50..10} 
# do
#     echo total_util "$total_util" >> trace
#     success_count=0
#     for iteration in {1..50}
#     do
#         echo iteration "$iteration" >> trace
#         ./data_generator "$total_util" "$level" "$scale"
#         for prio in {0..119}
#         do
#             sudo ./scheduling_experiment "$prio" >> /dev/null
#             if [ "$?" == "0" ];
#             then
#                 let success_count+=1
#                 echo success >> trace
#                 break
#             fi
#         done
#     done
#     echo "$success_count" >> result32
# done

level=2
scale=2
echo level "$level" scale "$scale" >> trace
for total_util in {100..150..10} 
do
    echo total_util "$total_util" >> trace
    success_count=0
    for iteration in {1..50}
    do
        echo iteration "$iteration" >> trace
        ./data_generator "$total_util" "$level" "$scale"
        for prio in {0..119}
        do
            sudo ./scheduling_experiment "$prio" >> trace
            if [ "$?" == "0" ];
            then
                let success_count+=1
                echo success >> trace
                break
            fi
        done
    done
    echo "$success_count" >> result22
done

level=1
scale=2
echo level "$level" scale "$scale" >> trace
for total_util in {50..150..10} 
do
    echo total_util "$total_util" >> trace
    success_count=0
    for iteration in {1..50}
    do
        echo iteration "$iteration" >> trace
        ./data_generator "$total_util" "$level" "$scale"
        for prio in {0..119}
        do
            sudo ./scheduling_experiment "$prio" >> trace
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

# level=3
# scale=3
# echo level "$level" scale "$scale" >> trace
# for total_util in {10..50..10} 
# do
#     echo total_util "$total_util" >> trace
#     success_count=0
#     for iteration in {1..50}
#     do
#         echo iteration "$iteration" >> trace
#         ./data_generator "$total_util" "$level" "$scale"
#         for prio in {0..119}
#         do
#             sudo ./scheduling_experiment "$prio" >> /dev/null
#             if [ "$?" == "0" ];
#             then
#                 let success_count+=1
#                 echo success >> trace
#                 break
#             fi
#         done
#     done
#     echo "$success_count" >> result33
# done

level=2
scale=3
echo level "$level" scale "$scale" >> trace
for total_util in {30..130..10} 
do
    echo total_util "$total_util" >> trace
    success_count=0
    for iteration in {1..50}
    do
        echo iteration "$iteration" >> trace
        ./data_generator "$total_util" "$level" "$scale"
        for prio in {0..119}
        do
            sudo ./scheduling_experiment "$prio" >> trace
            if [ "$?" == "0" ];
            then
                let success_count+=1
                echo success >> trace
                break
            fi
        done
    done
    echo "$success_count" >> result23
done

level=1
scale=3
echo level "$level" scale "$scale" >> trace
for total_util in {30..130..10} 
do
    echo total_util "$total_util" >> trace
    success_count=0
    for iteration in {1..50}
    do
        echo iteration "$iteration" >> trace
        ./data_generator "$total_util" "$level" "$scale"
        for prio in {0..119}
        do
            sudo ./scheduling_experiment "$prio" >> trace
            if [ "$?" == "0" ];
            then
                let success_count+=1
                echo success >> trace
                break
            fi
        done
    done
    echo "$success_count" >> result13
done
