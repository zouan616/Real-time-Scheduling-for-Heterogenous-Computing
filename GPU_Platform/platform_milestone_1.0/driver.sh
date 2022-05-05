#!/bin/bash
# run in sudo

function daemon() {
  size1=$(stat trace | grep Size)
  size2=null
  while true; do
    sleep 150
    size2=$(stat trace | grep Size)
    if [ "$size1" == "$size2" ]; then
      pkill -f ./scheduling_experiment
      echo killed >>trace
    fi
    size1=$size2
  done
}

function do_test() {
  level=$1
  scale=$2
  u_start=$3
  u_end=$4
  echo level $level scale $scale >>trace
  for total_util in $(seq $u_start 10 $u_end); do
    echo total_util $total_util >>trace
    success_count=0
    for iteration in {1..100}; do
      echo iteration $iteration >>trace
      ./data_generator $total_util $level $scale
      for prio in {0..119}; do
        echo $prio >>trace
        ./scheduling_experiment $prio
        if [ $? -eq 0 ]; then
          let success_count+=1
          echo success >>trace
          break
        fi
      done
    done
    echo u=$total_util $success_count >>result$level$scale
  done
}

echo >trace
nvcc data_generator.cu -o data_generator
nvcc scheduling_experiment.cu -o scheduling_experiment
daemon &
do_test 3 3 180 180
