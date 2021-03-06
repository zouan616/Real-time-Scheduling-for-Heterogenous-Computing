#!/bin/bash
# run in sudo

function daemon() {
  size1=$(stat trace | grep Size)
  size2=null
  while true; do
    sleep 300
    size2=$(stat trace | grep Size)
    if [ "$size1" == "$size2" ]; then
      pkill -f ./scheduling_experiment
      echo killed >>trace
    fi
    size1=$size2
  done
}

function do_test() {
  u_start=$1
  u_end=$2
  for total_util in $(seq "$u_start" 10 "$u_end"); do
    echo total_util "$total_util" >>trace
    success_count=0
    for iteration in {1..100}; do
      echo iteration "$iteration" >>trace
      ./data_generator "$total_util"
      # cat pthreadData.dat >>pthreadData.db
      # sed -n "$iteration"p pthreadData.db >pthreadData.dat
      for prio in {0..119}; do
        echo "$prio" >>trace
        if ./scheduling_experiment "$prio"; then
          ((success_count += 1))
          echo success >>trace
          break
        fi
      done
    done
    echo u="$total_util" "$success_count" >>result
  done
}

echo >trace
nvcc data_generator.cu -o data_generator
nvcc scheduling_experiment.cu -o scheduling_experiment
daemon &
do_test 370 370
