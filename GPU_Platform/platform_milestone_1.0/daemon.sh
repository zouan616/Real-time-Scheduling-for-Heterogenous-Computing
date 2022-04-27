#!/bin/bash
# run in sudo

size1=$(stat trace | grep Size | sed 's/.*Size: \([0-9]*\) .*/\1/g')
size2=a
while true
do
  sleep 180
  size2=$(stat trace | grep Size | sed 's/.*Size: \(.*\) *Blocks.*/\1/g')
  if [ "$size1" == "$size2" ];
  then
      pid=$(ps -aux | grep scheduling_experiment | head -n 1 | sed 's/root *\([0-9]*\) .*/\1/g')
      kill $pid
      echo killed >> trace
  fi
  size1=$size2
done
