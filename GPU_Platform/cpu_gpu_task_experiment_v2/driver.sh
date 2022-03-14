#!bin/bash
echo "" > out
echo "" > trace
echo "" > result1
echo "" > result2
echo "" > result3
for level in {1..3}
do
    for util in {0.4..170..2}
    do
        for i in {1..100}
        do
            echo level "$level" util "$util" i "$i" >> trace
            ./a.out "$util" "$level" >> out
            sleep 1
        done
        grep -o "Thread" out | wc -l >> result"$level"
        echo "" > out
    done
done
