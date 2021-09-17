#!/bin/bash
time=$(date "+%Y-%m-%d-%H-%M-%S")

for num in 5 10;do
{
nohup python -u main.py --_balance True --client_num $num --cluster_num 2 --_running_time $time > running1.out 2>&1 &
nohup python -u main.py --_balance True --client_num $num --cluster_num 4 --_running_time $time > running2.out 2>&1 &
nohup python -u main.py --_balance True --client_num $num --cluster_num 6 --_running_time $time > running3.out 2>&1 &
wait
nohup python -u main.py --_balance False --client_num $num --cluster_num 2 --_running_time $time > running1.out 2>&1 &
nohup python -u main.py --_balance False --client_num $num --cluster_num 4 --_running_time $time > running2.out 2>&1 &
nohup python -u main.py --_balance False --client_num $num --cluster_num 6 --_running_time $time > running3.out 2>&1 &
wait
}
done


