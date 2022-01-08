
# Note on the papar *Scheduling Security-Critical Real-Time Applications on Clusters*

## SAEDF algorithm
|meaning|symbol|
|:--|:--|
|ith Task|$T_i$|
|security overhead of ith task|$c_i$|
|security overhead of ith task, jth service|$c_i^j$|
|earliest start time of ith task|$es_i$|
|earliest start time of ith task on jth node|$es_i(N_j)$|
|arrival time of ith task|$a_i$|
|execution time of ith task|$e_i$|
|finish time of ith task|$f_i$|
|deadline of ith task|$d_i$|
|data amount of ith task|$l_i$|
|security level of ith task range|$S_i$|
|security level of ith task|$s_i$,$sl_i$|
|parameter for security level of ith task|$\mu_i$|
|weight of jth security service|$w_j$|
|jth node|$N_j$|
|whether ith task is accepted|$y_i$|


for each task $T_i$ on queue

for each node $N_j$ in cluster

$es_j(T_i) := eq12(T_i, N_j)$

$c_i^{min} := eq13(T_i)$

if ($T_i$ is feasible)

sort $\{w_i^a,w_i^c,w_i^g\}$ by $w_i^{v_1}<w_i^{v_2}<w_i^{v_3}$

for each service $\{a,c,g\}$, increase $s_i^{v_j}$ from lowest until (!property2)

$SL_i^j := eq1(s_i),y_i := 1$

else $SL_i^j := 0,y_i := 0$

endfor

complexity $O(knm)$

## Notes on algorithm

- **eq1** Safety level of a task is the sum of safety level of each service times its weight $SL(s_i) = \Sigma_{j=1}^{q}w_i^js_i^j$
- **eq12** Earliest start time of a tast is: the sum of execution time and overhead of all tasks with earlier deadlines, plus the current executing task. $es_j(T_i = r_j+\Sigma_{T_k \in N_j,d_k\leq d_i}(e_k+\Sigma_{l \in \{a,c,g\}}c_k^l)$
- **eq13** minimum security overhead of a task is the sum of overheads of insecure safety services. $c_i^{min} = \Sigma_{j \in \{a,c,g\}}min(S_i^j)$
- **Property2** The earliest start time, plus the execution time, plus all overhead, should not exceed the deadline. $es_j(T_i)+e_i+\Sigma_{j\in \{a,c,g\}}c_i^j \leq d_i$

## Simulation

Find real-world tasks, compare SAEDF with EDF,LLF,FCFS. 
- Fig5: compare different TBase with all security service; 
- Fig6,7,8: compare different TBase with a,c,g service;
- Fig9,10: compare different weight 
- Fig11: compare different CPU speed(change overhead?)
- Fig12: compare different node number
- Fig13: compare improvement to other strategies
- Fig14: change of data size
- 6.9 A certain situation to implement the model. Seems like very simplified.
## Questions
 
1. The ``Guarentee Ratio`` parameter did treat all tasks with different sizes as the same?
2. During simulation, the deadline base $\beta$ is determined to be all the same?
3. The legiticacy of equation6, and 8? $s_i^c=13.5/\mu_i^c,s_i^g=4.36/\mu_c^g$
4. $a,c,g$ weight purely objective? In Fig9.c should have compared different weight assignment?Overall System Performance is a function $f(a,c,g)$, probably can find an optimal weight assignment?
5. What is 6.8 SAREC?
