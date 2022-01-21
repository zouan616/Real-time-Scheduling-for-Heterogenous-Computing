# -!- coding: utf-8 -!-
import math
from random import randint


def execution_time(i, task):
    return task[i][0]

def critical_section_time_part(i,j,task):
    return task[i][j]

def critical_section_time(i,task):
    result = 0
    for j in range(1, max_critical_section_num + 1):
        result += task[i][j]
    return result

def wc_CPU_excution(i,task):
    result = critical_section_time(i,task) + execution_time(i,task)
    return result

def WCRT(k,task):
    if wcrt[k] != 0:
        return wcrt[k]
    ## worst case response time
    result = wc_CPU_excution(k,task) + blocking_time(k,task)
    if k == 0:
        return result
    else:
        while 1:
            temp = 0
            for i in range(k):
                temp += math.ceil(((result+WCRT(i,task)-wc_CPU_excution(i,task))/T[i]))*wc_CPU_excution(i,task)
            temp += wc_CPU_excution(k,task) + blocking_time(k,task)
            if (result >= temp) or (result >= 10000):
                break
            else:
                result = result + 1
        wcrt[k] = result
        return result



def alpha(i, h,task):
#upper bound on the number of instances of τh released during the execution of a single job of τi
##  result = math.ceil(((WCRT(i)+WCRT(h)-wc_CPU_excution(h))/T(h)))
    result = math.ceil(((wc_CPU_excution(i,task)+WCRT(h,task)-wc_CPU_excution(h,task))/T[h]))
    return result

"""
def b(i,h):
##upper bound on the number of activations of τh during the blocking duration
    result = math.ceil((direct_blocking_time(i)+WCRT(h)-wc_CPU_excution(h))/T(h))
    return result
"""

def H(j,x,task):
##The worst-case response time of the xth critical section of task j
##    result = task[j][x]+indirect_blocking_time_part(j,x)
    result = critical_section_time_part(j,x,task)
    return result


def theta(i,l,task):
##defined as an upper bound on the number of instances of a lower-priority task τl that may be active during the execution of τi
    result = math.ceil((execution_time(i,task) + D[l] - execution_time(l,task))/T[l])
    return result


def kth_longest_critical_section(i,k,task):
    ##find the kth_longest_critical_section for the task which has lower priority than taski
    heap = []
    for y in range(k):
        heap.append(0)
    for tasknum in range(i+1,n):
        for j in range(1, max_critical_section_num + 1):
            heap.append(critical_section_time_part(tasknum,j,task))
    index = []
    for y in range(n*max_critical_section_num):
        index.append(y)
    heap, index = (list(t) for t in zip(*sorted(zip(heap, index))))
    return heap[len(heap) - k], index[len(heap) - k]

"""
def direct_blocking_time_hp(i,h):
    result = 0
    b = math.ceil((direct_blocking_time(i)+WCRT(h)-wc_CPU_excution(h))/T(h))
    delta = min(b,a(i,h)) ##upper-bounds the cumulative number of requests by τh to the locks accessed by critical sections of τi.
    for j in range(1,max_critical_section_num+1):
        result = result + delta*H(h,j)
    return result
"""

def direct_blocking_time_lp(i,task):
    if(i == (n-1)):
        return 0
    bdr = 0
    for k in range(1,max_critical_section_num*n+1):
        num = 0
        sum = 0
        kth_longest_cs, index = kth_longest_critical_section(i,k,task)
        for t in range(0,k):
            index = t//2
            sum = sum +num
            num = max(min(max_critical_section_num - sum, theta(i,index,task)), 0)
            'num = max(min(max_critical_section_num - sum,1),0)'
        bdr = bdr+num * kth_longest_cs
    return bdr
"""
def indirect_blocking_time_part(i,j):
    return 0

def prioritized_blocking_time_part(i,j):
    return 0
"""

def direct_blocking_time(i,task):
    if(directblocktime[i] != 0):
        return directblocktime[i]
    result = direct_blocking_time_lp(i,task)
    if i == 0:
        return result
    while 1:
        temp = 0
        timer = 0
        I = 0
        for k in range(i):
            I += alpha(i,k,task) * wc_CPU_excution(k,task)
        for k in range(i):
            b = math.ceil((result + WCRT(k,task) - wc_CPU_excution(k,task)) / T[k])
            a = math.ceil(((wc_CPU_excution(i,task) + I + result + WCRT(k,task) - wc_CPU_excution(k,task)) / T[k]))
            delta = min(b, a)
            'upper-bounds the cumulative number of requests by τh to the locks accessed by critical sections of τi.'
            for j in range(1, max_critical_section_num + 1):
                temp = temp + delta * H(k, j,task)
        if (result == temp) and (timer == 0):
            result += 1
        if (result >= temp + direct_blocking_time_lp(i,task)) or (timer > 10000):
            break
        else:
            result += 1
            timer += 1
    directblocktime[i] = result
    return result

def blocking_time(i,task):
    return direct_blocking_time(i,task)


count = 100
max_critical_section_num = 2
n = 3
c_max = 200
utilization = 0.05
taskpass = 1
taskpassnum = 0
batch = 0

while batch < count:
    directblocktime = []
    wcrt = []
    print("batchnum=",batch + 1)
    for i_ in range(n):
        directblocktime.append(0)
        wcrt.append(0)
    task = [[0 for _ in range(max_critical_section_num + 1)] for _ in range(n)]
    M = [max_critical_section_num for _ in range(n)]
    T = [0 for _ in range(n)]
    D = [0 for _ in range(n)]
    U = [0 for _ in range(n)]
    # set utilization rate
    U_sum = 0
    for ui in range(n):
        U[ui] = randint(1, 10)
        U_sum += U[ui]
    resolution = U_sum / utilization
    for ui in range(n):
        U[ui] = U[ui] / resolution
        print("Utilization for task[", ui, "] is", U[ui])
    # set computation segments
    for i_ in range(n):
        task[i_][0] = randint(1, c_max)
        # for j in range(m - 1):
        #     task[i][2 * j + 1] = randint(1, s_max)
    # set Period & Deadline
    for i_ in range(n):
        T[i_] = int(task[i_][0] / U[i_])
        D[i_] = T[i_]
        print("Deadline of task",i_,":",D[i_])
    # set suspension segments
    error = 0
    for i_ in range(n):
        # for j in range(m):
        #     task[i][2 * j] = randint(1, c_max)
        for j in range(1, max_critical_section_num + 1):
            s_max_temp = T[i_] - task[i_][0]
            lower_bound = max(int(0.01 * s_max_temp / max_critical_section_num), 1)
            upper_bound = int(0.1 * s_max_temp / max_critical_section_num)
            error_count = 0
            while lower_bound >= upper_bound:
                error_count += 1
                lower_bound = max(int(0.01 * s_max_temp / max_critical_section_num), 1)
                upper_bound = int(0.1 * s_max_temp / max_critical_section_num)
                if error_count >= 1000:
                    error = 1
                    break
            task[i_][j] = randint(lower_bound, upper_bound)
    if error: batch -= 1
    else:
        for i in range(n):
            print("blocking time of task", i, ":", blocking_time(i,task))
        for i in range(n):
            print("worst case response time of task:", i, ":", WCRT(i,task))
        ##print(task)
        for i_ in range(n):
            if WCRT(i_,task) >= D[i_]:
                print("Not pass")
                taskpass = 0
                break
            else:
                taskpass = 1
        if taskpass:
            print("pass")
            taskpassnum += 1
        batch += 1

print("tasknum =",count,"\npasstasknum =",taskpassnum)
print("Passed rate = ", 100 * taskpassnum / count, "%")

"""
    with open("input.txt", "w") as fp:
        fp.write(str(n) + "\n")
        fp.write(str(M[0]))
        for _ in range(1, n):
            fp.write(" " + str(M[_]))
        fp.write("\n")
        fp.write(str(T[0]))
        for _ in range(1, n):
            fp.write(" " + str(T[_]))
        fp.write("\n")
        fp.write(str(D[0]))
        for _ in range(1, n):
            fp.write(" " + str(D[_]))
        fp.write("\n")
        for i in range(n):
            for j in range(2 * max_critical_section_num - 2):
                fp.write(str(task[i][j]) + " ")
            fp.write(str(task[i][2 * max_critical_section_num - 2]) + "\n")
    fp.close()
"""



