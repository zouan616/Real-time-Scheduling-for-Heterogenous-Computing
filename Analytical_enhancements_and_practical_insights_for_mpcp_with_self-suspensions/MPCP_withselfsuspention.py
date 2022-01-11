# -!- coding: utf-8 -!-
import math
def execution_time(i):
    return task[i][0]

def critical_section_time(i):
    result = 0
    for j in range(1, max_critical_section_num + 1):
        result += task[i][j]
    return result

def wc_CPU_excution(i):
    result = critical_section_time(i) + execution_time(i)
    return result

def WCRT(k):
    ## worst case response time
    result = wc_CPU_excution(k) + blocking_time(k)
    tasknum = k
    if k == 0:
        return result
    else:
        for i in range(k):
          result = result + I(tasknum,i)
    return result

def D(i):
    if i == 0:
        return 102
    elif i == 1:
        return 10000
    else:
        return 1106

def T(i):
    return D(i)

def a(i,h):
##upper bound on the number of instances of τh released during the execution of a single job of τi
##  result = math.ceil(((WCRT(i)+WCRT(h)-wc_CPU_excution(h))/T(h)))
    result = math.ceil(((wc_CPU_excution(i)+WCRT(h)-wc_CPU_excution(h))/T(h)))
    return result

def I(i,h):
##maximum preemption delay from each higher-priority task
    result = a(i,h)*wc_CPU_excution(h)
    return result


"""
def b(i,h):
##upper bound on the number of activations of τh during the blocking duration
    result = math.ceil((direct_blocking_time(i)+WCRT(h)-wc_CPU_excution(h))/T(h))
    return result
"""

def H(j,x):
##The worst-case response time of the xth critical section of task j
##    result = task[j][x]+indirect_blocking_time_part(j,x)
    result = task[j][x]
    return result


def theta(i,l):
##defined as an upper bound on the number of instances of a lower-priority task τl that may be active during the execution of τi
    result = math.ceil((WCRT(i) + D(l) - execution_time(l))/T(l))
    return result


def kth_longest_critical_section(i,k):
    ##find the kth_longest_critical_section for the task which has lower priority than taski
    heap = []
    for y in range(k):
        heap.append(0)
    for tasknum in range(i+1,n):
        for j in range(1, max_critical_section_num + 1):
            heap.append(task[tasknum][j])
    heap.sort()
    return heap[len(heap) - k]

"""
def direct_blocking_time_hp(i,h):
    result = 0
    b = math.ceil((direct_blocking_time(i)+WCRT(h)-wc_CPU_excution(h))/T(h))
    delta = min(b,a(i,h)) ##upper-bounds the cumulative number of requests by τh to the locks accessed by critical sections of τi.
    for j in range(1,max_critical_section_num+1):
        result = result + delta*H(h,j)
    return result
"""

def direct_blocking_time_lp(i):
    if(i == (n-1)):
        return 0
    bdr = 0
    for k in range(1,max_critical_section_num*n+1):
        num = 0
        sum = 0
        for t in range(1,k+1):
            sum = sum +num
            ## num = min(max_critical_section_num - sum,theta(i,t))
            num = max(min(critical_section_num[i] - sum,1),0)
        bdr = bdr+num*kth_longest_critical_section(i,k)
    return bdr
"""
def indirect_blocking_time_part(i,j):
    return 0

def prioritized_blocking_time_part(i,j):
    return 0
"""

def direct_blocking_time(i):
    result = direct_blocking_time_lp(i)
    if i == 0:
        return result
    while 1:
        temp = 0
        timer = 0
        for k in range(i):
            b = math.ceil((result + WCRT(k) - wc_CPU_excution(k)) / T(k))
            delta = min(b, a(i,k))  ##upper-bounds the cumulative number of requests by τh to the locks accessed by critical sections of τi.
            for j in range(1, max_critical_section_num + 1):
                temp = temp + delta * H(k, j)
        if (result == temp) and (timer == 0):
            result += 1
        if (result >= temp) or (timer > 10000):
            break
        else:
            result += 1
            timer += 1
    return result

def blocking_time(i):
    return direct_blocking_time(i)


max_critical_section_num = 2
n = 3

task = [[0 for i in range(2 * max_critical_section_num - 1)] for j in range(n)]
task[0][0] = 1
task[0][1] = 1
task[0][2] = 0

task[1][0] = 1
task[1][1] = 100
task[1][2] = 0

task[2][0] = 1000
task[2][1] = 1
task[2][2] = 1

critical_section_num = [1, 1, 2]


print("blocking time of task1:",blocking_time(0))
print("blocking time of task2:",blocking_time(1))
print("blocking time of task3:",blocking_time(2))

print("worst case response time of task1:",WCRT(0))
print("worst case response time of task2:",WCRT(1))
print("worst case response time of task3:",WCRT(2))
