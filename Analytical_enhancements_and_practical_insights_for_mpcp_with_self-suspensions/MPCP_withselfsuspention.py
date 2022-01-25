# -!- coding: utf-8 -!-
import math
from random import randint
import itertools


def execution_time(i, task_):
    return task_[i][0]


def critical_section_time_part(i, j, task_):
    return task_[i][j]


def critical_section_time(i, task_):
    result = 0
    for j in range(1, max_critical_section_num + 1):
        result += task_[i][j]
    return result


def wc_cpu_excution(i, task_):
    result = execution_time(i, task_)
    return result


def WCRT(k, task, T, R, zeta):
    if wcrt[k] != 0:
        return wcrt[k]
    # worst case response time
    result = wc_cpu_excution(k, task) + blocking_time(k, task, T, R, zeta)
    if k == 0:
        return result
    else:
        while 1:
            temp = 0
            for i in range(k):
                temp += math.ceil(((result+WCRT(i, task, T, R, zeta)-wc_cpu_excution(i, task))/T[i]))*wc_cpu_excution(i, task)
            temp += wc_cpu_excution(k, task) + blocking_time(k, task, T, R, zeta)
            if (result >= temp) or (result >= 10000):
                break
            else:
                result = result + 1
        wcrt[k] = result
        return result


def alpha(i, h, task_, t_):
    # upper bound on the number of instances of τh released during the execution of a single job of τi
    result = math.ceil(((wc_cpu_excution(i, task_)+WCRT(h, task_, t_, R, zeta)-wc_cpu_excution(h, task_))/t_[h]))
    return result


def H(j, x, task_, R, zeta):
    # The worst-case response time of the xth critical section of task j
    # result = task[j][x]+indirect_blocking_time_part(j,x)
    result = critical_section_time_part(j, x, task_) + indirect_blocking_time(j, task_, R, zeta)
    return result


def theta(i, l, task_):
    # defined as an upper bound on the number of instances of-
    # -a lower-priority task τl that may be active during the execution of τi
    result = math.ceil((execution_time(i, task_) + D[l] - execution_time(l, task_))/T[l])
    return result


def kth_longest_critical_section(i, k, task_):
    # find the kth_longest_critical_section for the task which has lower priority than taski
    heap = []
    for y in range(k):
        heap.append(0)
    for tasknum in range(i+1, n):
        for j in range(1, max_critical_section_num + 1):
            heap.append(critical_section_time_part(tasknum, j, task_))
    index = []
    for y in range(n*max_critical_section_num):
        index.append(y)
    heap, index = (list(t) for t in zip(*sorted(zip(heap, index))))
    return heap[len(heap) - k], index[len(heap) - k]


def direct_blocking_time_lp(i, task_):
    if i == (n-1):
        return 0
    bdr = 0
    for k in range(1, max_critical_section_num*n+1):
        num = 0
        s = 0
        kth_longest_cs, index = kth_longest_critical_section(i, k, task_)
        for t in range(0, k):
            index = t//2
            if R[index] == R[i]:
                s = s + num
                num = max(min(max_critical_section_num - s, theta(i, index, task_)), 0)
                'num = max(min(max_critical_section_num - sum,1),0)'
        bdr = bdr+num * kth_longest_cs
    return bdr


def direct_blocking_time(i, task, T, R, zeta):
    if directblocktime[i] != 0:
        return directblocktime[i]
    result = direct_blocking_time_lp(i, task)
    if i == 0:
        return result
    while 1:
        temp = 0
        timer = 0
        I = 0
        for k in range(i):
            if R[k] == R[i]:
                I += alpha(i, k, task, T) * wc_cpu_excution(k, task)
        for k in range(i):
            if R[k] == R[i]:
                b = math.ceil((result + WCRT(k, task, T, R, zeta) - wc_cpu_excution(k, task)) / T[k])
                a = math.ceil(((wc_cpu_excution(i, task) + I + result + WCRT(k, task, T, R, zeta) - wc_cpu_excution(k, task))/T[k]))
                delta = min(b, a)
                'upper-bounds the cumulative number of requests by τh to the locks accessed by critical sections of τi.'
                for j in range(1, max_critical_section_num + 1):
                    temp = temp + delta * H(k, j, task, R, zeta)
        if (result == temp) and (timer == 0):
            result += 1
        if (result >= temp + direct_blocking_time_lp(i, task)) or (timer > 100000):
            break
        else:
            result += 1
            timer += 1
    directblocktime[i] = result
    return result


def indirect_blocking_time(i, task_, R, zeta):
    excution_time_sum = 0
    for k in range(i):
        if R[i] != R[k]:
            excution_time_sum += task_[k][0]
    bir = excution_time_sum*(zeta + 1)
    return bir


def blocking_time(i, task, T, R, zeta):
    return direct_blocking_time(i, task, T, R, zeta) + max_critical_section_num*indirect_blocking_time(i, task, R, zeta)


def generator(c_max, utilization, n):
    task = [[0 for _ in range(max_critical_section_num + 1)] for _ in range(n)]
    T = [0 for _ in range(n)]
    D = [0 for _ in range(n)]
    U = [0 for _ in range(n)]
    R = [0 for _ in range(n)]
    # set utilization rate
    U_sum = 0
    for Ri in range(n):
        R[Ri] = Ri % 2
    for ui in range(n):
        U[ui] = randint(1, 10)
        U_sum += U[ui]
    resolution = U_sum / utilization
    for ui in range(n):
        U[ui] = U[ui] / resolution
        print("Utilization for task[", ui, "] is", U[ui])
    # set computation segments
    for i in range(n):
        task[i][0] = randint(1, c_max)
        # for j in range(m - 1):
        #     task[i][2 * j + 1] = randint(1, s_max)
    # set Period & Deadline
    for i in range(n):
        T[i] = int(task[i][0] / U[i])
        D[i] = T[i]
    # set suspension segments
    errflag = 0
    for i in range(n):
        # for j in range(m):
        #     task[i][2 * j] = randint(1, c_max)
        for j in range(1, max_critical_section_num + 1):
            s_max_temp = T[i] - task[i][0]
            lower_bound = max(int(0.01 * s_max_temp / max_critical_section_num), 1)
            upper_bound = int(0.1 * s_max_temp / max_critical_section_num)
            error_count = 0
            while lower_bound >= upper_bound:
                error_count += 1
                lower_bound = max(int(0.01 * s_max_temp / max_critical_section_num), 1)
                upper_bound = int(0.1 * s_max_temp / max_critical_section_num)
                if error_count >= 1000:
                    errflag = 1
                    break
            if errflag:
                task[i][j] = 0
            else:
                task[i][j] = randint(lower_bound, upper_bound)
    max_critical_section_time = 0
    min_deadline = 100000
    for i in range(n):
        min_deadline = min(min_deadline, D[i])
        for j in range(1, max_critical_section_num + 1):
            max_critical_section_time = max(max_critical_section_time,task[i][j])
    if min_deadline < max_critical_section_time:
        errflag = 1
    return task, D, R, errflag


def calc(task, D, n):
    taskcombine = []
    iter = itertools.permutations(task, n)
    taskcombine.append(list(iter))
    Dcombine = []
    iter = itertools.permutations(D, n)
    Dcombine.append(list(iter))
    Rcombine = []
    iter = itertools.permutations(R, n)
    Rcombine.append(list(iter))
    task_pass = 0
    for j_ in range(math.factorial(n)):
        for i in range(n):
            wcrt[i] = 0
            directblocktime[i] = 0
        task_ = taskcombine[0][j_]
        D_ = Dcombine[0][j_]
        R_ = Rcombine[0][j_]
        print("task:", task_)
        for k in range(n):
            print("Deadline of task", k, ":", D_[k])
        for i in range(n):
            if WCRT(i, task_, D_, R_, zeta) >= D_[i]:
                task_pass = 0
                break
            else:
                task_pass = 1
        if task_pass:
            print("Final result:pass")
            break
        else:
            print("Not pass, change the combination")
    if task_pass == 0:
        print("Final result:Not pass")
    return task_pass


count = 100
max_critical_section_num = 2  # the number of critical section
n = 5  # the number of task
zeta = 1  # Number of suspensions in a critical section
c_max = 10
utilization = 0.45
taskpass = 1
taskpassnum = 0
batch = 0

while batch < count:
    directblocktime = []
    wcrt = []
    for i_ in range(n):
        directblocktime.append(0)
        wcrt.append(0)
    print("batchnum=", batch + 1)
    task, D, R, error = generator(c_max, utilization, n)
    T = D
    if error:
        batch -= 1
    else:
        taskpass = calc(task, D, n)
        if taskpass:
            taskpassnum += 1
    batch += 1

print("tasknum =", count, "\npasstasknum =", taskpassnum)
print("Passed rate = ", 100 * taskpassnum / count, "%")
