# -!- coding: utf-8 -!-
import math
from random import randint
from typing import List
import itertools


def suspension_time_part(
        task_: List[List[int]],
        i: int,
        j: int):
    """
    j-th suspension segment of task \tau_i

    :param task_: taskset
    :param i: task index
    :param j: segment index
    :return: suspension time (segment)
    """
    return task_[i][2 * j + 1]


def suspension_time(
        task_: List[List[int]],
        i: int,
        M_: List[int]):
    """
    Total suspension time of task \tau_i

    :param task_: taskset
    :param i: task index
    :param M_: CPU segment number
    :return: suspension time
    """
    result = 0
    for j in range(0, M_[i] - 1):
        result += suspension_time_part(task_, i, j)
    return result


def execution_time_part(
        task_: List[List[int]],
        i: int,
        j: int):
    """
    j-th execution segment of task \tau_i

    :param task_: taskset
    :param i: task index
    :param j: segment index
    :return: execution time (segment)
    """
    return task_[i][2 * j]


def execution_time(
        task_: List[List[int]],
        i: int,
        M_: List[int]):
    """
    Total execution time of task \tau_i

    :param task_: taskset
    :param i: task index
    :param M_: CPU segment number
    :return: execution time
    """
    result = 0
    for j in range(0, M_[i]):
        result += execution_time_part(task_, i, j)
    return result


def WCRT(k, task_, T_, R_, zeta_, M_, max_critSection_):
    if wcrt[k] != 0:
        # print("WCRT of task", k, "is recorded, call from WCRt()")
        return wcrt[k]
    blockTime = blocking_time(k, task_, T_, R_, zeta_, M_, max_critSection_)
    # print("total blocking time = ", blockTime)
    if blockTime == -1:
        return -1
    result = execution_time(task_, k, M) + suspension_time(task_, k, M) + blockTime
    # print("WCRT starting point = ", result)
    if k == 0:
        return result
    else:
        iteration_count = 0
        while 1:
            if iteration_count > 100000:
                print("WCRT timeout, iteration count > 100000")
                return -1
            temp = 0
            for h in range(k):
                temp += math.ceil(
                    (result + WCRT(h, task_, T_, R_, zeta_, M_, max_critSection_) - max_critSection_[h]) / (T[h])) * \
                        max_critSection_[h]
            if result >= temp:
                # print("Value of Sum_I = ", temp)
                # print("Currently, WCRT = ", result)
                break
            result += 1
            iteration_count += 1
        wcrt[k] = result
        return result


# def alpha(i, h, task_, t_):
#     # upper bound on the number of instances of τh released during the execution of a single job of τi
#     # result = math.ceil(((wc_cpu_excution(i, task_)+WCRT(h, task_, t_, R, zeta)-wc_cpu_excution(h, task_))/t_[h]))
#     result = math.ceil(((WCRT(i, task_, t_, R, zeta) + WCRT(h, task_, t_, R, zeta)-wc_cpu_excution(h, task_))/t_[h]))
#     print("Calculating alpha of ", i, h, "result =", result)
#     return result


# def theta(i, l, task_):
#     # defined as an upper bound on the number of instances of-
#     # -a lower-priority task τl that may be active during the execution of τi
#     result = math.ceil((execution_time(i, task_, M) + D[l] - execution_time(l, task_, M))/T[l])
#     return result


# def kth_longest_critical_section(i, k, task_):
#     # find the kth_longest_critical_section for the task which has lower priority than taski
#     heap = []
#     for y in range(k):
#         heap.append(0)
#     for tasknum in range(i+1, n):
#         for j in range(1, M[i] + 1):
#             heap.append(critical_section_time_part(tasknum, j, task_))
#     index = []
#     for y in range(n*(M[i])):
#         index.append(y)
#     heap, index = (list(t) for t in zip(*sorted(zip(heap, index))))
#     return heap[len(heap) - k], index[len(heap) - k]


# def direct_blocking_time_lp(i, task_):
#     if i == (n-1):
#         return 0
#     bdr = 0
#     for k in range(1, (M[i])*n+1):
#         num = 0
#         s = 0
#         kth_longest_cs, index = kth_longest_critical_section(i, k, task_)
#         for t in range(0, k):
#             index = t//M[i]
#             # index = t//2
#             if R[index] == R[i]:
#                 s = s + num
#                 num = max(min(M[i] - s, theta(i, index, task_)), 0)
#                 'num = max(min(max_critical_section_num - sum,1),0)'
#         bdr = bdr+num * kth_longest_cs
#     return bdr


def H_(
        l: int,
        k: int,
        R_: List[int],
        task_: List[List[int]],
        zeta_: int,
        max_critSection_: List[int]):
    """
    WCRT of k-th critical section of task \tau_l. Equation (14)
    Assumption: 10% of CPU execution is critical

    :param l: task index
    :param k: critical section index
    :param R_: lock
    :param task_: taskset
    :param zeta_: number of suspension in a critical section, default = 0
    :param max_critSection_: max critical section length
    :return: WCRT of k-th critical section of task \tau_l
    """

    temp = int(0.1 * execution_time_part(task_, l, k)) + suspension_time_part(task_, l, k)
    temp += indirect_blocking_time(l, R_, zeta_, max_critSection_)
    return temp


def direct_blocking_time_req_helper(
        i: int,
        R_: List[int],
        task_: List[List[int]],
        zeta_: int,
        T_: List[int],
        M_: List[int],
        max_critSection_: List[int]):
    """
    Helper function of Request-driven direct blocking time.
    Equation (5)

    :param i: task index
    :param R_: lock
    :param task_: taskset
    :param zeta_: number of suspension in a critical section, default = 0
    :param T_: period
    :param M_: CPU segment number
    :param max_critSection_: max critical section length
    :return: segment-level worst case blocking time
    """
    leftSum = 0
    max_H_lp = 0
    max_H_hp = 0

    # WCRT of critical section of all lp tasks
    for l in range(i, n):
        if R[l] == R[i]:
            for k in range(M_[l] - 1):
                temp = H_(l, k, R_, task_, zeta_, max_critSection_)
                if temp > max_H_lp:
                    max_H_lp = temp
    leftSum += max_H_lp
    rightSum = max_H_hp
    iteration_count = 0
    while 1:
        if iteration_count > 100000:
            print("Direct blocking request-approach timeout, iteration count > 100000")
            return -1
        for h in range(i):
            if R[h] == R[i]:
                wcrt_h = WCRT(h, task_, T_, R_, zeta_, M_, max_critSection_)
                beta = math.ceil((leftSum + wcrt_h - execution_time(task_, h, M_)) / (T_[h]))
                for k in range(M_[h] - 1):
                    temp = H_(h, k, R_, task_, zeta_, max_critSection_)
                    if temp > max_H_hp:
                        max_H_hp = temp
                rightSum += beta * max_H_hp
        if leftSum >= rightSum:
            break
        else:
            leftSum += 10
        iteration_count += 1
    return leftSum


def direct_blocking_time_req(
        i: int,
        task_: List[List[int]],
        T_: List[int],
        R_: List[int],
        zeta_: int,
        M_: List[int],
        max_critSection_: List[int]):
    """
    Request-Driven Approach

    :param i: task index
    :param task_: taskset
    :param T_: period
    :param R_: lock
    :param zeta_: number of suspension in a critical section, default=0
    :param M_: CPU segment number
    :param max_critSection_: max critical section
    :return: direct block time
    """

    if directblocktime[i] != 0:
        return directblocktime[i]
    # result = 0
    # for _ in range(M[i]):
    #     result += direct_blocking_time_req_helper(i, R_, task_, zeta_, T_)
    seg_blockingTime = direct_blocking_time_req_helper(i, R_, task_, zeta_, T_, M_, max_critSection_)
    if seg_blockingTime == -1:
        return -1
    result = M_[i] * seg_blockingTime
    return result


def direct_blocking_time_job(
        i: int,
        task_: List[List[int]],
        T_: List[int],
        R_: List[int],
        zeta_: int,
        max_critSection_: List[int],
        M_: List[int]):
    """
    Job-Driven Approach

    :param i: task index
    :param task_: taskset
    :param T_: period
    :param R_: lock
    :param zeta_: number of suspension in a critical section, default=0
    :param max_critSection_: max critical section
    :param M_: CPU segment number
    :return: direct block time
    """

    leftSum = 0
    temp = 0
    max_H_lp = 0
    max_H_hp = 0
    eta = max(int(0.1 * (M[i] - 1)), 1)
    for l in range(i, n):
        if R[l] == R[i]:
            for k in range(M_[l] - 1):
                temp = H_(l, k, R_, task_, zeta_, max_critSection_)
                if temp > max_H_lp:
                    max_H_lp = temp
    leftSum += max_H_lp * eta
    rightSum = max_H_lp * eta
    iteration_count = 0
    while 1:
        if iteration_count > 100000:
            print("direct blocking job-approach timeout, iteration count > 100000")
            return -1
        for h in range(i):
            wcrt_h = WCRT(h, task_, T_, R_, zeta_, M_, max_critSection_)
            alpha = math.ceil((leftSum + wcrt_h - execution_time(task_, h, M_)) / T[h])
            for k in range(M_[h] - 1):
                temp = H_(h, k, R_, task_, zeta_, max_critSection_)
                if temp > max_H_hp:
                    max_H_hp = temp
            rightSum += alpha * temp
        if leftSum >= rightSum:
            break
        else:
            leftSum += 10
        iteration_count += 1
    return leftSum


def direct_blocking_time_hybrid_hp(
        i: int,
        task_: List[List[int]],
        T_: List[int],
        R_: List[int],
        zeta_: int,
        max_critSection_: List[int],
        M_: List[int]):
    """
    Hybrid Approach -- hp

    :param i: task index
    :param task_: taskset
    :param T_: period
    :param R_: lock
    :param zeta_: number of suspension in a critical section, default=0
    :param max_critSection_: max critical section length
    :param M_: CPU segment number
    :return: direct block time of higher priority tasks on same CPU, under hybrid approach
    """
    result = int(0.9 * execution_time(task_, i, M)) + suspension_time(task_, i, M)
    max_H_hp = 0
    b_dmh = 0
    for h in range(i):
        alpha = math.ceil((result + WCRT(h, task_, T_, R_, zeta_, M_, max_critSection_) - max_critSection_[h]) / T[h])
        beta = math.ceil((result + WCRT(h, task_, T_, R_, zeta_, M_, max_critSection_) - max_critSection_[h]) / (T_[h]))
        delta = min(alpha, int(0.1 * M_[i]) * beta)
        for k in range(M_[h] - 1):
            temp = H_(h, k, R_, task_, zeta_, max_critSection_)
            if temp > max_H_hp:
                max_H_hp = temp
        b_dmh += delta * max_H_hp
    return b_dmh


def direct_blocking_time_hybrid_lp(i, task_, T_, R_, zeta_, max_critSection_, M_):
    """
    Hybrid Approach -- lp

    :param i: task index
    :param task_: taskset
    :param T_: period
    :param R_: lock
    :param zeta_: const
    :return: direct block time
    """
    b_dml = 0
    Q_ij = 0
    L_ijk = 0
    for j in range(n):
        if R[j] == R[i]:
            for l in range(i, n):
                if R[l] == R[j]:
                    for k in range(M_[l] - 1):
                        temp = H_(l, k, R_, task_, zeta_, max_critSection_)
                        if temp > Q_ij:
                            Q_ij = temp
                            L_ijk = max_critSection_[l]
            b_dml += Q_ij * L_ijk
    return b_dml


def direct_blocking_time_hybrid(i, task_, T_, R_, zeta_, max_critSection_, M_):
    """
    Hybrid Approach

    :param i: task index
    :param task_: taskset
    :param T_: period
    :param R_: lock
    :param zeta_: const
    :return: direct block time
    """
    result = direct_blocking_time_hybrid_lp(i, task_, T_, R_, zeta_, max_critSection_, M_) + \
             direct_blocking_time_hybrid_hp(i, task_, T_, R_, zeta_, max_critSection_, M_)
    return result


def indirect_blocking_time(
        j: int,
        R_: List[int],
        zeta_: int,
        max_critSection_: List[int]):
    """
    Indirect worst case blocking time incurred upon \tau_j.
    Counting the sum of max critical section length of all hp tasks on the same CPU.

    :param j: task index
    :param R_: lock
    :param zeta_: number of suspension in a critical section, default = 0
    :param max_critSection_: max critical section length
    :return: Worst case blocking time
    """
    sum_criticalSection = 0
    for q in range(j):
        if R_[j] == R_[q]:  # same CPU
            sum_criticalSection += max_critSection_[q]
    b_ir = sum_criticalSection * (zeta_ + 1)
    return b_ir


def prioritized_blocking_time_req(i, R_, max_critSection_):
    """
    Request-Driven Approach

    :param i: task index
    :param R_: lock
    :param max_critSection_: max critical section length
    :return: request-driven prioritized blocking time
    """
    sum_lpp_blocking = 0
    for l in range(i, n):
        if R_[i] == R_[l]:
            sum_lpp_blocking += max_critSection_[l]
    b_pr = sum_lpp_blocking * (M[i] + 1)
    return b_pr


# def prioritized_blocking_time_job(i, R_, max_critSection_):


def blocking_time(i, task_, T_, R_, zeta_, M_, max_critSection_):
    direct_B_req = direct_blocking_time_req(i, task_, T_, R_, zeta_, M_, max_critSection_)
    direct_B_job = direct_blocking_time_job(i, task_, T_, R_, zeta_, max_critSection_, M_)
    direct_B_hybrid = direct_blocking_time_hybrid(i, task_, T_, R_, zeta_, max_critSection_, M_)
    print("direct_B_req = ", direct_B_req)
    print("direct_B_job = ", direct_B_job)
    print("direct_B_hybrid = ", direct_B_hybrid)
    if direct_B_req == -1 and direct_B_job == -1 and direct_B_hybrid == -1:
        return -1
    direct_B = min(x for x in (direct_B_req, direct_B_job, direct_B_hybrid) if x > 0)
    prioritized_B = prioritized_blocking_time_req(i, R_, max_critSection_)
    result = direct_B + prioritized_B
    return result


def generator(utilization, n):
    task = [[0 for _ in range(203)] for _ in range(n)]
    T = [0 for _ in range(n)]
    D = [0 for _ in range(n)]
    U = [0 for _ in range(n)]
    R = [0 for _ in range(n)]

    # total CPU execution
    C_max_ = [0 for _ in range(n)]
    C_max_[0] = 4
    C_max_[1] = 15
    C_max_[2] = 238
    C_max_[3] = 350
    C_max_[4] = 25
    C_min = [0 for _ in range(n)]
    C_min[0] = 2
    C_min[1] = 13
    C_min[2] = 130
    C_min[3] = 150
    C_min[4] = 15
    S_max = [0 for _ in range(n)]
    S_max[0] = 25
    S_max[1] = 8
    S_max[2] = 1
    S_max[3] = 483
    S_max[4] = 10
    S_min = [0 for _ in range(n)]
    S_min[0] = 20
    S_min[1] = 5
    S_min[2] = 1
    S_min[3] = 482
    S_min[4] = 8
    max_critSection_value = [0 for _ in range(n)]

    # set utilization rate
    for Ri in range(n):
        R[Ri] = Ri % 2
    while 1:
        U_sum = 0
        flag_valid = 1
        for ui in range(n):
            U[ui] = randint(1, 10)
            U_sum += U[ui]
        resolution = U_sum / utilization
        for ui in range(n):
            U[ui] = U[ui] / resolution
            if U[ui] > 1:
                flag_valid = 0
        if flag_valid == 1:
            break
    for ui in range(n):
        print("Utilization for task[", ui, "] is", U[ui])
    # set computation segments
    for i in range(n):
        for j in range(M[i]):
            task[i][2 * j] = randint(C_min[i], C_max_[i])
            if int(0.1 * task[i][2 * j]) > max_critSection_value[i]:
                max_critSection_value[i] = int(0.1 * task[i][2 * j])
        for j in range(M[i] - 1):
            task[i][2 * j + 1] = randint(S_min[i], S_max[i])
    # set Period & Deadline
    for i in range(n):
        temp = execution_time(task, i, M) + suspension_time(task, i, M)
        T[i] = int(temp / U[i])
        D[i] = T[i]
    return task, D, R, max_critSection_value


def calc(task_, D_, T_, n_, R_, M_):
    row_permutations = list(itertools.permutations(task_))
    deadline_permutations = list(itertools.permutations(D_))
    period_permutations = list(itertools.permutations(T_))
    R_permutations = list(itertools.permutations(R_))
    M_permutations = list(itertools.permutations(M_))
    maxCritSection_permutations = list(itertools.permutations(max_critSection))
    task_pass = 0

    for idx, (permutation, deadline_permutation, period_permutation, R_permutation, M_permutation,
              maxCritSection_permutation) in enumerate(
            zip(row_permutations, deadline_permutations, period_permutations, R_permutations, M_permutations,
                maxCritSection_permutations)):
        NEW_task = list(permutation)
        # 输出新矩阵
        print("=================================================")
        print(f"Permutation {idx + 1}:")
        for row in NEW_task:
            print(row)
        # 输出其他矩阵的排列
        print("Deadline Permutation:", deadline_permutation)
        print("Period Permutation:", period_permutation)
        print("Lock Permutation:", R_permutation)
        print("Segment number Permutation:", M_permutation)

        task_pass = 0
        for i in range(n_):
            wcrt[i] = 0
            directblocktime[i] = 0
        for i in range(n_):
            WCRT_res = WCRT(i, NEW_task, deadline_permutation, R_permutation, zeta, M_permutation,
                            maxCritSection_permutation)
            if WCRT_res == -1:
                print("task ", i, "timeout")
                task_pass = 0
                break
            elif WCRT_res > deadline_permutation[i]:
                print("WCRT for task ", i, "= ", WCRT_res)
                print("Deadline = ", deadline_permutation[i])
                task_pass = 0
                break
            else:
                print("task ", i, "passed")
                task_pass = 1
        if task_pass:
            print("Final result:pass")
            break

    if task_pass == 0:
        print("Final result:Not pass")
    return task_pass
    # taskcombine = []
    # iter = itertools.permutations(task, n)
    # taskcombine.append(list(iter))
    # Dcombine = []
    # iter = itertools.permutations(D, n)
    # Dcombine.append(list(iter))
    # Rcombine = []
    # iter = itertools.permutations(R, n)
    # Rcombine.append(list(iter))
    # task_pass = 0
    # for j_ in range(math.factorial(n)):
    #     for i in range(n):
    #         wcrt[i] = 0
    #         directblocktime[i] = 0
    #     task_ = taskcombine[0][j_]
    #     D_ = Dcombine[0][j_]
    #     R_ = Rcombine[0][j_]
    #     print("task:", task_)
    #     for k in range(n):
    #         print("Deadline of task", k, ":", D_[k])
    #     for i in range(n):
    #         WCRT_res = WCRT(i, task_, D_, R_, zeta)
    #         if WCRT_res >= D_[i]:
    #             print("WCRT = ", WCRT_res)
    #             print("Deadline = ", D_[i])
    #             task_pass = 0
    #             break
    #         else:
    #             task_pass = 1
    #     if task_pass:
    #         print("Final result:pass")
    #         break
    #     # else:
    #         # print("Not pass, change the combination")
    # if task_pass == 0:
    #     print("Final result:Not pass")
    # return task_pass


count = 10
max_critical_section_num = 5  # the number of critical section
n = 5  # the number of task
zeta = 0  # Number of suspensions in a critical section
utilization = 0.2
taskpass = 1
taskpassnum = 0
batch = 0
M = [102 for _ in range(n)]
M[0] = 102
M[1] = 21
M[2] = 4
M[3] = 2
M[4] = 10
# max WCET of critical section of task i
max_critSection = [0 for _ in range(n)]

while batch < count:
    directblocktime = []
    wcrt = []
    for i_ in range(n):
        directblocktime.append(0)
        wcrt.append(0)
    print("batchnum=", batch + 1)
    task, D, R, max_critSection = generator(utilization, n)
    T = D
    taskpass = calc(task, D, T, n, R, M)
    if taskpass:
        taskpassnum += 1
    batch += 1

print("tasknum =", count, "\npasstasknum =", taskpassnum)
print("Passed rate = ", 100 * taskpassnum / count, "%")
