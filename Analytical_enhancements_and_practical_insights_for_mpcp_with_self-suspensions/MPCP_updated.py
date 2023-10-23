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


def WCRT(
        k: int,
        task_: List[List[int]],
        T_: List[int],
        D_: List[int],
        R_: List[int],
        zeta_: int,
        M_: List[int],
        max_critSection_: List[int]):
    """
    Worst Case Response Time of task \tau_k

    :param k: task index
    :param task_: taskset
    :param T_: period
    :param D_: deadline
    :param R_: lock
    :param zeta_: number of suspension in critical section, default=0
    :param M_: CPU segment number
    :param max_critSection_: max critical section length
    :return: WCRT of task k
    """

    if wcrt[k] != 0:
        return wcrt[k]
    blockTime = blocking_time(k, task_, T_, D_, R_, zeta_, M_, max_critSection_)
    if blockTime == -1:
        return -1
    result = execution_time(task_, k, M) + suspension_time(task_, k, M) + blockTime
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
                worstCaseResponseTime = WCRT(h, task_, T_, D_, R_, zeta_, M_, max_critSection_)
                temp += math.ceil((result + worstCaseResponseTime - execution_time(task_, h, M_))
                                  / (T[h])) * execution_time(task_, h, M_)
            if result >= temp:
                break
            result += 1
            iteration_count += 1
        wcrt[k] = result
        return result


def H_(
        _l: int,
        k: int,
        R_: List[int],
        task_: List[List[int]],
        zeta_: int,
        max_critSection_: List[int]):
    """
    WCRT of k-th critical section of task \tau_l. Equation (14)
    Assumption: 10% of CPU execution is critical

    :param _l: task index
    :param k: critical section index
    :param R_: lock
    :param task_: taskset
    :param zeta_: number of suspension in a critical section, default = 0
    :param max_critSection_: max critical section length
    :return: WCRT of k-th critical section of task \tau_l
    """

    temp = int(critical_proportion * execution_time_part(task_, _l, k)) + suspension_time_part(task_, _l, k)
    temp += indirect_blocking_time(_l, R_, zeta_, max_critSection_)
    return temp


def direct_blocking_time_req_helper(
        i: int,
        R_: List[int],
        task_: List[List[int]],
        zeta_: int,
        T_: List[int],
        D_: List[int],
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
    :param D_: deadline
    :param M_: CPU segment number
    :param max_critSection_: max critical section length
    :return: segment-level worst case blocking time
    """
    leftSum = 0
    max_H_lp = 0
    max_H_hp = 0

    # WCRT of critical section of all lp tasks
    for _l in range(i + 1, n):
        if R[_l] == R[i]:
            for k in range(M_[_l] - 1):
                temp = H_(_l, k, R_, task_, zeta_, max_critSection_)
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
                wcrt_h = WCRT(h, task_, T_, D_, R_, zeta_, M_, max_critSection_)
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
        D_: List[int],
        R_: List[int],
        zeta_: int,
        M_: List[int],
        max_critSection_: List[int]):
    """
    Request-Driven Approach

    :param i: task index
    :param task_: taskset
    :param T_: period
    :param D_: deadline
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
    seg_blockingTime = direct_blocking_time_req_helper(i, R_, task_, zeta_, T_, D_, M_, max_critSection_)
    if seg_blockingTime == -1:
        return -1
    result = M_[i] * seg_blockingTime
    return result


def direct_blocking_time_job(
        i: int,
        task_: List[List[int]],
        T_: List[int],
        D_: List[int],
        R_: List[int],
        zeta_: int,
        max_critSection_: List[int],
        M_: List[int]):
    """
    Job-Driven Approach

    :param i: task index
    :param task_: taskset
    :param T_: period
    :param D_: deadline
    :param R_: lock
    :param zeta_: number of suspension in a critical section, default=0
    :param max_critSection_: max critical section
    :param M_: CPU segment number
    :return: direct block time
    """

    max_H_lp = 0
    eta_i = max(int(critical_proportion * (M[i] - 1)), 1)
    for _l in range(i + 1, n):
        if R[_l] == R[i]:
            for k in range(M_[_l] - 1):
                temp = H_(_l, k, R_, task_, zeta_, max_critSection_)
                if temp > max_H_lp:
                    max_H_lp = temp
    b_dj = max_H_lp * eta_i
    for h in range(i):
        max_H_hp = 0
        worstCastResponseTime_h = WCRT(h, task_, T_, D_, R_, zeta_, M_, max_critSection_)
        alpha = math.ceil((T[i] - worstCastResponseTime_h - execution_time(task_, h, M_)) / T[h])
        for k in range(M_[h] - 1):
            temp = H_(h, k, R_, task_, zeta_, max_critSection_)
            if temp > max_H_hp:
                max_H_hp = temp
        b_dj += alpha * max_H_hp
    # iteration_count = 0
    # while 1:
    #     if iteration_count > 100000:
    #         print("direct blocking job-approach timeout, iteration count > 100000")
    #         return -1
    #     for h in range(i):
    #         worstCastResponseTime_h = WCRT(h, task_, T_, D_, R_, zeta_, M_, max_critSection_)
    #         alpha = math.ceil((leftSum + worstCastResponseTime_h - execution_time(task_, h, M_)) / T[h])
    #         for k in range(M_[h] - 1):
    #             temp = H_(h, k, R_, task_, zeta_, max_critSection_)
    #             if temp > max_H_hp:
    #                 max_H_hp = temp
    #         rightSum += alpha * temp
    #     if leftSum >= rightSum:
    #         break
    #     else:
    #         leftSum += 10
    #     iteration_count += 1
    return b_dj


def direct_blocking_time_hybrid_hp(
        i: int,
        task_: List[List[int]],
        T_: List[int],
        D_: List[int],
        R_: List[int],
        zeta_: int,
        max_critSection_: List[int],
        M_: List[int]):
    """
    Hybrid Approach -- hp

    :param i: task index
    :param task_: taskset
    :param T_: period
    :param D_: deadline
    :param R_: lock
    :param zeta_: number of suspension in a critical section, default=0
    :param max_critSection_: max critical section length
    :param M_: CPU segment number
    :return: direct block time of higher priority tasks on same CPU, under hybrid approach
    """
    result = int((1 - critical_proportion) * execution_time(task_, i, M)) + suspension_time(task_, i, M)
    max_H_hp = 0
    b_dmh = 0
    for h in range(i):
        worstCaseResponseTime_h = WCRT(h, task_, T_, D_, R_, zeta_, M_, max_critSection_)
        alpha = math.ceil((result + worstCaseResponseTime_h - max_critSection_[h]) / T[h])
        beta = math.ceil((result + worstCaseResponseTime_h - max_critSection_[h]) / (T_[h]))
        delta = min(alpha, int(critical_proportion * M_[i]) * beta)
        for k in range(M_[h] - 1):
            temp = H_(h, k, R_, task_, zeta_, max_critSection_)
            if temp > max_H_hp:
                max_H_hp = temp
        b_dmh += delta * max_H_hp
    return b_dmh


def direct_blocking_time_hybrid_lp(
        i: int,
        task_: List[List[int]],
        R_: List[int],
        zeta_: int,
        max_critSection_: List[int],
        M_: List[int]):
    """
    Hybrid Approach -- lp

    :param i: task index
    :param task_: taskset
    :param R_: lock
    :param zeta_: number of suspension in a critical section, default=0
    :param max_critSection_: max critical section length
    :param M_: CPU segment number
    :return: direct block time of lower priority tasks
    """
    b_dml = 0
    L_ijk = 0

    for j in range(n):
        if R[j] == R[i]:
            Q_ij = 0
            for _l in range(i + 1, n):
                if R[_l] == R[j]:
                    Q_ij += 1
                    for k in range(M_[_l] - 1):
                        temp = H_(_l, k, R_, task_, zeta_, max_critSection_)
                        if temp > L_ijk:
                            L_ijk = temp
                    b_dml += L_ijk
    return b_dml


def direct_blocking_time_hybrid(
        i: int,
        task_: List[List[int]],
        T_: List[int],
        D_: List[int],
        R_: List[int],
        zeta_: int,
        max_critSection_: List[int],
        M_: List[int]):
    """
    Hybrid Approach

    :param i: task index
    :param task_: taskset
    :param T_: period
    :param D_: deadline
    :param R_: lock
    :param zeta_: number of suspension in a critical section, default=0
    :param max_critSection_: max critical section length
    :param M_: CPU segment number
    :return: direct block time
    """
    result = direct_blocking_time_hybrid_lp(i, task_, R_, zeta_, max_critSection_, M_) + \
        direct_blocking_time_hybrid_hp(i, task_, T_, D_, R_, zeta_, max_critSection_, M_)
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


def prioritized_blocking_time_req(
        i: int,
        R_: List[int],
        max_critSection_: List[int]):
    """
    Request-Driven Approach

    :param i: task index
    :param R_: lock
    :param max_critSection_: max critical section length
    :return: request-driven prioritized blocking time
    """

    sum_lpp_blocking = 0
    for _l in range(i + 1, n):
        if R_[i] == R_[_l]:
            sum_lpp_blocking += max_critSection_[_l]
    eta = max(int(critical_proportion * (M[i] - 1)), 1)
    b_pr = sum_lpp_blocking * (eta + 1)
    return b_pr


def prioritized_blocking_time_job(
        i: int,
        R_: List[int],
        zeta_: int,
        task_: List[List[int]],
        T_: List[int],
        D_: List[int],
        M_: List[int],
        max_critSection_: List[int]):
    """
    prioritized blocking time -- job-driven approach

    :param i: task index
    :param R_: lock
    :param zeta_: number of suspension in a critical section, default=0
    :param task_: taskset
    :param T_: period
    :param D_: deadline
    :param M_: CPU segment number
    :param max_critSection_: max critical section length
    :return: prioritized blocking time
    """
    # iteration_count = 0
    # leftSum = execution_time(task_, i, M_) + suspension_time(task_, i, M_)
    B_pj = 0
    for _l in range(i + 1, n):
        theta = math.ceil((T_[_l] + D_[_l] - execution_time(task_, _l, M_)) / T_[_l])
        B_pj += theta * int(critical_proportion * execution_time(task_, _l, M_))
    # while 1:
    #     if iteration_count > 100000:
    #         print("prioritized blocking job-approach timeout, iteration count > 100000")
    #     B_pj = 0
    #     for _l in range(i, n):
    #         if R_[_l] == R_[i]:
    #             theta = math.ceil((T_[_l] + D_[_l] - execution_time(task_, _l, M_)) / T_[_l])
    #             B_pj += theta * int(0.1 * execution_time(task_, _l, M_))
    #     rightSum = leftSum + B_pj
    #     instance = 0
    #     for h in range(i):
    #         worstCastResponseTime_h = WCRT(h, task_, T_, D_, R_, zeta_, M_, max_critSection_)
    #         alpha = math.ceil((leftSum + worstCastResponseTime_h - execution_time(task_, h, M_))/T_[h])
    #         instance += alpha * execution_time(task, h, M_)
    #     rightSum += instance
    #     if leftSum >= rightSum:
    #         break
    #     else:
    #         leftSum += 10
    #     iteration_count += 1
    return B_pj


def prioritized_blocking_time_hybrid(
        i: int,
        R_: List[int],
        task_: List[List[int]],
        T_: List[int],
        D_: List[int],
        M_: List[int],
        max_critSection_: List[int]):
    """
    Prioritized blocking time -- hybrid approach

    :param i: task index
    :param R_: lock
    :param task_: taskset
    :param T_: period
    :param D_: deadline
    :param M_: CPU segment number
    :param max_critSection_: max critical section
    :return: prioritized blocking time under hybrid approach
    """
    eta_i = max(int(critical_proportion * (M[i] - 1)), 1)
    b_pm = 0
    for _l in range(i + 1, n):
        if R_[_l] == R_[i]:
            theta = math.ceil((T_[_l] + D_[_l] - execution_time(task_, _l, M_)) / T_[_l])
            eta_l = max(int(critical_proportion * (M[_l] - 1)), 1)
            psi = 0
            for k in range(1, eta_l + 1):
                iteration_count = 0
                while 1:
                    if iteration_count > 100000:
                        print("prioritized hybrid-approach blocking timeout, iteration count > 100000")
                        return -1
                    leftSum = eta_i + 1
                    for t in range(1, k):
                        leftSum -= psi
                    if psi >= min(leftSum, theta):
                        break
                    else:
                        psi += 1
                    iteration_count += 1
                b_pm += psi * max_critSection_[_l]
    return b_pm


def blocking_time(
        i: int,
        task_: List[List[int]],
        T_: List[int],
        D_: List[int],
        R_: List[int],
        zeta_: int,
        M_: List[int],
        max_critSection_: List[int]):
    """
    Total blocking time

    :param i: task index
    :param task_: taskset
    :param D_: deadline
    :param T_: period
    :param R_: lock
    :param zeta_: number of suspension in a critical section, default=0
    :param M_: CPU segment number
    :param max_critSection_: max critical section length
    :return: minimum total blocking time
    """

    direct_B_req = direct_blocking_time_req(i, task_, T_, D_, R_, zeta_, M_, max_critSection_)
    prioritized_B_req = prioritized_blocking_time_req(i, R_, max_critSection_)
    if direct_B_req == -1 or prioritized_B_req == -1:
        blocking_reqDriven = -1
    else:
        blocking_reqDriven = direct_B_req + prioritized_B_req

    direct_B_job = direct_blocking_time_job(i, task_, T_, D_, R_, zeta_, max_critSection_, M_)
    prioritized_B_job = prioritized_blocking_time_job(i, R_, zeta_, task_, T_, D_, M_, max_critSection_)
    if direct_B_job == -1 or prioritized_B_job == -1:
        blocking_jobDriven = -1
    else:
        blocking_jobDriven = direct_B_job + prioritized_B_job

    direct_B_hybrid = direct_blocking_time_hybrid(i, task_, T_, D_, R_, zeta_, max_critSection_, M_)
    prioritized_B_hybrid = prioritized_blocking_time_hybrid(i, R_, task_, T_, D_, M_, max_critSection_)
    if direct_B_hybrid == -1 or prioritized_B_hybrid == -1:
        blocking_hybrid = -1
    else:
        blocking_hybrid = direct_B_hybrid + prioritized_B_hybrid

    if blocking_reqDriven == -1 and blocking_jobDriven == -1 and blocking_hybrid == -1:
        return -1

    try:
        result = min(x for x in (blocking_reqDriven, blocking_jobDriven, blocking_hybrid) if x > 0)
    except ValueError:
        return -1
    # print("Request-driven: blocking = ", blocking_reqDriven)
    # print("Job-driven: blocking = ", blocking_jobDriven)
    # print("Hybrid: blocking = ", blocking_hybrid)
    return result


def generator(_utilization, _n):
    """
    Task generation

    :param _utilization: (double) net utilization
    :param _n: number of tasks in taskset
    :return: generated taskset & accompanying info
    """
    _task = [[0 for _ in range(203)] for _ in range(_n)]
    _T = [0 for _ in range(_n)]
    _D = [0 for _ in range(_n)]
    U = [0 for _ in range(_n)]
    _R = [0 for _ in range(_n)]

    # total CPU execution
    C_max_ = [0 for _ in range(_n)]
    C_max_[0] = 4
    C_max_[1] = 15
    C_max_[2] = 238
    C_max_[3] = 350
    C_max_[4] = 25
    C_min = [0 for _ in range(_n)]
    C_min[0] = 2
    C_min[1] = 13
    C_min[2] = 130
    C_min[3] = 150
    C_min[4] = 15
    S_max = [0 for _ in range(_n)]
    S_max[0] = 25
    S_max[1] = 8
    S_max[2] = 1
    S_max[3] = 483
    S_max[4] = 10
    S_min = [0 for _ in range(_n)]
    S_min[0] = 20
    S_min[1] = 5
    S_min[2] = 1
    S_min[3] = 482
    S_min[4] = 8
    max_critSection_value = [0 for _ in range(_n)]

    # set utilization rate
    for Ri in range(_n):
        _R[Ri] = Ri % 2
    while 1:
        U_sum = 0
        flag_valid = 1
        for ui in range(_n):
            U[ui] = randint(1, 10)
            U_sum += U[ui]
        resolution = U_sum / _utilization
        for ui in range(_n):
            U[ui] = U[ui] / resolution
            if U[ui] > 1:
                flag_valid = 0
        if flag_valid == 1:
            break
    for ui in range(_n):
        print("Utilization for task[", ui, "] is", U[ui])
    # set computation segments
    for i in range(_n):
        for j in range(M[i]):
            _task[i][2 * j] = randint(C_min[i], C_max_[i])
            if int(critical_proportion * _task[i][2 * j]) > max_critSection_value[i]:
                max_critSection_value[i] = int(critical_proportion * _task[i][2 * j])
        for j in range(M[i] - 1):
            _task[i][2 * j + 1] = randint(S_min[i], S_max[i])
    # set Period & Deadline
    for i in range(_n):
        temp = execution_time(_task, i, M) + suspension_time(_task, i, M)
        _T[i] = int(temp / U[i])
        _D[i] = _T[i]
    return _task, _D, _R, max_critSection_value


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
        # print taskset under new priority
        print("=================================================")
        print(f"Permutation {idx + 1}:")
        for row in NEW_task:
            print(row)
        print("Deadline Permutation:", deadline_permutation)
        print("Period Permutation:", period_permutation)
        print("Lock Permutation:", R_permutation)
        print("Segment number Permutation:", M_permutation)

        task_pass = 0
        for i in range(n_):
            wcrt[i] = 0
            directblocktime[i] = 0
        for i in range(n_):
            WCRT_res = WCRT(i, NEW_task, period_permutation, deadline_permutation, R_permutation, zeta, M_permutation,
                            maxCritSection_permutation)
            if WCRT_res == -1:
                print("task", i, "timeout")
                task_pass = 0
                break
            elif WCRT_res > deadline_permutation[i]:
                print("WCRT for task ", i, "= ", WCRT_res)
                print("Deadline = ", deadline_permutation[i])
                task_pass = 0
                break
            else:
                print("task", i, "passed")
                task_pass = 1
        if task_pass:
            print("Final result:pass")
            break

    if task_pass == 0:
        print("Final result:Not pass")
    return task_pass


count = 100
max_critical_section_num = 5  # the number of critical section
n = 5  # the number of task
zeta = 0  # Number of suspensions in a critical section
utilization = 1
taskpass = 1
taskpassnum = 0
batch = 0
critical_proportion = 0.1
M = [102 for _ in range(n)]
M[0] = 102
M[1] = 21
M[2] = 4
M[3] = 2
M[4] = 10
max_critSection = [0 for _ in range(n)]  # max WCET of critical section of task i

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

with open("log.txt", "a") as file:
    print("tasknum =", count, "\npasstasknum =", taskpassnum, file=file)
    print("Passed rate = ", 100 * taskpassnum / count, "%", file=file)

#
# print("tasknum =", count, "\npasstasknum =", taskpassnum)
# print("Passed rate = ", 100 * taskpassnum / count, "%")
