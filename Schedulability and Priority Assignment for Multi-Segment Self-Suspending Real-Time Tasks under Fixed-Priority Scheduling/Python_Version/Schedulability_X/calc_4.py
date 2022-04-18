# STGM:
import math


def get_Mi_Max(num):
    m_max = M[0]
    for i in range(0, num):
        m_max = M[i] if m_max < M[i] else m_max
    return m_max


def suspension_time_part(task_, i, j):
    return task_[i][2 * j + 1]


# Function: calculate the total suspension time of task i
def suspension_time(task_, i):
    result = 0
    for j in range(0, M[i] - 1):
        result += suspension_time_part(task_, i, j)
    return result


def execution_time_part(task_, i, j):
    return task_[i][2 * j]


# Function: calculate the total computation time of task i
def execution_time(task_, i):
    result = 0
    for j in range(0, M[i]):
        result += execution_time_part(task_, i, j)
    return result


def blocking_time_m(task_, i):
    return 0


def max_G(task_, i, M):
    suspension_segment = M[i] - 1
    max_res = -1
    for omega in range(suspension_segment):
        if suspension_time_part(task_, i, omega) > max_res:
            max_res = suspension_time_part(task_, i, omega)
    return max_res


def blocking_time_eij(task_, i, M):
    res = 0
    for u in range(n):
        if u != i:
            res += max_G(task_, u, M)
    return res


# 返回0，因为gpu核不共享
def blocking_time_e(task_, i, M):
    # res = 0
    # for j in range(M[i] - 1):
    #     res += blocking_time_eij(task_, i, M)
    # return res
    return 0


def blocking_time_l(task_, i, M):
    return 0


def blocking_time(task_, i, M):
    result = blocking_time_l(task_, i, M) + blocking_time_e(task_, i, M) + blocking_time_m(task_, i)
    return result


def workload_function(task_, i, M, T, workload):
    W_k = execution_time(task_, i) + suspension_time(task_, i) + blocking_time(task_, i, M)
    fp = execution_time(task_, i) + suspension_time(task_, i) + blocking_time(task_, i, M)
    # print("First part = ", fp)
    workload[i] = W_k
    count = 0
    while 1:
        # print("-------------------------")
        # print("new iteration")
        # print("k =", count)
        sum = 0
        for h in range(i):
            temp = W_k + workload[h] - execution_time(task_, h)
            sum += execution_time(task_, h) * math.ceil(temp / T[h])
        W_k = fp + sum
        # print("Last Workload =", Workload[i])
        # print("New Workload =", W_k)
        if W_k == workload[i]:
            return W_k
        if count >= 100:
            return -1
        workload[i] = W_k
        count += 1


line_count = 0
n = 0
with open("input.txt", "r") as ins:
    for lines in ins:
        line_count += 1
        # lines = lines.replace(" ", "")
        lines = lines.replace("\n", "")
        lines = lines.replace(",", " ")
        numbers_str = lines.split()
        numbers_int = [int(x) for x in numbers_str]
        if line_count == 1:
            n = numbers_int[0]
            M = [0 for i in range(n)]
            T = [0 for i in range(n)]
            D = [0 for i in range(n)]
            U = [0 for i in range(n)]
            Workload = [0 for i in range(n)]
        elif line_count == 2:
            for i in range(n):
                M[i] = numbers_int[i]
            max_Mi = get_Mi_Max(n)
            task = [[0 for i in range(2 * max_Mi - 1)] for j in range(n)]
        elif line_count == 3:
            for i in range(n):
                T[i] = numbers_int[i]
        elif line_count == 4:
            for i in range(n):
                D[i] = numbers_int[i]
        else:
            # T[line_count - 3] = numbers_int[0]
            # D[line_count - 3] = numbers_int[1]
            for i in range(2 * M[line_count - 5] - 1):
                task[line_count - 5][i] = numbers_int[i]
for ui in range(n):
    U[ui] = execution_time(task, ui) / T[ui]
ins.close()


def swap(task_, row1, row2, t, d, m, u):
    # print("Original task:", task_)
    # global T
    temp = task_[row1]
    task_[row1] = task_[row2]
    task_[row2] = temp
    t_temp = t[row1]
    t[row1] = t[row2]
    t[row2] = t_temp
    d_temp = d[row1]
    d[row1] = d[row2]
    d[row2] = d_temp
    m_temp = m[row1]
    m[row1] = m[row2]
    m[row2] = m_temp
    u_temp = u[row1]
    u[row1] = u[row2]
    u[row2] = u_temp
    # print("After swapping:", task_)


def rearrange(task_, n_):
    # 将所有task按照utilization从高到低排列
    for i in range(n_):
        for j in range(n_ - i - 1):
            if U[j] <= U[j + 1]:
                swap(task_, j, j + 1, T, D, M, U)


print(task)
rearrange(task, n)
print("After rearrangement", task)
sche = True
U_g1 = 0
U_g2 = 0
task_g1 = [[0 for i in range(2 * max_Mi - 1)] for j in range(n)]
task_g2 = [[0 for i in range(2 * max_Mi - 1)] for k in range(n)]
zero_task = [0 for z in range(2 * max_Mi - 1)]
n_g1 = 0
n_g2 = 0
workload_g1 = [0 for i in range(n)]
workload_g2 = [0 for i in range(n)]
for i in range(n):
    # print("New task======================================")
    # print("Consider task", i, "=====================================")
    # print("Task_g1 =", task_g1)
    # print("Task_g2 =", task_g2)
    # print("Workload_g1 =", workload_g1)
    # print("Workload_g2 =", workload_g2)
    # print("N_g1 = ", n_g1)
    # print("N_g2 = ", n_g2)
    # print("U_g1 = ", U_g1)
    # print("U_g2 = ", U_g2)
    if U_g1 <= U_g2:
        # print("Case 1: U_g1 <= U_g2 ---------------------------------")
        task_g1[n_g1] = task[i]
        res = workload_function(task_g1, n_g1, M, T, workload_g1)
        if (1 - U_g1 >= U[i]) & (res != -1) & (res <= T[i]):
            # print("Satisfied, add task", i, "to task_g1")
            U_g1 += U[i]
            n_g1 += 1
        else:
            task_g2[n_g2] = task_g1[n_g1]
            task_g1[n_g1] = zero_task
            workload_g1[n_g1] = 0
            res = workload_function(task_g2, n_g2, M, T, workload_g2)
            if (1 - U_g2 >= U[i]) & (res != -1) & (res <= T[i]):
                # print("Satisfied, add task", i, "to task_g2")
                U_g2 += U[i]
                n_g2 += 1
            else:
                # print("not schedulable")
                workload_g2[n_g2] = 0
                task_g2[n_g2] = zero_task
                sche = False
                break
    elif U_g1 > U_g2:
        # print("Case 2: U_g1 > U_g2 ---------------------------------")
        # print("Task_g2 = ", task_g2)
        # print("Try add task", i, "into task_g2")
        task_g2[n_g2] = task[i]
        # print("Task_g2 = ", task_g2)
        res = workload_function(task_g2, n_g2, M, T, workload_g2)
        if (1 - U_g2 >= U[i]) & (res != -1) & (res <= T[i]):
            # print("Satisfied, add task", i, "to task_g2")
            U_g2 += U[i]
            n_g2 += 1
        else:
            # print("Failed. Task_g1 =", task_g1)
            # print("Try add task", i, "into task_g1")
            task_g1[n_g1] = task_g2[n_g2]
            # print("Task_g1 = ", task_g1)
            workload_g2[n_g2] = 0
            # print("After that, task_g1 =", task_g1)
            task_g2[n_g2] = zero_task
            # print("After delete task_g2, task_g1 =", task_g1)
            res = workload_function(task_g1, n_g1, M, T, workload_g1)
            if (1 - U_g1 >= U[i]) & (res != -1) & (res <= T[i]):
                # print("Satisfied, add task", i, "to task_g1")
                U_g1 += U[i]
                n_g1 += 1
            else:
                # print("not schedulable")
                workload_g1[n_g1] = 0
                task_g1[n_g1] = zero_task
                sche = False
                break
print("===================================================")
if sche:
    print("All tasks passed")
else:
    print("Failed")
print("In the end:")
print("Task_g1 = ", task_g1)
print("Task_g2 = ", task_g2)
