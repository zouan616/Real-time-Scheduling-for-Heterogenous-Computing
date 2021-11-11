# Set a task
from random import randint


def suspension_time_part(i, j):
    return task[i][2 * j + 1]


def suspension_time(i):
    result = 0
    for j in range(0, M_i - 1):
        result += suspension_time_part(i, j)
    return result


def execution_time_part(i, j):
    return task[i][2 * j]


def execution_time(i):
    result = 0
    for j in range(0, M_i):
        result += execution_time_part(i, j)
    return result


def D(i):
    if i == 0:
        return 20
    else:
        return 30
    # return execution_time(i) / utilization


def T(i):
    return D(i)


def s(i, j):
    if j % M_i != M_i - 1:
        return suspension_time_part(i, j % M_i)
    elif j <= M_i:
        return T(i) - D(i)

    else:
        return T(i) - execution_time(i) - suspension_time(i)


def find_l_max(i, h, t):
    result = 0
    j = h
    temp = execution_time_part(i, j % M_i) + s(i, j)
    if temp > t:
        if h == 0:
            return 0
        else:
            return -1
    result += execution_time_part(i, j % M_i) + s(i, j)
    while result <= t:
        j += 1
        result += execution_time_part(i, j % M_i) + s(i, j)
    return j - 1


def workload_function_part(i, h, t):
    result = 0
    l_max = find_l_max(i, h, t)
    for j in range(h, l_max + 1):
        # print("value for j = ", j)
        result += execution_time_part(i, j % M_i)
    time_part_one = execution_time_part(i, (l_max + 1) % M_i)
    time_part_two = t
    for j in range(h, l_max + 1):
        temp = (execution_time_part(i, j % M_i) + s(i, j))
        # print("temp: ", temp)
        time_part_two -= temp
    # print("l_max for", i, " = ", l_max)
    # print("time_part_one for", i, " = ", time_part_one)
    # print("time_part_two for", i, " = ", time_part_two)
    # print("result = ", result)
    result += time_part_one if time_part_one < time_part_two else time_part_two
    return result


def workload_function(i, t):
    result = 0
    for h in range(M_i):
        if result < workload_function_part(i, h, t):
            result = workload_function_part(i, h, t)
    return result


def worst_case_response_time(k_):
    result = execution_time(k_) + suspension_time(k_)
    if k_ == 0:
        return result
    while 1:
        temp = 0
        for i in range(k_):
            temp += workload_function(i, result)
        if result >= execution_time(k_) + suspension_time(k_) + temp:
            break
        else:
            result += 1
            if result >= 10000:
                break
    return result


#  -----------------------------------------------------------------------------------------------------
M_i = 2  # size N for each task
C_max = 10
S_max = 5
utilization = 0.1
n = 3  # totally n tasks


task = [[0 for i in range(2 * M_i - 1)] for j in range(n)]
task[0][0] = 2
task[0][1] = 4
task[0][2] = 1

task[1][0] = 3
task[1][1] = 5
task[1][2] = 7


print(task)
i = 0
h = 0
t = 20

# print("suspension time = ", suspension_time(0))
# print("execution time = ", execution_time(0))
# print("s(i,j) = ", s(0, 0))
# print("s(i,j) = ", s(0, 1))
# print("s(i,j) = ", s(0, 3))
print("l_max = ", find_l_max(0, 0, t))
print("workload function part =", workload_function_part(0, 0, t))
print("workload function = ", workload_function(0, t))
print("Rk for i = 0: ", worst_case_response_time(0))
print("Rk for i = 1: ", worst_case_response_time(1))

r = 21
print("Workload function for r = ", r, ": ", workload_function(0, r))

# passed_total = 0
# num_tasks = 100
# for q in range(num_tasks):
#     task = [[0 for i in range(2 * M_i - 1)] for j in range(n)]
#     for i in range(n):
#         for j in range(M_i):
#             task[i][2 * j] = randint(0, C_max)
#         for j in range(M_i - 1):
#             task[i][2 * j + 1] = randint(0, S_max)
#     print(task)
#     for k in range(M_i):
#         if worst_case_response_time(k) < D(k):
#             print("Task", k, "passed")
#         else:
#             print("Task", k, "failed")
#             break
#     passed_total += 1
# print("Passed rate = ", passed_total / num_tasks)
