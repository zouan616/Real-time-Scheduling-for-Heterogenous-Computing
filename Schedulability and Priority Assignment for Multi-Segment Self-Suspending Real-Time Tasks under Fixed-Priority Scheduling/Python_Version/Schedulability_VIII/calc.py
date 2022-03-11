def get_Mi_Max(num):
    m_max = M[0]
    for i in range(0, num):
        m_max = M[i] if m_max < M[i] else m_max
    return m_max


def suspension_time_part(i, j):
    return task[i][2 * j + 1]


# Function: calculate the total suspension time of task i
def suspension_time(i):
    result = 0
    for j in range(0, M[i] - 1):
        result += suspension_time_part(i, j)
    return result


def execution_time_part(i, j):
    return task[i][2 * j]


# Function: calculate the total computation time of task i
def execution_time(i):
    result = 0
    for j in range(0, M[i]):
        result += execution_time_part(i, j)
    return result


# Function: calculate the minimum interval-arrival time
#           between the (j-1)th and the j-th released computation segments
def s(i, j):
    if j % M[i] != M[i] - 1:
        return suspension_time_part(i, j % M[i])
    elif j <= M[i]:
        return T[i] - D[i]
    else:
        return T[i] - execution_time(i) - suspension_time(i)


def find_l_max(i, h, t):
    result = 0
    j = h
    temp_ = execution_time_part(i, j % M[i]) + s(i, j)
    if temp_ > t:
        if h == 0:
            return 0
        else:
            return -1
    result += execution_time_part(i, j % M[i]) + s(i, j)
    while result <= t:
        j += 1
        result += execution_time_part(i, j % M[i]) + s(i, j)
    return j - 1


# Function: calculate the upper bound on the amount of execution
#           that the jobs of task i can perform in the time interval of t,
#           where h states the phasing.
def workload_function_part(i, h, t):
    result = 0
    l_max = find_l_max(i, h, t)
    for j in range(h, l_max + 1):
        result += execution_time_part(i, j % M[i])
    time_part_one = execution_time_part(i, (l_max + 1) % M[i])
    time_part_two = t
    for j in range(h, l_max + 1):
        temp_ = (execution_time_part(i, j % M[i]) + s(i, j))
        time_part_two -= temp_
    result += time_part_one if time_part_one < time_part_two else time_part_two
    return result


# Maximum workload function
def workload_function(i, t):
    result = 0
    for h in range(M[i]):
        if result < workload_function_part(i, h, t):
            result = workload_function_part(i, h, t)
    return result


def worst_case_response_time_part(k, j):
    result = execution_time_part(k, j)
    if k == 0:
        return result
    while 1:
        temp_ = 0
        for i in range(k):
            temp_ += workload_function(i, result)
        if result >= execution_time_part(k, j) + temp_:
            break
        else:
            result += 1
            if result >= 50000:
                break
    return result


# Function: calculate the worst-case response time of multi-segment SSS task k_
def worst_case_response_time(k_, t):
    # global count
    result = execution_time(k_) + suspension_time(k_)
    if k_ == 0:
        return result
    while 1:
        temp_ = 0
        for i in range(k_):
            temp_ += workload_function(i, result)
        # print("Sigma_W = ", temp_)
        if result >= execution_time(k_) + suspension_time(k_) + temp_:
            # result = range(result - 9, result + 1)

            for real_result in range(result, result + 1):
                if real_result >= execution_time(k_) + suspension_time(k_) + temp_:
                    result = real_result
                    break
            # return result
            if result <= t[k_]:
                result_toCmp = suspension_time(k_)
                for j in range(M[k_]):
                    result_toCmp += worst_case_response_time_part(k_, j)
                # print("=====================================")
                # print("Count =", count)
                # print("Deadline = ", t[k_])
                # print("Worst case response time = ", result)
                # print("Worst case response time toCmp = ", result_toCmp)
                if result > result_toCmp:
                    return result_toCmp
                else:
                    return result
            else:
                break
        else:
            if result >= 50000:
                break
            if result > t[k_]:
                return -1
            result += 1
    return result


def worst_case_response_time_2(k_, t, num_cpu):
    result = 0
    c_index = 0
    s_index = 0
    temp_counter = 0
    sigma_counter = 0
    while 1:
        result += 1
        # print("======================================================")
        # print("Time = ", result)
        temp_ = 0
        for i in range(k_):
            temp_ += workload_function(i, result)
        # print("Workload function net for task", k_, " = ", temp_)
        # print("M * t = ", num_cpu * result)
        # print("Sigma_Workload = ", temp_ + temp_counter)
        # print("-----------------------------------------------------------")
        if num_cpu * result > temp_ + sigma_counter:
            # print("function satisfied, currently result = ", result)
            temp_counter += 1
            sigma_counter += 1
            # print("currently temp_counter = ", temp_counter)
            # print("currently sigma_counter = ", sigma_counter)
            # print("--------------------------------------------------------")
            if temp_counter >= task[k_][2 * c_index]:
                # print("temp_counter is larger than computation segment")
                # print("temp_counter:", temp_counter)
                # print("computation segment:", task[k_][2 * c_index])
                temp_counter = 0
                # print("C_index = ", c_index)
                if c_index == M[k_] - 1:
                    if result > t[k_]:
                        return -1
                    else:
                        return result
                else:
                    result += task[k_][2 * s_index + 1]
                    sigma_counter += num_cpu * task[k_][2 * s_index + 1]
                    # print("jump the suspension segment, result = ", result)
                    # print("--------------------------------------------------------")
                    s_index += 1
                c_index += 1
            if result > t[k_]:
                return -1

        else:
            if result > t[k_]:
                return -1


# Function: initialize the tasks
# n for task number
# M[i] records number of computation segments of task[i]
# T[i] and D[i] records the period and relative deadline of task[i]
# each line of task[][] is characterized in: (C0, S0, C1, S1, ..., S(M-2), C(M-1))

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
ins.close()


# def test(n_):
#     print("=========================================================")
#     for i in range(n_):
#         temp = worst_case_response_time(i)
#         print("Worst case response time", i, " = ", temp)
#         if temp <= T[i]:
#             print("task[", i, "] passed")
#         else:
#             print("task[", i, "] failed")
#             break
#     print("=========================================================")

# print("=========================================================")
# for i in range(n):
#     temp = worst_case_response_time(i, T)
#     # if temp == -1:
#     #     print("task[", i, "] failed")
#     #     break
#     print("Worst case response time", i, " = ", temp)
#     if temp <= T[i]:
#         print("task[", i, "] passed")
#     else:
#         print("task[", i, "] failed")
#         break
# print("=========================================================")

# print("=========================================================")
# for i in range(n):
#     temp = worst_case_response_time(i, T)
#     if temp == -1:
#         print("task[", i, "] failed")
#         break
#     print("Worst case response time", i, " = ", temp)
#     print("Deadline", i, " = ", T[i])
#     if temp <= T[i]:
#         print("task[", i, "] passed")
#     else:
#         print("task[", i, "] failed")
#         break
# print("=========================================================")
def swap(task_, row1, row2, t, d):
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
    # print("After swapping:", task_)


def per(task_, i_, n_, t, d, num_cpu):
    global flag_all_passed
    global count
    # global T
    if i_ == n_ - 1:
        if flag_all_passed == 1:
            return
        count += 1
        flag = 1
        for i in range(n):
            temp = worst_case_response_time_2(i, T, num_cpu)
            if temp == -1:
                flag = 0
                break
            elif temp > T[i]:
                flag = 0
                break
        if flag == 1:
            flag_all_passed = 1
            print("Count = ", count)
        return
    for j in range(i_, n):
        swap(task_, i_, j, t, d)
        per(task_, i_ + 1, n_, t, d, num_cpu)
        swap(task_, i_, j, t, d)


# task_cpy = task
flag_all_passed = 0
count = 0
num_cpu = 2

# flag_ = 1
# for i in range(n):
#     temp = worst_case_response_time_2(i, T, num_cpu)
#     # print("Worst case response time for task", i, "= ", temp)
#     # print("==========================================================")
#     if temp == -1:
#         flag_ = 0
#         break
#     elif temp > T[i]:
#         flag_ = 0
#         break
# if flag_ == 1:
#     print("All tasks passed")
# elif flag_ == 0:
#     print("Failed")

per(task, 0, n, T, D, num_cpu)
print("=====================================================")
if flag_all_passed == 0:
    print("Failed")
elif flag_all_passed == 1:
    print("All tasks passed")
else:
    print("ERROR")
print("=====================================================")
# task = task_cpy
# for loop, 如果有一种case全pass就break
