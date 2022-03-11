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
    for j in range(0, m(task_, i) - 1):
        result += suspension_time_part(task_, i, j)
    return result


def execution_time_part(task_, i, j):
    return task_[i][2 * j]


# Function: calculate the total computation time of task i
def execution_time(task_, i):
    result = 0
    for j in range(0, m(task_, i)):
        result += execution_time_part(task_, i, j)
    return result


def m(task_, i):
    if task_ == task:
        return M[i]
    elif task_ == task_g1:
        return M_g1[i]
    elif task_ == task_g2:
        return M_g2[i]


def period(task_, i):
    if task_ == task:
        return T[i]
    elif task_ == task_g1:
        return T_g1[i]
    elif task_ == task_g2:
        return T_g2[i]


# Function: calculate the minimum interval-arrival time
#           between the (j-1)th and the j-th released computation segments
def s(task_, i, j):
    if j % m(task_, i) != m(task_, i) - 1:
        return suspension_time_part(task_, i, j % m(task_, i))
    elif j <= m(task_, i):
        return 0
    else:
        return period(task_, i) - execution_time(task_, i) - suspension_time(task_, i)


def find_l_max(task_, i, h, t):
    result = 0
    j = h
    temp_ = execution_time_part(task_, i, j % m(task_, i)) + s(task_, i, j)
    if temp_ > t:
        if h == 0:
            return 0
        else:
            return -1
    result += execution_time_part(task_, i, j % m(task_, i)) + s(task_, i, j)
    while result <= t:
        j += 1
        result += execution_time_part(task_, i, j % m(task_, i)) + s(task_, i, j)
    return j - 1


# Function: calculate the upper bound on the amount of execution
#           that the jobs of task i can perform in the time interval of t,
#           where h states the phasing.
def workload_function_part(task_, i, h, t):
    result = 0
    l_max = find_l_max(task_, i, h, t)
    for j in range(h, l_max + 1):
        result += execution_time_part(task_, i, j % m(task_, i))
    time_part_one = execution_time_part(task_, i, (l_max + 1) % m(task_, i))
    time_part_two = t
    for j in range(h, l_max + 1):
        temp_ = (execution_time_part(task_, i, j % m(task_, i)) + s(task_, i, j))
        time_part_two -= temp_
    result += time_part_one if time_part_one < time_part_two else time_part_two
    return result


# Maximum workload function
def workload_function(task_, i, t):
    result = 0
    for h in range(m(task_, i)):
        if result < workload_function_part(task_, i, h, t):
            result = workload_function_part(task_, i, h, t)
    return result


def worst_case_response_time_part(task_, k, j):
    result = execution_time_part(task_, k, j)
    if k == 0:
        return result
    while 1:
        temp_ = 0
        for i in range(k):
            temp_ += workload_function(task_, i, result)
        if result >= execution_time_part(task_, k, j) + temp_:
            break
        else:
            result += 1
            if result >= 50000:
                break
    return result


# Function: calculate the worst-case response time of multi-segment SSS task k_
def worst_case_response_time(task_, k_):
    # global count
    result = execution_time(task_, k_) + suspension_time(task_, k_)
    if k_ == 0:
        return result
    while 1:
        temp_ = 0
        for i in range(k_):
            temp_ += workload_function(task_, i, result)
        # print("Sigma_W = ", temp_)
        if result >= execution_time(task_, k_) + suspension_time(task_, k_) + temp_:
            for real_result in range(result, result):
                if real_result >= execution_time(task_, k_) + suspension_time(task_, k_) + temp_:
                    result = real_result
                    break
            # return result
            if result <= period(task_, k_):
                result_toCmp = suspension_time(task_, k_)
                for j in range(m(task_, k_)):
                    result_toCmp += worst_case_response_time_part(task_, k_, j)
                if result > result_toCmp:
                    return result_toCmp
                else:
                    return result
            else:
                break
        else:
            if result >= 50000:
                break
            if result > period(task_, k_):
                return -1
            result += 1
    return result


def split(task_, n_):
    U_sum = 0
    min_val = 2
    global pos_j
    global pos_i
    for ui in range(n_):
        U[ui] = execution_time(task_, ui) / T[ui]
        U_sum += U[ui]
    for i_ in range(n_):
        left_sum = U[i_]
        right_sum = U_sum - U[i_]
        diff = abs(left_sum - right_sum)
        if diff < min_val:
            min_val = diff
            pos_i = i_
            pos_j = -1
    for i_ in range(n_):
        for j_ in range(i_ + 1, n_):
            left_sum = U[i_] + U[j_]
            right_sum = U_sum - left_sum
            diff = abs(left_sum - right_sum)
            if diff < min_val:
                min_val = diff
                pos_i = i_
                pos_j = j_


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
            U = [0 for i in range(n)]
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


# task_cpy = task
count = 0
num_cpu = 2
pos_i = -2
pos_j = -2
split(task, n)
if pos_j == -1:
    task_g1 = task[0: 1]
    task_g1[0] = task[pos_i]
    task_g2 = task[0: 4]
    M_g1 = M[0: 1]
    M_g1[0] = M[pos_i]
    M_g2 = M[0: 4]
    T_g1 = T[0: 1]
    T_g1[0] = T[pos_i]
    T_g2 = T[0: 4]
    j = 0
    for i in range(n):
        if i != pos_i:
            task_g2[i - j] = task[i]
            M_g2[i - j] = M[i]
            T_g2[i - j] = T[i]
        else:
            j += 1
else:
    task_g1 = task[0: 2]
    task_g2 = task[0: 3]
    task_g1[0] = task[pos_i]
    task_g1[1] = task[pos_j]
    M_g1 = M[0: 2]
    M_g1[0] = M[pos_i]
    M_g1[1] = M[pos_j]
    M_g2 = M[0: 3]
    T_g1 = T[0: 2]
    T_g1[0] = T[pos_i]
    T_g1[1] = T[pos_j]
    T_g2 = T[0: 3]
    j = 0
    for i in range(n):
        if (i != pos_i) & (i != pos_j):
            task_g2[i - j] = task[i]
            M_g2[i - j] = M[i]
            T_g2[i - j] = T[i]
        else:
            j += 1
print(pos_i, pos_j)
print(task)
print(task_g1)
print(task_g2)
n_g1 = len(task_g1)
n_g2 = len(task_g2)
flag_g1 = 0
flag_g2 = 0
# print("=========================================================")
# flag_all_passed = 1
# for i in range(n_g1):
#     temp = worst_case_response_time(task_g1, i)
#     if temp == -1:
#         flag_all_passed = 0
#         break
#     elif temp > period(task_g1, i):
#         flag_all_passed = 0
#         break
# for i in range(n_g2):
#     temp = worst_case_response_time(task_g2, i)
#     if temp == -1:
#         flag_all_passed = 0
#         break
#     elif temp > period(task_g2, i):
#         flag_all_passed = 0
#         break
# if flag_all_passed == 1:
#     print("All tasks passed")
# else:
#     print("Failed")
# print("=========================================================")


def swap_g1(task_, row1, row2):
    temp = task_[row1]
    task_[row1] = task_[row2]
    task_[row2] = temp
    t_temp = T_g1[row1]
    T_g1[row1] = T_g1[row2]
    T_g1[row2] = t_temp


def swap_g2(task_, row1, row2):
    temp = task_[row1]
    task_[row1] = task_[row2]
    task_[row2] = temp
    t_temp = T_g2[row1]
    T_g2[row1] = T_g2[row2]
    T_g2[row2] = t_temp


def per_g1(task_, i_, n_):
    global flag_g1
    # global T
    if i_ == n_ - 1:
        if flag_g1 == 1:
            return
        flag = 1
        for i in range(n_):
            temp = worst_case_response_time(task_, i)
            if temp == -1:
                flag = 0
                break
            elif temp > period(task_, i):
                flag = 0
                break
        if flag == 1:
            flag_g1 = 1
        return
    for j in range(i_, n_):
        swap_g1(task_, i_, j)
        per_g1(task_, i_ + 1, n_)
        swap_g1(task_, i_, j)


def per_g2(task_, i_, n_):
    global flag_g2
    # global T
    if i_ == n_ - 1:
        if flag_g2 == 1:
            return
        flag = 1
        for i in range(n_):
            temp = worst_case_response_time(task_, i)
            if temp == -1:
                flag = 0
                break
            elif temp > period(task_, i):
                flag = 0
                break
        if flag == 1:
            flag_g2 = 1
        return
    for j in range(i_, n_):
        swap_g2(task_, i_, j)
        per_g2(task_, i_ + 1, n_)
        swap_g2(task_, i_, j)


per_g1(task_g1, 0, n_g1)
per_g2(task_g2, 0, n_g2)
if flag_g1 == 1 & flag_g2 == 1:
    print("All tasks passed")
else:
    print("Failed")
# per(task, 0, n, T, D, num_cpu)
# print("=====================================================")
# if flag_all_passed == 0:
#     print("Failed")
# elif flag_all_passed == 1:
#     print("All tasks passed")
# else:
#     print("ERROR")
# print("=====================================================")

# task = task_cpy
# for loop, 如果有一种case全pass就break
