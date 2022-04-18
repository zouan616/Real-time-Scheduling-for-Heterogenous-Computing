from random import randint


def suspension_time_part(task, i, j):
    return task[i][2 * j + 1]


# Function: calculate the total suspension time of task i
def suspension_time(task, i, M):
    result = 0
    for j in range(0, M[i] - 1):
        result += suspension_time_part(task, i, j)
    return result


def execution_time_part(task, i, j):
    return task[i][2 * j]


# Function: calculate the total computation time of task i
def execution_time(task, i, M):
    result = 0
    for j in range(0, M[i]):
        result += execution_time_part(task, i, j)
    return result


def gen(n, m, c_max, utilization_net, s1, s2):
    task = [[0 for _ in range(2 * m - 1)] for _ in range(n)]
    M = [m for _ in range(n)]
    T = [0 for _ in range(n)]
    D = [0 for _ in range(n)]
    U = [0 for _ in range(n)]
    # set utilization
    while 1:
        U_sum = 0
        flag_valid = 1
        for ui in range(n):
            U[ui] = randint(1, 10)
            U_sum += U[ui]
        resolution = U_sum / utilization_net
        for ui in range(n):
            U[ui] = U[ui] / resolution
            if U[ui] > 1:
                flag_valid = 0
        if flag_valid == 1:
            break
    for ui in range(n):
        print("Utilization for task[", ui, "] is", U[ui])
    # set computation segments & suspension segments
    for i in range(n):
        for j in range(m):
            task[i][2 * j] = randint(1, c_max)
        # for j in range(m - 1):
        #     task[i][2 * j + 1] = randint(1, s_max)
    # set Period & Deadline
    for i_ in range(n):
        temp = execution_time(task, i_, M)
        T[i_] = int(temp / U[i_])
        D[i_] = T[i_]
    # set suspension segments
    for i in range(n):
        # for j in range(m):
        #     task[i][2 * j] = randint(1, c_max)
        for j in range(m - 1):
            s_max_temp = T[i] - execution_time(task, i, M)
            lower_bound = max(int(s1 * s_max_temp / (m - 1)), 1)
            upper_bound = max(int(s2 * s_max_temp / (m - 1)), 2)
            while lower_bound >= upper_bound:
                lower_bound = max(int(s1 * s_max_temp / (m - 1)), 1)
                upper_bound = max(int(s2 * s_max_temp / (m - 1)), 2)
            task[i][2 * j + 1] = randint(lower_bound, upper_bound)
    print(task)
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
            for j in range(2 * m - 2):
                fp.write(str(task[i][j]) + " ")
            fp.write(str(task[i][2 * m - 2]) + "\n")
    fp.close()
# print("input integer N = ")
# N = int(input())
# print("input integer M = ")
# m = int(input())
# M = [m for _ in range(N)]
# print("input integer C_max = ")
# c_max = int(input())
# print("input integer S_max = ")
# s_max = int(input())
# print("input utilization rate = ")
# utilization = float(input())
