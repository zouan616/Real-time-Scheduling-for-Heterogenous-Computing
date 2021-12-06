from random import randint


def execution_time_part(task, i, j):
    return task[i][2 * j]


# Function: calculate the total computation time of task i
def execution_time(task, i, M):
    result = 0
    for j in range(0, M[i]):
        result += execution_time_part(task, i, j)
    return result


def gen(n, m, c_max, utilization):
    task = [[0 for _ in range(2 * m - 1)] for _ in range(n)]
    M = [m for _ in range(n)]
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
    for i in range(n):
        for j in range(m):
            task[i][2 * j] = randint(1, c_max)
        # for j in range(m - 1):
        #     task[i][2 * j + 1] = randint(1, s_max)
    # set Period & Deadline
    for i_ in range(n):
        T[i_] = int(execution_time(task, i_, M) / U[i_])
        D[i_] = T[i_]
    # set suspension segments
    for i in range(n):
        # for j in range(m):
        #     task[i][2 * j] = randint(1, c_max)
        for j in range(m - 1):
            s_max_temp = T[i] - execution_time(task, i, M)
            task[i][2 * j + 1] = randint(max(int(0.1 * s_max_temp / (m - 1)), 1), int(0.6 * s_max_temp / (m - 1)))
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
