from random import randint


def execution_time_part(task, i, j):
    return task[i][2 * j]


# Function: calculate the total computation time of task i
def execution_time(task, i, M):
    result = 0
    for j in range(0, M[i]):
        result += execution_time_part(task, i, j)
    return result


def gen(n, m, c_max, s_max, utilization):
    task = [[0 for _ in range(2 * m - 1)] for _ in range(n)]
    M = [m for _ in range(n)]
    T = [0 for _ in range(n)]
    D = [0 for _ in range(n)]
    for i in range(n):
        for j in range(m):
            task[i][2 * j] = randint(1, c_max)
        for j in range(m - 1):
            task[i][2 * j + 1] = randint(1, s_max)
    for i_ in range(n):
        T[i_] = int(execution_time(task, i_, M) / utilization)
        D[i_] = T[i_]
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
