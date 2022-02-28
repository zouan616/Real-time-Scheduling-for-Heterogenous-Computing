# import threading
# import time
#
# lockA = threading.Lock()
# lockB = threading.Lock()
#
#
# def printA(n):
#     if n < 0:
#         return
#     lockA.acquire()
#     print("+++")
#     lockB.release()
#     time.sleep(1)
#     printA(n - 1)
#
#
# def printB(n):
#     if n < 0:
#         return
#     lockB.acquire()
#     print("+++")
#     lockA.release()
#     time.sleep(1)
#     printB(n - 1)
#
#
# lockB.acquire()
# t1 = threading.Thread(target=printA, args=(10,))
# t2 = threading.Thread(target=printB, args=(10,))
# t1.start()
# t2.start()
# t1.join()
# t2.join()
import os
import time
import calc
import generator

# print("input integer N = ")
# N = int(input())
# print("input integer M = ")
# m = int(input())
# # M = [m for _ in range(N)]
# print("input integer C_max = ")
# c_max = int(input())
# print("input integer S_max = ")
# s_max = int(input())
# print("input utilization rate = ")
# utilization = float(input())
N = 5
m = 5
c_max = 100
s_max = 100
num_cpu = 2
utilization_net = 2
# if os.path.exists("log.txt"):
#     os.remove("log.txt")
if os.path.exists("out.txt"):
    os.remove("out.txt")

# failed_count = 0
count = 100
for _ in range(count):
    # os.system("generator.py")
    generator.gen(N, m, c_max, s_max, utilization_net)
    # time.sleep(1)
    # os.system("calc.py")
    os.system("calc.py >> out.txt")
    # test(N, failed_count)
# print("Passed rate = ", 100 - failed_count * 100 / count, "%")
with open("out.txt", "r") as f:
    passed_count = 0
    line = f.readline()
    while line:
        if line[0: 16] == "All tasks passed":
            passed_count += 1
        line = f.readline()
f.close()
print("Passed rate = ", 100 * passed_count / count, "%")
