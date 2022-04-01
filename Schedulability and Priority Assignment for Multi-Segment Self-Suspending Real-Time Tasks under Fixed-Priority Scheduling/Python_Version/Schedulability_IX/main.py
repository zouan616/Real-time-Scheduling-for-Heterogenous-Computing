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
m = 20
c_max = 10
s1 = 0.6
s2 = 1
# s = [0.1, 0.6](T - C)
num_cpu = 2
utilization_net = 0.5
# if os.path.exists("log.txt"):
#     os.remove("log.txt")
if os.path.exists("out.txt"):
    os.remove("out.txt")
if os.path.exists("out2.txt"):
    os.remove("out2.txt")
if os.path.exists("out3.txt"):
    os.remove("out3.txt")

# failed_count = 0
count = 100
for _ in range(count):
    # os.system("generator.py")
    generator.gen(N, m, c_max, utilization_net, s1, s2)
    # time.sleep(1)
    # os.system("calc.py")
    os.system("calc.py >> out.txt")
    os.system("calc_2.py >> out2.txt")
    os.system("calc_3.py >> out3.txt")

with open("out.txt", "r") as f:
    passed_count = 0
    line = f.readline()
    while line:
        if line[0: 16] == "All tasks passed":
            passed_count += 1
        line = f.readline()
f.close()
print("New mth: Passed rate = ", 100 * passed_count / count, "%")


with open("out2.txt", "r") as f:
    passed_count = 0
    line = f.readline()
    while line:
        if line[0: 16] == "All tasks passed":
            passed_count += 1
        line = f.readline()
f.close()
print("Stupid old mth: Passed rate = ", 100 * passed_count / count, "%")


with open("out3.txt", "r") as f:
    passed_count = 0
    line = f.readline()
    while line:
        if line[0: 16] == "All tasks passed":
            passed_count += 1
        line = f.readline()
f.close()
print("XDM: Passed rate = ", 100 * passed_count / count, "%")
