import matplotlib.pyplot as plt

#  M = 5  utilization = (C + S) / T
U = [0.05, 0.1, 0.15, 0.2, 0.25,
     0.3, 0.35, 0.4, 0.45, 0.5,
     0.55, 0.6, 0.65, 0.7, 0.75,
     0.8, 0.85, 0.9, 0.95, 1,
     1.05, 1.1, 1.15, 1.2, 1.25,
     1.3, 1.35, 1.4, 1.45, 1.5,
     1.55, 1.6, 1.65, 1.7, 1.75,
     1.8, 1.85, 1.9, 1.95, 2,
     2.05, 2.1, 2.15, 2.2, 2.25,
     2.3, 2.35, 2.4, 2.45, 2.5,
     2.55, 2.6, 2.65, 2.7, 2.75,
     2.8, 2.85, 2.9, 2.95, 3,
     3.05, 3.1, 3.15, 3.2, 3.25,
     3.3, 3.35, 3.4, 3.45, 3.5,
     3.55, 3.6, 3.65, 3.7, 3.75,
     3.8, 3.85, 3.9, 3.95, 4]

Acc1 = [1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 0.998, 0.998,
        0.996, 0.993, 0.990, 0.978,
        0.968, 0.964, 0.951, 0.916,
        0.904, 0.868, 0.864, 0.848,
        0.777, 0.750, 0.695, 0.661,
        0.626, 0.567, 0.537, 0.499,
        0.424, 0.387, 0.367, 0.321,
        0.255, 0.250, 0.171, 0.144,
        0.123, 0.103, 0.078, 0.058,
        0.046, 0.032, 0.020, 0.018,
        0.014, 0.010, 0.003, 0,
        0, 0, 0, 0]

Acc2 = [1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 0.999, 0.998, 0.997,
        0.994, 0.984, 0.961, 0.941,
        0.901, 0.832, 0.784, 0.682,
        0.570, 0.511, 0.390, 0.317,
        0.215, 0.158, 0.117, 0.079,
        0.045, 0.023, 0.014, 0.006,
        0.004, 0.002, 0.001, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0]

Acc3 = [1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 0.998, 0.997, 0.997,
        0.989, 0.984, 0.970, 0.952,
        0.931, 0.914, 0.898, 0.824,
        0.812, 0.781, 0.730, 0.680,
        0.631, 0.588, 0.557, 0.494,
        0.444, 0.425, 0.336, 0.325,
        0.265, 0.226, 0.198, 0.165,
        0.116, 0.096, 0.070, 0.054,
        0.046, 0.032, 0.020, 0.026,
        0.006, 0.004, 0.003, 0.002,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0]

Acc4 = [1, 1, 1, 0.97,
        0.94, 0.79, 0.67, 0.50,
        0.39, 0.34, 0.26, 0.16,
        0.12, 0.07, 0.05, 0.04,
        0.02, 0.02, 0.01, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0]

Acc5 = [1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        0.987, 0.839, 0.309, 0.024,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0]

plt.title("")
plt.tick_params(labelsize=15)
# plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['figure.dpi'] = 800
plt.xlabel('Utilization', fontsize=17)
plt.ylabel('Acceptance ratio', fontsize=17)
plt.xlim(0, 4.0)
plt.ylim(0, 1.05)
plt.plot(U, Acc1, '+-', label='Our method', color='r', linewidth=1.2, markersize=10)
plt.plot(U, Acc1, color='r')
plt.plot(U, Acc2, 'x-', label='STGM', color='lime', linewidth=1.2, markersize=8)
plt.plot(U, Acc2, color='lime')
plt.plot(U, Acc3, '^-', label='SCAIR_OPA', color='purple', linewidth=1.2, markersize=8)
plt.plot(U, Acc3, color='purple')
plt.plot(U, Acc4, 'd-', label='MPCP', color='deepskyblue', linewidth=1.2, markersize=8)
plt.plot(U, Acc4, color='deepskyblue')
plt.plot(U, Acc5, 'o-', label='XDM', color='orange', linewidth=1.2, markersize=8)
plt.plot(U, Acc5, color='orange')
plt.grid(linestyle='-.')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
plt.legend(loc='upper right', fontsize=17)
plt.savefig('./SR.eps', format='eps', dpi=1000)
plt.show()
