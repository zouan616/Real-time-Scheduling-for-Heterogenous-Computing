import matplotlib.pyplot as plt

#  M = 2  utilization = (C + S) / T
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
        1, 1, 0.999, 0.998,
        0.997, 0.994, 0.988, 0.977,
        0.967, 0.943, 0.918, 0.907,
        0.874, 0.831, 0.806, 0.777,
        0.712, 0.656, 0.608, 0.556,
        0.483, 0.440, 0.406, 0.358,
        0.321, 0.274, 0.238, 0.188,
        0.160, 0.137, 0.113, 0.093,
        0.081, 0.069, 0.055, 0.047,
        0.034, 0.029, 0.013, 0.016,
        0.012, 0.012, 0.009, 0.006,
        0.002, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0]

Acc2 = [1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 0.998,
        0.997, 0.994, 0.994, 0.993,
        0.993, 0.986, 0.978, 0.975,
        0.957, 0.937, 0.904, 0.902,
        0.846, 0.821, 0.756, 0.701,
        0.608, 0.533, 0.487, 0.417,
        0.336, 0.274, 0.242, 0.191,
        0.137, 0.090, 0.067, 0.065,
        0.040, 0.027, 0.017, 0.014,
        0.011, 0.006, 0.005, 0.005,
        0.002, 0.002, 0.002, 0.001,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0]

Acc3 = [1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 0.999,
        0.999, 0.994, 0.993, 0.987,
        0.982, 0.964, 0.949, 0.915,
        0.907, 0.866, 0.815, 0.795,
        0.743, 0.696, 0.634, 0.591,
        0.542, 0.521, 0.441, 0.423,
        0.351, 0.296, 0.278, 0.228,
        0.209, 0.169, 0.124, 0.114,
        0.094, 0.077, 0.053, 0.048,
        0.037, 0.028, 0.014, 0.014,
        0.011, 0.009, 0.008, 0.001,
        0.001, 0.002, 0.001, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0]

Acc4 = [1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 0.97, 0.90,
        0.81, 0.69, 0.60, 0.44,
        0.37, 0.29, 0.25, 0.17,
        0.11, 0.09, 0.05, 0.03,
        0.02, 0.01, 0.01, 0,
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
        1, 0.963, 0.862, 0.726,
        0.482, 0.248, 0.073, 0.005,
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
