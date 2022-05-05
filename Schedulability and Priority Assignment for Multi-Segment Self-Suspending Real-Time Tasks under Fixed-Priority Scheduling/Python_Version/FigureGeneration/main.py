import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image

matplotlib.rc('font', family='SimHei', weight='bold')
#  M = 2 的合并图
#  ----------------------------------------------------------------------------------
#  下列数字代表各方法在SR下能百分百pass的utilization最大值
SR_newMethod = 1.0
SR_SCAIR_OPA = 0.6
SR_STGM = 0.55
SR_MPCP = 0.01
SR_XDM = 0.01
SR = [SR_XDM, SR_MPCP, SR_SCAIR_OPA, SR_STGM, SR_newMethod]
#  下列数字代表各方法在SR下能pass的utilization最大值
#  数值 = 能pass的utilization最大值 - 能百分百pass的utilization最大值
SR_newMethod_2 = 0.8
SR_SCAIR_OPA_2 = 1.1
SR_STGM_2 = 1.25
SR_MPCP_2 = 1.14
SR_XDM_2 = 0.64
SR_2 = [SR_XDM_2, SR_MPCP_2, SR_SCAIR_OPA_2, SR_STGM_2, SR_newMethod_2]
#  ----------------------------------------------------------------------------------
MR_newMethod = 0.85
MR_SCAIR_OPA = 0.45
MR_STGM = 0.5
MR_MPCP = 0.01
MR_XDM = 0.01
MR = [MR_XDM, MR_MPCP, MR_SCAIR_OPA, MR_STGM, MR_newMethod]

MR_newMethod_2 = 0.65
MR_SCAIR_OPA_2 = 0.9
MR_STGM_2 = 0.85
MR_MPCP_2 = 0.44
MR_XDM_2 = 0.01
MR_2 = [MR_XDM_2, MR_MPCP_2, MR_SCAIR_OPA_2, MR_STGM_2, MR_newMethod_2]
#  ----------------------------------------------------------------------------------
LR_newMethod = 0.1
LR_SCAIR_OPA = 0.01
LR_STGM = 0.01
LR_MPCP = 0.01
LR_XDM = 0.01
LR = [LR_XDM, LR_MPCP, LR_SCAIR_OPA, LR_STGM, LR_newMethod]

LR_newMethod_2 = 0.89
LR_SCAIR_OPA_2 = 0.84
LR_STGM_2 = 0.59
LR_MPCP_2 = 0.54
LR_XDM_2 = 0.01
LR_2 = [LR_XDM_2, LR_MPCP_2, LR_SCAIR_OPA_2, LR_STGM_2, LR_newMethod_2]


#  ----------------------------------------------------------------------------------


def pic1():
    fig, ax = plt.subplots(figsize=(10, 7))
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    yticks = np.arange(len(SR))
    ax.set_xlim(0, 2.0)
    plt.tick_params(labelsize=15)
    algorithm_name = ['XDM', 'MPCP', 'SCAIR-OPA', 'STGM', 'Our method']
    ax.barh(yticks, SR, height=0.5, label='SR', color='red')
    ax.barh(yticks, SR_2, height=0.5, label='SR', left=SR, color='coral')
    ax.barh(yticks - 6, MR, height=0.5, label='MR', color='green')
    ax.barh(yticks - 6, MR_2, height=0.5, label='MR', left=MR, color='lightgreen')
    ax.barh(yticks - 12, LR, height=0.5, label='LR', color='purple')
    ax.barh(yticks - 12, LR_2, height=0.5, label='LR', left=LR, color='violet')
    ax.set_title("")
    ax.set_xlabel('Utilization', fontsize=17)
    ax.set_ylabel('', fontsize=17)
    ax.legend(loc='upper right', fontsize=17)
    ax.set_yticks(yticks)
    ax.set_yticklabels(algorithm_name, fontsize=16)
    plt.savefig('./pic1.png', transparent=True, format='png', dpi=800)
    # plt.show()


def pic1_1():
    fig, ax = plt.subplots(figsize=(10, 7))
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    yticks = np.arange(len(SR))
    ax.set_xlim(0, 2.0)
    plt.tick_params(labelsize=15)
    algorithm_name = ['XDM', 'MPCP', 'SCAIR-OPA', 'STGM', 'Our method']
    ax.barh(yticks, SR, height=0.5, label='SR', color='red')
    ax.barh(yticks, SR_2, height=0.5, label='SR', left=SR, color='coral')
    ax.barh(yticks - 6, MR, height=0.5, label='MR', color='green')
    ax.barh(yticks - 6, MR_2, height=0.5, label='MR', left=MR, color='lightgreen')
    ax.barh(yticks - 12, LR, height=0.5, label='LR', color='purple')
    ax.barh(yticks - 12, LR_2, height=0.5, label='LR', left=LR, color='violet')
    ax.set_title("")
    ax.set_xlabel('Utilization', fontsize=17)
    ax.set_ylabel('', fontsize=17)
    ax.legend(loc='upper right', fontsize=17)
    ax.set_yticks(yticks - 6)
    ax.set_yticklabels(algorithm_name, fontsize=16)
    plt.savefig('./pic1_1.png', transparent=True, format='png', dpi=800)
    # plt.show()


def pic1_2():
    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    yticks = np.arange(len(SR))
    ax.set_xlim(0, 2.0)
    plt.tick_params(labelsize=15)
    algorithm_name = ['XDM', 'MPCP', 'SCAIR-OPA', 'STGM', 'Our method']
    ax.barh(yticks, SR, height=0.5, label='SR', color='red')
    ax.barh(yticks, SR_2, height=0.5, label='SR', left=SR, color='coral')
    ax.barh(yticks - 6, MR, height=0.5, label='MR', color='green')
    ax.barh(yticks - 6, MR_2, height=0.5, label='MR', left=MR, color='lightgreen')
    ax.barh(yticks - 12, LR, height=0.5, label='LR', color='purple')
    ax.barh(yticks - 12, LR_2, height=0.5, label='LR', left=LR, color='violet')
    ax.set_title("")
    ax.set_xlabel('Utilization', fontsize=17)
    ax.set_ylabel('', fontsize=17)
    ax.legend(loc='upper right', fontsize=17)
    ax.set_yticks(yticks - 12)
    ax.set_yticklabels(algorithm_name, fontsize=16)
    plt.savefig('./pic1_2.png', transparent=True, format='png', dpi=800)
    # plt.show()


def pic2():
    type_1 = ['Our method', 'Our method', 'Our method']
    newMethod = [1.0, 0.85, 0.1]
    newMethod_2 = [0.8, 0.65, 0.89]
    SCAIR_OPA = [0.6, 0.45, 0.01]
    SCAIR_OPA_2 = [1.1, 0.9, 0.84]
    STGM = [0.55, 0.5, 0.01]
    STGM_2 = [1.25, 0.85, 0.59]
    MPCP = [0.01, 0.01, 0.01]
    MPCP_2 = [1.14, 0.44, 0.54]
    XDM = [0.01, 0.01, 0.01]
    XDM_2 = [0.64, 0.01, 0.01]
    fig, ax = plt.subplots(figsize=(16, 9))
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    ax.set_ylim(0, 2.0)
    ax.set_xlim(-0.2, 3.2)
    plt.tick_params(labelsize=17)
    xticks = np.arange(len(newMethod))
    ax.bar(xticks, newMethod, width=0.1, label='100% acceptance', color='red')
    ax.bar(xticks, newMethod_2, width=0.1, label='0-100% acceptance', bottom=newMethod, color='coral')
    ax.bar(xticks + 0.15, SCAIR_OPA, width=0.1, label='100% acceptance', color='green')
    ax.bar(xticks + 0.15, SCAIR_OPA_2, width=0.1, label='0-100% acceptance', bottom=SCAIR_OPA, color='lightgreen')
    ax.bar(xticks + 0.3, STGM, width=0.1, label='100% acceptance', color='purple')
    ax.bar(xticks + 0.3, STGM_2, width=0.1, label='0-100% acceptance', bottom=STGM, color='violet')
    ax.bar(xticks + 0.45, MPCP, width=0.1, label='100% acceptance', color='deepskyblue')
    ax.bar(xticks + 0.45, MPCP_2, width=0.1, label='0-100% acceptance', bottom=MPCP, color='lightblue')
    ax.bar(xticks + 0.6, XDM, width=0.1, label='100% acceptance', color='orange')
    ax.bar(xticks + 0.6, XDM_2, width=0.1, label='0-100% acceptance', bottom=XDM, color='wheat')
    ax.set_title("")
    ax.set_ylabel('Utilization', fontsize=19)
    ax.set_xlabel('', fontsize=17)
    ax.legend(bbox_to_anchor=(0.75, 0.9), fontsize=17)
    ax.set_xticks(xticks)
    ax.set_xticklabels(type_1, fontsize=14, rotation=45)
    plt.text(0.27, 1.85, "SR", fontsize=19, color='black')
    plt.text(1.27, 1.85, "MR", fontsize=19, color='black')
    plt.text(2.27, 1.85, "LR", fontsize=19, color='black')
    plt.savefig('./pic2_1.png', transparent=True, format='png', dpi=400)
    # plt.show()


def pic2_2():
    type_1 = ['SCAIR-OPA', 'SCAIR-OPA', 'SCAIR-OPA']
    newMethod = [1.0, 0.85, 0.1]
    fig, ax = plt.subplots(figsize=(16, 9))
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    ax.set_ylim(0, 2.0)
    ax.set_xlim(-0.2, 3.2)
    plt.tick_params(labelsize=17)
    xticks = np.arange(len(newMethod))
    ax.set_title("")
    ax.set_ylabel('Utilization', fontsize=19)
    ax.set_xlabel('', fontsize=17)
    ax.set_xticks(xticks + 0.15)
    ax.set_xticklabels(type_1, fontsize=14, rotation=45)
    plt.savefig('./pic2_2.png', transparent=True, format='png', dpi=400)


def pic2_3():
    type_1 = ['STGM', 'STGM', 'STGM']
    newMethod = [1.0, 0.85, 0.1]
    fig, ax = plt.subplots(figsize=(16, 9))
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    ax.set_ylim(0, 2.0)
    ax.set_xlim(-0.2, 3.2)
    plt.tick_params(labelsize=17)
    xticks = np.arange(len(newMethod))
    ax.set_title("")
    ax.set_ylabel('Utilization', fontsize=19)
    ax.set_xlabel('', fontsize=17)
    ax.set_xticks(xticks + 0.3)
    ax.set_xticklabels(type_1, fontsize=14, rotation=45)
    plt.savefig('./pic2_3.png', transparent=True, format='png', dpi=400)


def pic2_4():
    type_1 = ['MPCP', 'MPCP', 'MPCP']
    newMethod = [1.0, 0.85, 0.1]
    fig, ax = plt.subplots(figsize=(16, 9))
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    ax.set_ylim(0, 2.0)
    ax.set_xlim(-0.2, 3.2)
    plt.tick_params(labelsize=17)
    xticks = np.arange(len(newMethod))
    ax.set_title("")
    ax.set_ylabel('Utilization', fontsize=19)
    ax.set_xlabel('', fontsize=17)
    ax.set_xticks(xticks + 0.45)
    ax.set_xticklabels(type_1, fontsize=14, rotation=45)
    plt.savefig('./pic2_4.png', transparent=True, format='png', dpi=400)


def pic2_5():
    type_1 = ['XDM', 'XDM', 'XDM']
    newMethod = [1.0, 0.85, 0.1]
    fig, ax = plt.subplots(figsize=(16, 9))
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    ax.set_ylim(0, 2.0)
    ax.set_xlim(-0.2, 3.2)
    plt.tick_params(labelsize=17)
    xticks = np.arange(len(newMethod))
    ax.set_title("")
    ax.set_ylabel('Utilization', fontsize=19)
    ax.set_xlabel('', fontsize=17)
    ax.set_xticks(xticks + 0.6)
    ax.set_xticklabels(type_1, fontsize=14, rotation=45)
    plt.savefig('./pic2_5.png', transparent=True, format='png', dpi=400)


pic2()
pic2_2()
pic2_3()
pic2_4()
pic2_5()
st_1 = Image.open('./pic2_1.png')
st_2 = Image.open('./pic2_2.png')
st_3 = Image.open('./pic2_3.png')
st_4 = Image.open('./pic2_4.png')
st_5 = Image.open('./pic2_5.png')
st_1.paste(st_2, (0, 0), st_2)
st_1.paste(st_3, (0, 0), st_3)
st_1.paste(st_4, (0, 0), st_4)
st_1.paste(st_5, (0, 0), st_5)
st_1.save('./pic2.png', format='png')
st_1.show()

# pic1()
# pic1_1()
# pic1_2()
# st = Image.open('./pic1.png')
# st_1 = Image.open("./pic1_1.png")
# st_2 = Image.open("./pic1_2.png")
# st = st.resize(st_1.size)
# st = st.resize(st_2.size)
# st_1.paste(st, (0, 0), st)
# st_1.paste(st_2, (0, 0), st_2)
# st_1.show()
# st_1.save('./pic_combined.png', format='png')
# st_1.save('./pic_combined.eps', format='eps')
