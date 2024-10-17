import os
import random
from sklearn.utils import shuffle

import numpy as np
import torch
from matplotlib import pyplot as plt, rcParams
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import sys
import matplotlib.colors as mcolors
import sys
import seaborn as sns
sys.path.append("..")
from train_bloch_braiding.mode import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = int(1)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
size=28
labelpad1=-10
config={
    "font.family": 'Times New Roman',        # 默认字体
    "font.size":size,                         # 默认字体大小
    "font.weight": 'normal',                   # 默认字体粗细
    "mathtext.fontset": 'stix',              # 数学文本的字体集

    "figure.figsize": (7, 6),                # 默认图形大小 (宽, 高) 单位为英寸
    # "lines.linewidth": 2,                    # 默认线宽度
    # "lines.marker": 'o',                     # 默认线标记

    "axes.labelsize": size,                    # 默认坐标轴标签大小
    "axes.titlesize": size,                    # 默认坐标轴标题大小
    "xtick.labelsize": size,                   # 默认X轴刻度标签大小
    "ytick.labelsize":size,
    # "xlabel.labelsize":size,
    # "ylabel.labelsize":size,

    # 默认Y轴刻度标签大小
    # "axes.linewidth": 2,
    # "legend.fontsize": 14,                   # 默认图例字体大小
    # "xtick.major.width": 3,
    # "ytick.major.width": 3,
    # "grid.linestyle": '--',                  # 默认网格线样式
    # "grid.linewidth": 0.5,                   # 默认网格线宽度
    # "grid.alpha": 0.5,                       # 默认网格线透明度

    "savefig.dpi": 300,
}
rcParams.update(config)
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.double)
        self.labels = torch.tensor(labels, dtype=torch.double)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_dataset(ALL, W, n1,n2, T2, T3,L,m,n):
    ALL = np.array(ALL)
    W = np.array(W)
    T2 = np.array(T2)
    T3 = np.array(T3)
    plt.contourf(T2.reshape(n1, n2), T3.reshape(n1, n2), W.reshape(n1, n2), cmap='hsv')
    plt.colorbar()
    plt.title('True phase of H({},{})'.format(m,n))
    plt.savefig('H({},{}).svg'.format(m,n),dpi=300)
    plt.show()
    print(ALL.shape)
    # all = ALL.reshape(n1 * n2,1, L, 2, order='F')

    dataset1 = CustomDataset(ALL, W)
    dataset = shuffle(dataset1, random_state=seed)
    return dataset
def energy_winding2(t0, tm, tn, m, n, L):
    Ek1=[]
    Ek2=[]
    Ek3=[]
    gamma=0.4
    # delta=1
    # t0=0
    for k in np.linspace(0, 2 * np.pi, L, endpoint=True):
        HK=np.zeros((3,3),dtype=complex)
        HK[0,1]=t0+gamma*1j
        HK[1,0]=t0-gamma*1j
        HK[1, 2] = t0+gamma*1j
        HK[2,0]=tm*np.exp(1j*m*k)+tn*np.exp(1j*n*k)
        HK[2, 1] =t0-gamma*1j

        # HK[0,1]=t0+delta*np.exp(gamma*1j)
        # HK[1,0]=t0-delta*np.exp(-gamma*1j)
        # HK[1, 2] = t0+delta*np.exp(gamma*1j)
        # HK[2,0]=tm*np.exp(1j*m*k)+tn*np.exp(1j*n*k)
        # HK[2, 1] =t0-delta*np.exp(-gamma*1j)

        val,vec=np.linalg.eig(HK)
        Ek1.append(min(val))
        Ek2.append(max(val))
        Ek3.append(sum(val)-max(val)-min(val))
    Ek1=np.array(Ek1)
    Ek2 = np.array(Ek2)
    Ek3 = np.array(Ek3)
    E1=np.square(Ek1-Ek2)
    # E1 = (Ek1 - Ek2)
    theta = np.angle(E1)
    delta_theta = (theta[1:] - theta[:-1])
    delta_theta[delta_theta >= np.pi] -= 2 * np.pi
    delta_theta[delta_theta < -np.pi] += 2 * np.pi
    w1 = np.round((np.sum(delta_theta) / 2 / np.pi))
    E2=np.square(Ek1-Ek3)
    # E1 = (Ek1 - Ek3)
    theta = np.angle(E2)
    delta_theta = (theta[1:] - theta[:-1])
    delta_theta[delta_theta >= np.pi] -= 2 * np.pi
    delta_theta[delta_theta < -np.pi] += 2 * np.pi
    w2 = np.round((np.sum(delta_theta) / 2 / np.pi))
    E3=np.square(Ek3-Ek2)
    # E1 = (Ek3 - Ek2)
    theta = np.angle(E3)
    delta_theta = (theta[1:] - theta[:-1])
    delta_theta[delta_theta >= np.pi] -= 2 * np.pi
    delta_theta[delta_theta < -np.pi] += 2 * np.pi
    w3 = np.round((np.sum(delta_theta) / 2 / np.pi))
    W=w1+w2+w3
    # 归一化实部和虚部
    tt1_1 = E1.real / np.sqrt(E1.real ** 2 + E1.imag ** 2)
    tt2_1 = E1.imag / np.sqrt(E1.real ** 2 + E1.imag ** 2)
    tt1_2 = E2.real / np.sqrt(E2.real ** 2 + E2.imag ** 2)
    tt2_2 = E2.imag / np.sqrt(E2.real ** 2 + E2.imag ** 2)
    tt1_3 = E3.real / np.sqrt(E3.real ** 2 + E3.imag ** 2)
    tt2_3 = E3.imag / np.sqrt(E3.real ** 2 + E3.imag ** 2)
    return np.column_stack((tt1_1,tt2_1))[np.newaxis, :], np.column_stack((tt1_2,tt2_2))[np.newaxis, :],np.column_stack((tt1_3,tt2_3))[np.newaxis, :],W, tm, tn
def test_3dai(M,N):
    ALL1 = []
    ALL2 = []
    ALL3 = []
    W = []
    T2 = []
    T3 = []
    n1 = 300
    n2 = 300
    L = 201
    m=M
    n=N
    t0=1
    for tm in np.linspace(0, 4, n1):
        for tn in np.linspace(0, 4, n2):
            T2.append(tm)
            T3.append(tn)
            L1,L2,L3,w,tmm,tnn=energy_winding2(t0, tm, tn, m, n, L)
            ALL1.append(L1)
            ALL2.append(L2)
            ALL3.append(L3)
            W.append(w)
    ALL1 = np.array(ALL1)
    ALL2 = np.array(ALL2)
    ALL3 = np.array(ALL3)
    W = np.array(W)
    T2 = np.array(T2)
    T3 = np.array(T3)
    plt.contourf(T2.reshape(n1, n2), T3.reshape(n1, n2), W.reshape(n1, n2), cmap='hsv')
    plt.colorbar()
    plt.title('True phase of H({},{})'.format(m, n))
    plt.savefig('True_H({},{}).svg'.format(m, n), dpi=300)
    plt.show()

    X_test1, Y_test1 = torch.FloatTensor(ALL1), torch.FloatTensor(W)
    X_test2, Y_test2 = torch.FloatTensor(ALL2), torch.FloatTensor(W)
    X_test3, Y_test3 = torch.FloatTensor(ALL3), torch.FloatTensor(W)
    test_dataset1 = torch.utils.data.TensorDataset(X_test1, Y_test1)
    test_dataset2 = torch.utils.data.TensorDataset(X_test2, Y_test2)
    test_dataset3 = torch.utils.data.TensorDataset(X_test3, Y_test3)
    test_loader1 = DataLoader(dataset=test_dataset1, batch_size=100, shuffle=False)
    test_loader2 = DataLoader(dataset=test_dataset2, batch_size=100, shuffle=False)
    test_loader3 = DataLoader(dataset=test_dataset3, batch_size=100, shuffle=False)

    model = Tudui(L).to(device)
    print(model)
    model.load_state_dict(torch.load('../train_bloch_braiding/checkpoint.pt'))
    model = model.to(device)
    model.eval()
    Y = []
    with torch.no_grad():
        for x, y in test_loader1:
            x = x.to(device)
            y_pre = model(x)
            Y.append(y_pre.cpu().squeeze(1).tolist())
    Y1 = np.array(Y).reshape(-1, 1).astype(float)
    y_pre1 = np.round(Y1)
    Y = []
    with torch.no_grad():
        for x, y in test_loader2:
            x = x.to(device)
            y_pre = model(x)
            Y.append(y_pre.cpu().squeeze(1).tolist())
    Y1 = np.array(Y).reshape(-1, 1).astype(float)
    y_pre2 = np.round(Y1)
    Y = []
    with torch.no_grad():
        for x, y in test_loader3:
            x = x.to(device)
            y_pre = model(x)
            Y.append(y_pre.cpu().squeeze(1).tolist())
    Y1 = np.array(Y).reshape(-1, 1).astype(float)
    y_pre3 = np.round(Y1)
    y=y_pre1+y_pre3+y_pre2
    # colors1 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8)]
    # colors2 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8)]  # 自定义颜色列表
    # colors22 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8),(0.1, 0.5, 0.6)]
    # colors3 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8), (0.1, 0.5, 0.6), (0.1, 0.8, 0.6)]
    colors4 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8), (0.1, 0.5, 0.6), (0.1, 0.8, 0.6),(0.3, 0.8, 0.6),(0.3, 0.1, 0.6)]
    # # bounds = [0, 1, 2, 3, 4, 5,6] # 自定义颜色边界
    cmap = mcolors.ListedColormap(colors4)
    plt.figure(figsize=(7,6))
    plt.contourf(T2.reshape(n1, n2), T3.reshape(n1, n2), y.reshape(n1, n2), cmap=cmap)
    # plt.contourf(T2.reshape(n1, n2), T3.reshape(n1, n2), y.reshape(n1, n2), cmap='hsv')
    # plt.colorbar()
    xx=[0.6,0.6,0.6,0.6,0.6]
    yy=[0.2,0.7,1.1,1.6,2.2]
    plt.scatter(xx,yy,color='black')
    plt.xlabel('$t_{}$'.format(m),fontsize=32,labelpad=labelpad1)
    plt.ylabel('$t_{}$'.format(n),fontsize=32,labelpad=10)
    plt.title('H({},{})'.format(m, n))
    plt.savefig('preH({},{}).svg'.format(m, n), dpi=300,transparent=True)
    plt.show()
    # 计算混淆矩阵
    cm = confusion_matrix(W, y)

    # 使用 seaborn 绘制热度图
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,annot_kws={"size": 24})
    # 添加标签和标题
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("H({},{})".format(m, n))
    plt.savefig("pre_H_confusion({},{}).svg".format(m, n), bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    # return y_pre1,y_pre2,y_pre3
# a,b,c=test_3dai()
def plot_energy(tm,tn,m,n):
    L=201
    t0=1
    # m=1
    # n=3
    Ek1=[]
    Ek2=[]
    Ek3=[]
    gamma=0.4
    for k in np.linspace(0, 2 * np.pi, L, endpoint=True):
        HK=np.zeros((3,3),dtype=complex)
        HK[0,1]=t0+gamma*1j
        HK[1,0]=t0-gamma*1j
        HK[1, 2] = t0+gamma*1j
        HK[2,0]=tm*np.exp(1j*m*k)+tn*np.exp(1j*n*k)
        HK[2, 1] =t0-gamma*1j
        val,vec=np.linalg.eig(HK)
        Ek1.append(min(val))
        Ek2.append(max(val))
        Ek3.append(sum(val)-max(val)-min(val))
    # Ek1=np.array(Ek1)
    # Ek2 = np.array(Ek2)
    # Ek3 = np.array(Ek3)
    for i in range(2, len(Ek1)):
        if abs(Ek1[i] - Ek1[i - 1]) >= abs(Ek1[i] - Ek2[i - 1]) and abs(Ek2[i] - Ek2[i - 1]) >= abs(Ek2[i] - Ek1[i - 1]):
            c1 = Ek1[i]
            c2 = Ek2[i]
            Ek1[i] = c2
            Ek2[i] = c1

        if abs(Ek1[i] - Ek1[i - 1]) >= abs(Ek1[i] - Ek3[i - 1]) and abs(Ek3[i] - Ek3[i - 1]) >= abs(Ek3[i] - Ek1[i - 1]):
            c1 = Ek1[i]
            c2 = Ek3[i]
            Ek1[i] = c2
            Ek3[i] = c1

        if abs(Ek2[i] - Ek2[i - 1]) >= abs(Ek2[i] - Ek3[i - 1]) and abs(Ek3[i] - Ek3[i - 1]) >= abs(Ek3[i] - Ek2[i - 1]):
            c1 = Ek2[i]
            c2 = Ek3[i]
            Ek2[i] = c2
            Ek3[i] = c1

        if abs(Ek1[i] - Ek1[i - 1]) >= abs(Ek1[i] - Ek2[i - 1]) and abs(Ek2[i] - Ek2[i - 1]) >= abs(Ek2[i] - Ek1[i - 1]):
            c1 = Ek1[i]
            c2 = Ek2[i]
            Ek1[i] = c2
            Ek2[i] = c1
    a = Ek1 + Ek2+Ek3
    L=201
    z = np.linspace(0, 2 * np.pi, L)
    Z = np.hstack((z, z,z))

    colors = ['red'] * L+ ['blue'] * L+ ['green'] * L
    # color_to_pixel = {
    #     'red': [255, 0, 0],
    #     'blue': [0, 0, 255],
    #     'green': [94, 170, 38]  # 修改 green 的值
    # }
    #
    # # 使用列表推导将颜色替换为像素值
    # colors = [color_to_pixel[color] for color in colors]
    elev = 24
    azim = 75
    fig = plt.figure(figsize=(4,6))
    ax10 = fig.add_subplot(111, projection='3d')
    ax10.scatter(np.array(a).real, np.array(a).imag, Z, color=colors, s=5)
    ax10.scatter(np.array(a).real, np.array(a).imag, zs=0, zdir='z', linewidth=0.5, color=colors, s=5)
    # ax10.scatter(0, 0, 0, color='black', s=12)
    ax10.view_init(elev=elev, azim=azim)
    ax10.set_box_aspect([1, 1, 1.5])
    ax10.set_zticks([0, 2 * np.pi / 2, 2 * np.pi], labels=['0', '$\pi$', '$2\pi$'], fontsize=18)
    ax10.tick_params(axis='z', pad=5)
    ax10.set_xticks([])
    ax10.set_yticks([])
    ax10.set_xlabel('$Re[E]$', labelpad=labelpad1, fontsize=18)
    ax10.set_ylabel('$Im[E]$', labelpad=labelpad1, fontsize=18)
    ax10.set_zlabel('$k$', fontsize=20, labelpad=10)
    ax10.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax10.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax10.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.savefig("000.svg", dpi=300, transparent=True)
    plt.show()
if __name__ == "__main__":
    # m=[1,1,2,3,3]
    # n=[2,3,3,4,5]
    # L=201
    # for i,j in zip(m,n):
    test_3dai(1,2)

    #对于1，3
    #0,1,1.4,2.0
    # plot_energy(0.5,0)

    #对于1，2
    #x=0.6
    # 0.1, 0.7, 1.6, 2
    # (2,1) (-2,1)
    # plot_energy(2,-1,1,2)
    # plot_energy(-2, -1, 1, 2)
    # xx=[0.6,0.6,0.6,0.6,0.6]
    # yy=[0.2,0.7,1.1,1.6,2.2]
    plot_energy(0.6, 2.2, 1, 2)

# print(a,b,c)
# a=np.array([1,2])
# b=np.array([3,4])
# print(np.multiply(a,b))