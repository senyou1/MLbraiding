import os
import random
import numpy as np
from matplotlib import pyplot as plt, rcParams
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.utils import shuffle
from torch.utils.data import random_split, TensorDataset, DataLoader
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import multiprocessing as mp
from torch.utils.data import ConcatDataset
import matplotlib.colors as mcolors
import sys
import seaborn as sns
from sklearn.metrics import confusion_matrix
sys.path.append("..")

from train_bloch_braiding.mode import *  # Assuming `mode` module is defined elsewhere

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
seed = int(1)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.double)
        self.labels = torch.tensor(labels, dtype=torch.double)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def energy_winding2(t0, tm, tn, m, n, L):
    k = np.linspace(0, 2 * np.pi, L, endpoint=True)
    Ek1 = np.sqrt(t0 * (t0 + tm * np.exp(1j * m * k) + tn * np.exp(1j * n * k)))
    Ek2 = -np.sqrt(t0 * (t0 + tm * np.exp(1j * m * k) + tn * np.exp(1j * n * k)))
    # E = np.square(Ek1)
    E=np.square(Ek1-Ek2)
    # E=Ek1-Ek2
    theta = np.angle(E)
    delta_theta =(theta[1:] - theta[:-1])
    delta_theta[delta_theta >= np.pi] -= 2 * np.pi
    delta_theta[delta_theta < -np.pi] += 2 * np.pi
    w = np.round((np.sum(delta_theta) / 2 / np.pi))
    return w
def test(M,N,L):
    ALL = []
    W = []
    T2 = []
    T3 = []
    t0 = 1
    m = M
    n = N
    n1 = 300
    n2 = 300
    for tm in np.linspace(-4, 4, n1):
        for tn in np.linspace(-4, 4, n2):
            tt1 = []
            tt2 = []
            W.append(energy_winding2(t0, tm, tn, m, n, L))
            T2.append(tm)
            T3.append(tn)
            k = np.linspace(0, 2 * np.pi, L)
            Ek1 = np.sqrt(t0 * (t0 + tm * np.exp(1j * m * k) + tn * np.exp(1j * n * k)))
            Ek2 = -np.sqrt(t0 * (t0 + tm * np.exp(1j * m * k) + tn * np.exp(1j * n * k)))
            E = np.square(Ek1-Ek2)
            # E =(Ek1 - Ek2)
            for i in E:
                tt1.append(i.real / np.sqrt(i.real ** 2 + i.imag ** 2))
                tt2.append(i.imag / np.sqrt(i.real ** 2 + i.imag ** 2))
            ALL.append(tt1 + tt2)
    ALL = np.array(ALL)
    W = np.array(W)
    T2 = np.array(T2)
    T3 = np.array(T3)
    plt.contourf(T2.reshape(n1, n2), T3.reshape(n1, n2), W.reshape(n1, n2), cmap='hsv')
    plt.colorbar()
    plt.title('True phase of H({},{})'.format(m, n))
    plt.savefig('True_H({},{}).svg'.format(m, n), dpi=300)
    plt.show()
    data = ALL.reshape(n1 * n2, 1,L, 2, order='F')
    X_test, Y_test = torch.FloatTensor(data), torch.FloatTensor(W)
    print(X_test.shape)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    model = Tudui(L).to(device)
    print(model)
    model.load_state_dict(torch.load('../train_bloch_braiding/checkpoint.pt'))
    model = model.to(device)
    model.eval()
    Y = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_pre = model(x)
            Y.append(y_pre.cpu().squeeze(1).tolist())
    Y = np.array(Y).reshape(-1, 1).astype(float)

    mae = mean_absolute_error(Y_test.cpu(), Y)
    Y=np.round(Y)
    print('mae:', mae)
    R2 = r2_score(Y_test.cpu(), Y)
    print("r2:{}".format(R2))
    colors1 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8)]
    colors2 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8)]  # 自定义颜色列表
    colors22 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8),(0.1, 0.5, 0.6)]
    colors3 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8), (0.1, 0.5, 0.6), (0.1, 0.8, 0.6)]
    colors44 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8), (0.1, 0.5, 0.6), (0.1, 0.8, 0.6),
               (0.3, 0.8, 0.6)]
    colors4 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8), (0.1, 0.5, 0.6), (0.1, 0.8, 0.6),(0.3, 0.8, 0.6),(0.1, 0.1, 0.6)]
    # bounds = [0, 1, 2, 3, 4, 5,6] # 自定义颜色边界
    cmap = mcolors.ListedColormap(colors4)
    plt.contourf(T2.reshape(n1, n2), T3.reshape(n1, n2), Y.reshape(n1, n2), cmap="hsv")
    # , vmin = 0, vmax = 5
    cbar = plt.colorbar()
    # 设置 colorbar 的刻度位置
    # cbar.set_ticks([0.4, 1.5, 2.5,3.5,4.5,5.5])
    # cbar.set_ticklabels(['0', '1', '2', '3'])
    plt.title("H({},{})".format(m,n),fontsize=32)
    plt.xlabel('$t_{}$'.format(m),fontsize=32,labelpad=labelpad1)
    plt.ylabel('$t_{}$'.format(n),fontsize=32,labelpad=labelpad1)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("pre_H({},{}).svg".format(m, n),bbox_inches='tight',dpi=300, transparent=True)
    plt.show()
    plt.close()
    # 计算混淆矩阵
    cm = confusion_matrix(W, Y)

    # 使用 seaborn 绘制热度图
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,annot_kws={"size": 24})
    # 添加标签和标题
    plt.xlabel('Predicted')
    plt.ylabel('True')
    acc = accuracy_score(W, Y)
    plt.title("H({},{})".format(m, n))
    plt.savefig("pre_H_confusion({},{}).svg".format(m, n), bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

    print("H({},{})".format(m, n),acc)
if __name__ == "__main__":
    m=[3]
    n=[5]
    L=201
    for i,j in zip(m,n):
        test(i,j,L)

    # model = Tudui().to(device)
    # model.load_state_dict(torch.load('../train/checkpoint.pt'))
    # model = model.to(device)
    # print(model)
