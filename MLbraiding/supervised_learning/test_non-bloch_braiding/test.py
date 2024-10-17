import os
import cmath
from matplotlib import pyplot as plt, rcParams
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import shuffle
from torch.utils.data import random_split, TensorDataset, DataLoader
import torch.optim
from torch.utils.data import Dataset
import sys
sys.path.append("..")

import matplotlib.colors as mcolors
from train_bloch_braiding.mode import *  # Assuming `mode` module is defined elsewhere

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size=28
labelpad1=-12
config={
    "font.family": 'Times New Roman',        # 默认字体
    "font.size":size,                         # 默认字体大小
    "font.weight": 'normal',                   # 默认字体粗细
    "mathtext.fontset": 'stix',              # 数学文本的字体集

    "figure.figsize": (8, 6),                # 默认图形大小 (宽, 高) 单位为英寸
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


def energy_windingly(t,delta,m,m0,L):
    k = np.linspace(0, 2 * np.pi, L, endpoint=True)
    r = cmath.sqrt(abs((t + delta + 1j * m0) / (t - delta - 1j * m0)))
    beta = r * np.exp(1j * k)
    EZ = m + 0.5 * (t - delta - 1j * m0) * beta + 0.5 * (t + delta + 1j * m0) / beta
    EF = -m + 0.5 * (t - delta + 1j * m0) * beta + 0.5 * (t + delta - 1j * m0) / beta
    # E = EZ - EF
    E = np.square(EZ - EF)
    theta = np.angle(E)
    delta_theta = theta[1:] - theta[:-1]
    delta_theta[delta_theta >= np.pi] -= 2 * np.pi
    delta_theta[delta_theta < -np.pi] += 2 * np.pi
    w =((np.sum(delta_theta)/2/np.pi))
    return np.round(w)

def test_ly(L):
    ALL = []
    W = []
    T2 = []
    T3 = []
    t = 1
    delta=0.9
    n1 = 100
    n2 = 100
    for m in np.linspace(0.00000001, 1.2, n1):
        for m0 in np.linspace(0.00000001, 1, n2):
            tt1 = []
            tt2 = []
            W.append(energy_windingly(t,delta,m,m0,L))
            T2.append(m)
            T3.append(m0)
            k = np.linspace(0, 2 * np.pi, L)
            r = cmath.sqrt(abs((t + delta + 1j * m0) / (t - delta - 1j * m0)))
            beta = r * np.exp(1j * k)
            EZ = m + 0.5 * (t - delta - 1j * m0) * beta + 0.5 * (t + delta + 1j * m0) / beta
            EF = -m + 0.5 * (t - delta + 1j * m0) * beta + 0.5 * (t + delta - 1j * m0) / beta
            E = np.square(EZ - EF)
            # E = EZ - EF
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
    plt.title("lytrue")
    plt.savefig("ly_true.png")
    plt.show()
    data = ALL.reshape(n1 * n2,1, L, 2, order='F')
    X_test, Y_test = torch.FloatTensor(data), torch.FloatTensor(W)
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
    y_pre1 = np.round(Y)
    # np.savetxt("y_pre.csv", y_pre1, delimiter=',')
    print('在测试集上的绝对值误差为:', mae)
    R2 = r2_score(Y_test.cpu(), y_pre1)
    print("r2:{}".format(R2))
    m0=np.linspace(0, 1, n1)
    r = np.sqrt(abs((t + delta + 1j * m0) / (t - delta - 1j * m0)))
    m=m0*(r**2+1)/2/r
    # plt.plot(m,m0,color='black')
    colors1 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8)]
    colors2 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8)]  # 自定义颜色列表
    colors22 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8),(0.1, 0.5, 0.6)]
    colors3 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8), (0.1, 0.5, 0.6), (0.1, 0.8, 0.6)]
    colors4 = [(0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (0.5, 0.8, 0.8), (0.2, 0.4, 0.8), (0.1, 0.5, 0.6), (0.1, 0.8, 0.6),(0.3, 0.8, 0.6),(0.3, 0.1, 0.6)]
    # bounds = [0, 1, 2, 3, 4, 5,6] # 自定义颜色边界
    # cmap = mcolors.ListedColormap(colors1)
    plt.contourf(T2.reshape(n1, n2), T3.reshape(n1, n2), y_pre1.reshape(n1, n2), cmap='hsv')
    plt.plot(m, m0, color='black',linewidth=2,linestyle='--')
    plt.colorbar()
    # plt.title("lypre")
    plt.xticks([0.00000001,0.6,1.2],labels=['0','0.6','1.2'])
    plt.yticks([0.00000001, 0.5, 1], labels=['0', '0.5', '1'])
    plt.xlabel("$\gamma$",fontsize=32)
    plt.ylabel("$m$",fontsize=32)
    plt.savefig("lypre3.svg",bbox_inches='tight',dpi=300, transparent=True,pad_inches=0)
    plt.show()

if __name__ == "__main__":
    L=201
    test_ly(L)
