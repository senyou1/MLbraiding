import cmath
import math
import os
import random
import time
from matplotlib import rcParams
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import shuffle
from torch.utils.data import random_split, TensorDataset, DataLoader
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import multiprocessing as mp
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from mode import *  # Assuming `mode` module is defined elsewhere

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size=32
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


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def energy_winding2(t0, tm, tn, m, n, L):
    k = np.linspace(0, 2 * np.pi, L, endpoint=True)
    Ek1 = np.sqrt(t0 * (t0 + tm * np.exp(1j * m * k) + tn * np.exp(1j * n * k)))
    Ek2 = -np.sqrt(t0 * (t0 + tm * np.exp(1j * m * k) + tn * np.exp(1j * n * k)))
    # E = np.square(Ek1)
    E=np.square(Ek1-Ek2)
    # E=Ek1-Ek2
    theta = np.angle(E)
    delta_theta = (theta[1:] - theta[:-1])
    delta_theta[delta_theta >= np.pi] -= 2 * np.pi
    delta_theta[delta_theta < -np.pi] += 2 * np.pi
    w = (np.round(np.sum(delta_theta) / 2 / np.pi))
    return w
def generate_data(params):
    tm, tn, t0, m, n, L = params
    tt1 = []
    tt2 = []
    W = energy_winding2(t0, tm, tn, m, n, L)
    T2 = tm
    T3 = tn
    if np.random.uniform(0, 100, size=1)<1:
        t0 = t0 + np.random.uniform(-0.2 * t0, 0.2 * t0, size=1)
        # tm = tm+np.random.uniform(-0.1 * tm, 0.1 * tm, size=1)
        # tn = tn+np.random.uniform(-0.1 * tn, 0.1 * tn, size=1)
    k = np.linspace(0, 2 * np.pi, L)
    Ek1 = np.sqrt(t0 * (t0 + tm * np.exp(1j * m * k) + tn * np.exp(1j * n * k)))
    Ek2 = -np.sqrt(t0 * (t0 + tm * np.exp(1j * m * k) + tn * np.exp(1j * n * k)))
    E = np.square(Ek1-Ek2)
    # E = (Ek1 - Ek2)
    for i in E:
        tt1.append(i.real / np.sqrt(i.real ** 2 + i.imag ** 2))
        tt2.append(i.imag / np.sqrt(i.real ** 2 + i.imag ** 2))
    return tt1 + tt2, W, T2, T3

def create_dataset(ALL, W, n1,n2, T2, T3,L,m,n):
    ALL = np.array(ALL)
    W = np.array(W)
    T2 = np.array(T2)
    T3 = np.array(T3)
    plt.contourf(T2.reshape(n1, n2), T3.reshape(n1, n2), W.reshape(n1, n2), cmap='hsv')
    plt.colorbar()
    plt.title('True phase of H({},{})'.format(m,n))
    plt.savefig('H({},{}).pdf'.format(m,n),dpi=300)
    plt.show()
    print(ALL.shape)
    all = ALL.reshape(n1 * n2,1, L, 2, order='F')

    dataset1 = CustomDataset(all, W)
    dataset = shuffle(dataset1, random_state=seed)
    return dataset


def train_model(train_loader, val_loader, model, optimizer, scheduler, early_stopping, loss_fn, n_epochs):
    avg_train_losses = []
    avg_val_losses = []
    kkk=0
    for i in range(1, n_epochs + 1):
        kkk=kkk+1
        train_loss = 0.
        model.train()
        with tqdm(total=len(train_loader), desc=f'Epoch {i}/{n_epochs}', unit='batch', leave=False) as pbar:
            for data in train_loader:
                x, y = data
                x = x.to(device)
                y = y.unsqueeze(1).to(device)
                optimizer.zero_grad()
                outputs = model(x)
                # print(outputs.shape)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'Training Loss': loss.item()})
        epoch_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for data in val_loader:
                x, y = data
                x = x.to(device)
                y = y.unsqueeze(1).to(device)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        avg_train_losses.append(epoch_train_loss)
        avg_val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print_msg = f'Epoch [{i}/{n_epochs}] train_loss: {epoch_train_loss:.5f} val_loss: {epoch_val_loss:.5f} lr: {current_lr:.7f}'
        print(print_msg)


        early_stopping(epoch_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(kkk), avg_train_losses, c='blue',label='Training set')
    plt.plot(np.arange(kkk), avg_val_losses, c='red',label='Validation set')
    plt.xticks([0, 24,49], labels=['1', '25','50'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(framealpha=0,fontsize=20)
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    np.savetxt('avg_train_losses.csv',avg_train_losses,fmt='%f',delimiter=',')
    np.savetxt('avg_val_losses.csv', avg_val_losses, fmt='%f', delimiter=',')
    plt.savefig("loss.svg", dpi=300, bbox_inches='tight', transparent=True)


def train(L):
    t0 = 1
    m = 1
    n = 3
    n1 = 300
    n2 = 300
    T2 = []
    T3 = []
    ALL = []
    W = []
    pool = mp.Pool(processes=mp.cpu_count())
    params = [(tm, tn, t0, m, n, L) for tm in np.linspace(-4, 4, n1) for tn in np.linspace(-4, 4, n2)]
    results = pool.map(generate_data, params)
    pool.close()
    pool.join()

    for result in results:
        ALL.append(result[0])
        W.append(result[1])
        T2.append(result[2])
        T3.append(result[3])
    dataset1 = create_dataset(ALL, W, n1,n2, T2, T3,L,m,n)

    t0 = 1
    m = 1
    n = 2
    n1 = 300
    n2 = 300
    T2 = []
    T3 = []
    ALL = []
    W = []
    pool = mp.Pool(processes=mp.cpu_count())
    params = [(tm, tn, t0, m, n, L) for tm in np.linspace(-4, 4, n1) for tn in np.linspace(-4, 4, n2)]
    results = pool.map(generate_data, params)
    pool.close()
    pool.join()
    for result in results:
        ALL.append(result[0])
        W.append(result[1])
        T2.append(result[2])
        T3.append(result[3])

    dataset2 = create_dataset(ALL, W, n1,n2, T2, T3,L,m,n)
    dataset=ConcatDataset([dataset1,dataset2])
    print(len(dataset))
    train_data, val_data = random_split(dataset=dataset, lengths=[0.7, 0.3],
                                        generator=torch.Generator().manual_seed(seed))

    batch_size = 50
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    n_epochs = 50
    learning_rate = 0.01

    model = Tudui(L).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    loss_fn = nn.MSELoss(reduction='sum').to(device)

    train_model(train_loader, val_loader, model, optimizer, scheduler, early_stopping, loss_fn, n_epochs)

if __name__ == "__main__":
    L=201
    train(L)
    # test(m,n)
    # test_SSH()
    # test_ly()
    # test_ziran()
    # print(energy_windingSSH(1, 1, 0.01,120))