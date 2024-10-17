import os
from mpl_toolkits.mplot3d import Axes3D
os.environ["OMP_NUM_THREADS"] = '1'
import cmath
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from mpmath import sqrtm
from sklearn.cluster import KMeans, SpectralClustering
np.random.seed(1)
labelpad1=-30
size=18
config={
    "font.family": 'Times New Roman',        # 默认字体
    "font.size":size,                         # 默认字体大小
    "font.weight": 'normal',                   # 默认字体粗细
    "mathtext.fontset": 'stix',              # 数学文本的字体集

    "figure.figsize": (15, 12),                # 默认图形大小 (宽, 高) 单位为英寸
    # "lines.linewidth": 2,                    # 默认线宽度
    # "lines.marker": 'o',                     # 默认线标记

    "axes.labelsize": size,                    # 默认坐标轴标签大小
    "axes.titlesize": size,                    # 默认坐标轴标题大小
    "xtick.labelsize": size,                   # 默认X轴刻度标签大小
    "ytick.labelsize":size,                   # 默认Y轴刻度标签大小
    # "axes.linewidth": 2,
    # "legend.fontsize": 14,                   # 默认图例字体大小
    # "xtick.major.width": 3,
    # "ytick.major.width": 3,
    # "grid.linestyle": '--',                  # 默认网格线样式
    # "grid.linewidth": 0.5,                   # 默认网格线宽度
    # "grid.alpha": 0.5,                       # 默认网格线透明度

    "savefig.dpi": 1000,
}

rcParams.update(config)
# 准备数据
N=200
Delta=1.0
delta=0.7
m=0.5
ALL=[]
r=cmath.sqrt(abs((Delta+delta+1j*m)/(Delta-delta-1j*m)))
mc=m*(r**2+1)/2/r
print(mc)
n_samples=300

for gamma in np.linspace(0,1.2,n_samples):
    tt=[]
    for k in np.linspace(0,2*np.pi,N):
        beta=r*np.exp(1j*k)
        HH=np.zeros((2,2),dtype=complex)
        hx=Delta*(beta+1/beta)/2-1j*delta*(-1j/2*(beta-1/beta))
        hy=(gamma+m*(-1j/2*(beta-1/beta)))*1j
        HH[0,0]=hx
        HH[1, 1] = hx
        HH[0,1]=hy*(-1j)
        HH[1,0]=hy*(1j)
        EB1=gamma+0.5*(Delta-delta-1j*m)*beta+0.5*(Delta+delta+1j*m)*(1/beta)
        EB2=-gamma+0.5*(Delta-delta+1j*m)*beta+0.5*(Delta+delta-1j*m)*(1/beta)
        tt.append(hx/(EB1-EB2))
        tt.append(hy/(EB1-EB2))
    ALL.append(tt)
data=np.array(ALL)
distance_matrix=np.zeros((n_samples,n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        # 1范数计算
        distance_matrix[i][j] = np.linalg.norm(data[i]-data[j], ord=1)
        # 2范数计算
        # distance_matrix[i][j] = np.linalg.norm(data[i]-data[j], ord=2)
        # 无穷范数计算
        # distance_matrix[i][j] = np.linalg.norm(data[i] - data[j], ord=np.inf)
sigma = 0.01# 核参数
affinity_matrix = np.exp(-(distance_matrix ** 2 )/ (2*sigma*N*N))
# degree = 2  # 多项式核的度
# c = 1  # 偏置
# affinity_matrix = (distance_matrix + c) ** degree
# sigma = 0.01  # 核参数
# affinity_matrix = np.exp(-distance_matrix / sigma)
plt.figure(figsize=(4,3))
plt.tight_layout()
plt.imshow(affinity_matrix,cmap='jet', interpolation='None')
plt.colorbar()
# plt.savefig()
plt.xlabel('$\gamma^{(i)}$',fontsize=18,labelpad=0)
plt.ylabel('$\gamma^{(j)}$',fontsize=18,labelpad=0)
plt.xlim(0, 300)
plt.ylim(300, 0)
plt.xticks([0,100,200,300],labels=['0','0.4','0.8','1.2'])
plt.yticks([300,200,100,0],labels=['1.2','0.8','0.4','0'])
plt.savefig("non-bloch.svg",dpi=300,bbox_inches='tight',transparent=True,pad_inches=0)
plt.show()
row_sum = np.sum(affinity_matrix, axis=1)
print(affinity_matrix.shape)
print(np.diag(row_sum).shape)
markov_matrix = affinity_matrix / row_sum[:, np.newaxis]

eigenvalues, eigenvectors = np.linalg.eig(markov_matrix)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]

sorted_eigenvalues = eigenvalues[sorted_indices].real
sorted_eigenvectors = eigenvectors[:, sorted_indices].real

k = 4
selected_eigenvalues1 = sorted_eigenvalues[1:k]
selected_eigenvectors1 = sorted_eigenvectors[:, 1:k]
a0=(sorted_eigenvalues[0]**1)*sorted_eigenvectors[:, 0][:, np.newaxis]
a1=(sorted_eigenvalues[1]**1)*sorted_eigenvectors[:, 1][:, np.newaxis]
a2=(sorted_eigenvalues[2]**1)*sorted_eigenvectors[:, 2][:, np.newaxis]
a3=(sorted_eigenvalues[3]**1)*sorted_eigenvectors[:, 3][:, np.newaxis]

diffusion_map=np.concatenate((a0,a1),axis=1)


n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters)
cluster_labels = kmeans.fit_predict(diffusion_map)
# print(cluster_labels)

plt.figure(figsize=(4,3))
# plt.rc('text', usetex=True)
plt.scatter(diffusion_map[:,0],diffusion_map[:,1],c=cluster_labels,cmap="jet",s=15)
plt.xticks([])
plt.yticks([])
plt.xlabel('$\psi_{0}$',labelpad=5,fontsize=20, fontweight='bold')
plt.ylabel('$\psi_{1}$',labelpad=5,fontsize=20, fontweight='bold')
# plt.colorbar()
plt.savefig('k_means.svg',dpi=300,bbox_inches='tight',transparent=True)
plt.show()
plt.close()
# plt.text(0.1, 0.9,'$v=1$', transform=inset_axes.transAxes,
#              va='top', ha='left',fontsize=12)
# plt.text(0.75, 0.25,'$v=0$', transform=inset_axes.transAxes,
#              va='top', ha='left',fontsize=12)

plt.figure(figsize=(4,3))
plt.tight_layout()
plt.axvspan(-0.15, 1+0.15,0.9,1.1, color='lightgrey', alpha=0.5)
plt.scatter(range(8),sorted_eigenvalues[0:8],s=18,c="b")
# plt.colorbar()
# plt.savefig()
plt.xlabel('$n$',fontsize=24,labelpad=-6)
plt.ylabel('$\lambda_n$',fontsize=24,labelpad=4)
plt.savefig("eigenvalue.svg",dpi=300,bbox_inches='tight',transparent=True)
plt.show()

N=200
Delta=1.0
delta=0.7
m=0.5
ALL=[]
r=cmath.sqrt(abs((Delta+delta+1j*m)/(Delta-delta-1j*m)))
mc=m*(r**2+1)/2/r
print(mc)
n_samples=300

m1=0.3
gamma1=0.3
EB11_list = []
EB21_list = []
z=np.linspace(0, 2 * np.pi, N)
Z=np.hstack((z, z))
for k in np.linspace(0, 2 * np.pi, N):
    beta = r * np.exp(1j * k)
    hx = Delta * (beta + 1 / beta) / 2 - 1j * delta * (-1j / 2 * (beta - 1 / beta))
    hy = (gamma1 + m * (-1j / 2 * (beta - 1 / beta))) * 1j
    EB1 = gamma1 + 0.5 * (Delta - delta - 1j * m) * beta + 0.5 * (Delta + delta + 1j * m) * (1 / beta)
    EB2 = -gamma1 + 0.5 * (Delta - delta + 1j * m) * beta + 0.5 * (Delta + delta - 1j * m) * (1 / beta)
    EB11_list.append(EB1)
    EB21_list.append(EB2)
e1=EB11_list+EB21_list

gamma1=0.58
EB12_list = []
EB22_list = []
for k in np.linspace(0, 2 * np.pi, N):
    beta = r * np.exp(1j * k)
    hx = Delta * (beta + 1 / beta) / 2 - 1j * delta * (-1j / 2 * (beta - 1 / beta))
    hy = (gamma1 + m * (-1j / 2 * (beta - 1 / beta))) * 1j
    EB1 = gamma1 + 0.5 * (Delta - delta - 1j * m) * beta + 0.5 * (Delta + delta + 1j * m) * (1 / beta)
    EB2 = -gamma1 + 0.5 * (Delta - delta + 1j * m) * beta + 0.5 * (Delta + delta - 1j * m) * (1 / beta)
    EB12_list.append(EB1)
    EB22_list.append(EB2)
e2=EB12_list+EB22_list

gamma1=0.8
EB13_list = []
EB23_list = []
for k in np.linspace(0, 2 * np.pi, N):
    beta = r * np.exp(1j * k)
    hx = Delta * (beta + 1 / beta) / 2 - 1j * delta * (-1j / 2 * (beta - 1 / beta))
    hy = (gamma1 + m * (-1j / 2 * (beta - 1 / beta))) * 1j
    EB1 = gamma1 + 0.5 * (Delta - delta - 1j * m) * beta + 0.5 * (Delta + delta + 1j * m) * (1 / beta)
    EB2 = -gamma1 + 0.5 * (Delta - delta + 1j * m) * beta + 0.5 * (Delta + delta - 1j * m) * (1 / beta)
    EB13_list.append(EB1)
    EB23_list.append(EB2)
e3=EB13_list+EB23_list

elev=15
azim=80
fig = plt.figure(figsize=(6,4.5))
ax4 = fig.add_subplot(111, projection='3d')
colors = ['red'] * N + ['blue'] * N
ax4.scatter(np.array(e1).real, np.array(e1).imag, Z,color=colors,s=15)
ax4.scatter(np.array(EB11_list).real, np.array(EB11_list).imag,zs=0, zdir='z', linewidth=0.5, color='red',s=5)
# ax4.scatter(np.array(EB21_list).real, np.array(EB21_list).imag, z, color='blue',s=5,alpha=0.5)
ax4.scatter(np.array(EB21_list).real, np.array(EB21_list).imag, zs=0, zdir='z', linewidth=0.5, color='blue',s=5)
# ax4.scatter(0, 0, 0, color='black', s=25)
ax4.view_init(elev=elev, azim=azim)
ax4.set_box_aspect([1, 1, 1.5])
ax4.set_zticks([0, 2 * np.pi / 2, 2 * np.pi], labels=['0', '$\pi$', '$2\pi$'],fontsize=22)
ax4.tick_params(axis='z', pad=5)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_xlabel('Re$[E]$',labelpad=-10)
ax4.set_ylabel('Im$[E]$',labelpad=-10)
ax4.set_zlabel('$k$',fontsize=22,labelpad=10)
# ax4.text2D(0.1, 0.98, '(d)',va='top', ha='left', transform=ax4.transAxes,fontsize=18)
ax4.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax4.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax4.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.savefig("band1.svg",dpi=300,bbox_inches='tight',transparent=True)
plt.show()

fig = plt.figure(figsize=(6,4.5))
ax6 = fig.add_subplot(111, projection='3d')
ax6.scatter(np.array(e3).real, np.array(e3).imag, Z,color=colors,s=15)
ax6.scatter(np.array(EB13_list).real, np.array(EB13_list).imag,zs=0, zdir='z', linewidth=0.5, color='red',s=5)
# ax6.scatter(np.array(EB23_list).real, np.array(EB23_list).imag, z, color='blue',s=5)
ax6.scatter(np.array(EB23_list).real, np.array(EB23_list).imag, zs=0, zdir='z', linewidth=0.5, color='blue',s=5)

ax6.view_init(elev=elev, azim=azim)
ax6.set_box_aspect([1, 1, 1.5])
ax6.set_zticks([0, 2 * np.pi / 2, 2 * np.pi], labels=['0', '$\pi$', '$2\pi$'],fontsize=22)
ax6.tick_params(axis='z', pad=5)
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_xlabel('Re$[E]$',labelpad=-10)
ax6.set_ylabel('Im$[E]$',labelpad=-10)
ax6.set_zlabel('$k$',fontsize=22,labelpad=10)
# ax6.text2D(0.1, 0.98, '(f)',va='top', ha='left', transform=ax6.transAxes,fontsize=18)
ax6.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax6.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax6.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.savefig("band2.svg",dpi=300,bbox_inches='tight',transparent=True)
plt.show()

fig = plt.figure(figsize=(6,4.5))
ax6 = fig.add_subplot(111, projection='3d')
ax6.scatter(np.array(e2).real, np.array(e2).imag, Z,color=colors,s=15)
ax6.scatter(np.array(EB12_list).real, np.array(EB12_list).imag,zs=0, zdir='z', linewidth=0.5, color='red',s=5)
# ax6.scatter(np.array(EB23_list).real, np.array(EB23_list).imag, z, color='blue',s=5)
ax6.scatter(np.array(EB22_list).real, np.array(EB22_list).imag, zs=0, zdir='z', linewidth=0.5, color='blue',s=5)

ax6.view_init(elev=elev, azim=azim)
ax6.set_box_aspect([1, 1, 1.5])
ax6.set_zticks([0, 2 * np.pi / 2, 2 * np.pi], labels=['0', '$\pi$', '$2\pi$'],fontsize=22)
ax6.tick_params(axis='z', pad=5)
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_xlabel('Re$[E]$',labelpad=-10)
ax6.set_ylabel('Im$[E]$',labelpad=-10)
ax6.set_zlabel('$k$',fontsize=22,labelpad=10)
# ax6.text2D(0.1, 0.98, '(f)',va='top', ha='left', transform=ax6.transAxes,fontsize=18)
ax6.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax6.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax6.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.savefig("band3.svg",dpi=300,bbox_inches='tight',transparent=True)
plt.show()