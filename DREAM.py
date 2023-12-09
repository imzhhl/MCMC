# -*- coding: utf-8 -*-
"""
author： Hongliang Zhang - WHU
date：   2022-09-26
log: 1. 2022-10-05 伴随方法用于室内污染物寻源
     ______  _    _   _    _   _      
    |___  / | |  | | | |  | | | |     
       / /  | |__| | | |__| | | |     
      / /   |  __  | |  __  | | |     
     / /__  | |  | | | |  | | | |____ 
    /_____| |_|  |_| |_|  |_| |______|

"""
#%% 导入库
import ansys.fluent.core as pyfluent
import numpy as np
import random
import time
import re
import os
from decimal import Decimal
from tqdm import tqdm
import arviz as az
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.stats import norm
import sys
from scipy.integrate import solve_bvp
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import csv
from scipy.interpolate import griddata 
import scipy.stats as stats 
import pymc as pm
from matplotlib.pyplot import MultipleLocator
from matplotlib.font_manager import FontProperties


matplotlib.rc("font", family='Microsoft YaHei')

#%% 一些需要的函数定义

# 用于计算似然函数, c_predict和c_true为存储预测点值和检测点值的数组
# 用了log，将相乘变成相加
def likelihood_func_log(sigma, c_predict, c_true):
     #c_predict = data_predict_new
     #sigma = sigma
     #c_true = point_true_value
    return -len(c_true)*np.log(sigma * np.sqrt(2* np.pi)) - np.sum(((c_true-c_predict)**2) / (2*sigma**2))

# laplace型似然函数
def laplace_likelihood_func_log(sigma, c_predict, c_true):
     #c_predict = data_predict_new
     #sigma = sigma
     #c_true = point_true_value
    return np.sum(-np.log(2*sigma)-(np.sqrt((c_true-c_predict)**2)) / (sigma))

# 从fluent中导出的数据导入到numpy数组中
def adjoint_file_to_array(sensor_list):

    # 保存x坐标, 和y坐标的列表
    x=[]; y=[]

    # 创建N个二维空数组
    uds = [[] for i in range(len(sensor_list))]
    
    # 创建一维列表，用于存储二维np数组，进而转化为三维np数组
    uds_t = []

    # 读取csv文件，当然也可以读取类似txt之类的文件
    with open(r'F:\ZHHL\TE_Doctor\研究内容\SCI论文\5-动态风场下利用概率伴随方法进行室外污染物的溯源\MCMC采样\uds_adjoint','r')  as  csvfile:   
        #指定分隔符为","，因为我们刚才导出时就是逗号
        plots=csv.reader(csvfile,delimiter=',')
        #循环读取文件各列
        for row in plots:
            #为了跳过文件前面的非数据行  
            if plots.line_num == 1:
                continue
            # 读取x和y坐标
            x.append(float(row[1]))
            y.append(float(row[2]))
            # 读取uds_1-uds_19的场值到二维数组uds中
            for i in range(len(sensor_list)):
                uds[i].append(float(row[i+3]))
                
    # 将一维列表(元素为一维数组)转化为二维数组             
    uds = np.array(uds)

    # 确定meshgrid的坐标范围，并离散坐标
    xi=np.linspace(min(x),max(x), 5000)
    yi=np.linspace(min(y),max(y), 5000)
     
    # grid_x,grid_y坐标必须维数一致，且为二维
    grid_x, grid_y = np.meshgrid(xi, yi)
     
    # 对x，y位置的浓度值进行插值，插值的方法有'cubic', 'linear', 'nearest'
    # 注意传入的坐标参数需要以元组的形式成对传入
    # 当然matplotlib也自带griddata插值函数，该函数每个坐标是一个参数
    # 但matplotlib自带的griddata插值函数只能使用默认的linear插值

    # griddata插值，并存入uds_t列表中，列表中的元素为二维数组
    for i in range(len(sensor_list)):
        uds_t.append(griddata((x,y), uds[i], (grid_x,grid_y), method='nearest', fill_value = np.nan, rescale = False))

    # 将uds_t转换为三维np数组
    uds_t = np.array(uds_t)
    
    # 保存到文件中，下次使用是直接读取文件，避免重复上述过程(太慢！)
    np.save('uds_adjoint.npy', uds_t)

def direct_file_to_array(source_list):
    # 保存x坐标, 和y坐标的列表
    x=[]; y=[]

    # 创建N个二维空数组
    uds = [[] for i in range(len(source_list))]
    
    # 创建一维列表，用于存储二维np数组，进而转化为三维np数组
    uds_t = []

    # 读取csv文件，当然也可以读取类似txt之类的文件
    with open(r'F:\ZHHL\TE_Doctor\研究内容\SCI论文\5-动态风场下利用概率伴随方法进行室外污染物的溯源\MCMC采样\uds_direct','r')  as  csvfile:   
        #指定分隔符为","，因为我们刚才导出时就是逗号
        plots=csv.reader(csvfile,delimiter=',')
        #循环读取文件各列
        for row in plots:
            #为了跳过文件前面的非数据行  
            if plots.line_num == 1:
                continue
            # 读取x和y坐标
            x.append(float(row[1]))
            y.append(float(row[2]))
            # 读取uds_1-uds_6的场值到二维数组uds中
            for i in range(len(source_list)):
                uds[i].append(float(row[i+3]))
                
    # 将一维列表(元素为一维数组)转化为二维数组             
    uds = np.array(uds)
    

    # 确定meshgrid的坐标范围，并离散坐标
    xi=np.linspace(min(x),max(x), 5000)
    yi=np.linspace(min(y),max(y), 5000)
     
    # grid_x,grid_y坐标必须维数一致，且为二维
    grid_x, grid_y = np.meshgrid(xi, yi)
     
    # 对x，y位置的浓度值进行插值，插值的方法有'cubic', 'linear', 'nearest'
    # 注意传入的坐标参数需要以元组的形式成对传入
    # 当然matplotlib也自带griddata插值函数，该函数每个坐标是一个参数
    # 但matplotlib自带的griddata插值函数只能使用默认的linear插值

    # griddata插值，并存入uds_t列表中，列表中的元素为二维数组
    for i in range(len(source_list)):
        uds_t.append(griddata((x,y), uds[i], (grid_x,grid_y), method='nearest', fill_value = np.nan, rescale = False))

    # 将uds_t转换为三维np数组
    uds_t = np.array(uds_t)
    
    # 保存到文件中，下次使用是直接读取文件，避免重复上述过程(太慢！)
    np.save('uds_direct.npy', uds_t)
    
# 用于计算伴随浓度场在预测点(X_current, Y_current)处的浓度值
def adjoint_results_1(source_list, sensor_list):
    # 保存源处的伴随浓度值
    point_adjoint_value = [] 
    
    for i in range(len(sensor_list)):
        point_adjoint_value.append(uds_adjoint[i][int(5*source_list[0][1])][int(5*source_list[0][0])])
    # 浓度列表返回
    return np.array(point_adjoint_value)

# 用于计算伴随浓度场在预测点(X_current, Y_current)处的浓度值
def adjoint_results(X_current, Y_current, S_current, sensor_list):
    # 用于存储新预测位置下,各监测点处的值(对偶)
    data_predict_new = []
    
    for i in range(len(sensor_list)):
        data_predict_new.append(S_current*uds_adjoint[i][int(5*Y_current)][int(5*X_current)])
    
    # 返回存储各监测点浓度值的列表
    return data_predict_new

# 用于提取uds_7在监测点处的浓度值(真实值)
def direct_results(sensor_list):  
    # 保存监测点处的真实浓度值
    point_true_value = []
    for i in range(len(sensor_list)):
        point_true_value.append(uds_direct[0][int(5*sensor_list[i][1])][int(5*sensor_list[i][0])])
    # 浓度列表返回
    return point_true_value

#%% 生成监测点数据

# 设定真实污染源的位置 
X_true = 651
Y_true = 322
S_true = 3.43

source_list = [[651, 322]]
# 设定监测点的坐标
sensor_list = [[302.95, 656.38],
               [438.89, 719.76],
               [346.74, 621.62],
               [392.05, 642.75],
               [437.37, 663.88],
               [390.52, 586.84],
               [435.84, 608.00],
               [481.15, 629.13],
               [434.31, 552.12],
               [479.63, 573.25],
               [478.10, 517.37],
               [523.42, 538.50],
               [521.89, 482.62],
               [520.36, 426.74],
               [565.68, 447.87],
               [564.15, 391.99],
               [609.47, 413.12],
               [607.94, 357.24]]


# adjoint_file_to_array(sensor_list)
# direct_file_to_array(source_list)


# 从文件中导入uds_adjoint数组，并反转y轴
uds_adjoint = []; uds_adjoint = np.load('uds_adjoint.npy', uds_adjoint); #uds_adjoint = np.flip(uds_adjoint, 1)
# 从文件中导入uds_direct数组，并反转y轴
uds_direct = []; uds_direct = np.load('uds_direct.npy', uds_direct); #uds_direct = np.flip(uds_direct, 1)

# 交互0和1轴的数据
# uds_adjoint[[0,1],:,:] = uds_adjoint[[1,0],:,:]

# 真实污染源时，在监测点处污染物的浓度(uds_0在各监测点处的浓度值)
point_true_value = S_true * np.array(direct_results(sensor_list))
point_adjoint_value = S_true * adjoint_results_1(source_list, sensor_list)

# 打印真实污染源时监测点处污染物的浓度
for i in range(len(point_true_value)):
    print(f"point_{i+1}_true_value = {point_true_value[i]}") 
    
for i in range(len(point_adjoint_value)):
    print(f"point_{i+1}_adjoint_value = {point_adjoint_value[i]}") 

adjoint_results(651, 322, 1, sensor_list)
# %%关键参数设定及变量声明   
diff_sigma_Euler = []
diff_sigma_Intensity = []
my_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
for diff_sigma in my_list:
    
    Euler = []
    Intensity = []
    
    # 重复100遍
    for m in range (100):
        n_samples = 50000   # MCMC采样次数
        n_chains = 5        # Markov 链的数量
        n_dims = 3          # 参数的维度
        sigma = diff_sigma  # 目标分布中似然函数的标准差
        
        X_init = 566    # 用于设定X迭代初始值
        Y_init = 390    # 用于设定Y迭代初始值
        S_init = 3.43      # 用于设定Y迭代初始值
        
        samples = np.random.rand(n_chains, n_samples, n_dims)  # 每条链的初始点
        samples[: , 0, 0] = X_init         # 每条链的初始点
        samples[: , 0, 1] = Y_init         # 每条链的初始点
        samples[: , 0, 2] = S_init         # 每条链的初始点
        
        trunc_X = [0, 1000]  # 设置建议分布截断范围
        trunc_Y = [0, 1000]  # 设置建议分布截断范围
        trunc_S = [0, 5]     # 设置建议分布截断范围
        
        gamma = 2.38 / np.sqrt(2 * n_dims)  # 更新规则中的因子
        b_star = 1e-12
        b = 0.1
        epsilon = 1e-6  # 保证生成的提议点与当前点不同的小常数
        
        count = 0       # 用于计算接受率标识
        flag = True     # 用于记录是否接受了新值
        iteration = np.zeros(n_samples) # 用于计算逐时的接受率
        accepted = []  # 记录接受的采样
        rejected = []  # 记录拒绝的采样
        
        # ################################ DREAM 迭代开始 ##############################
        for i in tqdm(range(1, n_samples)):
            for j in range(n_chains):
                # 从剩下的链中选择两条
                r1, r2 = np.random.choice([k for k in range(n_chains) if k != j], size=2, replace=False)
                
                # 计算差分向量
                diff = samples[r1, i - 1, :] - samples[r2, i - 1, :]
        
                # 生成新的提议点
                proposed_p = samples[j, i - 1, :] + (np.ones(n_dims) + np.random.uniform(-b, b, n_dims)) * gamma * diff + np.random.normal(0, b_star, n_dims)
                # proposed_p = samples[j, i - 1, :] + gamma * diff + np.random.normal(0, b_star, n_dims)
                X_new, Y_new, S_new = proposed_p[0], proposed_p[1], proposed_p[2]
                
                # X坐标超边界罚回
                while X_new < trunc_X[0] or X_new > trunc_X[1]:
                    X_new = 566
                # Y坐标超边界罚回
                while Y_new < trunc_Y[0] or Y_new > trunc_Y[1]:
                    Y_new = 390
                # S坐标超边界罚回
                while S_new < trunc_S[0] or S_new > trunc_S[1]:
                    S_new = 2.5
              
                proposed_p[0], proposed_p[1], proposed_p[2] = X_new, Y_new, S_new
                X_old, Y_old, S_old = samples[j, i - 1, :]   
        
                # 计算新的提议点的,监测点处污染物浓度值
                data_predict_old = adjoint_results(X_old, Y_old, S_old, sensor_list)
                data_predict_new = adjoint_results(X_new, Y_new, S_new, sensor_list)
                 
                # 计算接受概率
                likelihood_i = likelihood_func_log(sigma, data_predict_old, point_true_value)
                likelihood_j = likelihood_func_log(sigma, data_predict_new, point_true_value)      
                alpha = min(1, np.exp(likelihood_j - likelihood_i))
        
                # 以 alpha 的概率接受提议点
                if np.random.rand() < alpha: # 接受该点
                    samples[j, i, :] = proposed_p
                    flag  = True
                    iteration[i] = 1.0
                    count += 1
                    accepted.append([X_new, Y_new, S_new])
                
                else:                        # 拒绝该点
                    samples[j, i, :] = samples[j, i - 1, :]
                    flag  = False
                    rejected.append([X_new, Y_new, S_new])
        
        ################################### 迭代结束 ###################################   
        
        accepted = np.array(accepted); rejected = np.array(rejected)
        
        kde_X = gaussian_kde(samples[0, -5000:, 0])
        # 生成一组用于评估KDE的值
        xx = np.linspace(samples[0, -5000:, 0].min(), samples[0, -5000:, 0].max(), 1000)
        # 评估kde_X在x上的概率密度值
        kde_X_values = kde_X.evaluate(xx)
        # 找到kde_X的极小值和极大值
        max_kde_X = xx[np.argmax(kde_X_values)]
        
        kde_Y = gaussian_kde(samples[0, -5000:, 1])
        # 生成一组用于评估KDE的值
        yy = np.linspace(samples[0, -5000:, 1].min(), samples[0, -5000:, 1].max(), 1000)
        # 评估kde_Y在x上的概率密度值
        kde_Y_values = kde_Y.evaluate(yy)
        # 找到kde_Y的极小值和极大值
        max_kde_Y = yy[np.argmax(kde_Y_values)]
        
        kde_S = gaussian_kde(samples[0, -5000:, 2])
        # 生成一组用于评估KDE的值
        ss = np.linspace(samples[0, -5000:, 2].min(), samples[0, -5000:, 2].max(), 1000)
        # 评估kde_Y在x上的概率密度值
        kde_S_values = kde_S.evaluate(ss)
        # 找到kde_Y的极小值和极大值
        max_kde_S = ss[np.argmax(kde_S_values)]
        
        Euler_distance = np.sqrt((max_kde_X - X_true)**2 + (max_kde_Y - Y_true)**2)
        Euler.append(Euler_distance)
        Intensity.append(abs(max_kde_S-S_true))
        
        print(f"current diff_sigma = {diff_sigma}, current m = {m}\n")
    diff_sigma_Euler.append(np.mean(Euler))
    diff_sigma_Intensity.append(np.mean(Intensity))
            
        # print(f"x均值={np.mean(samples[0, -5000:, 0])}") # 均值
        # print(f"x标准差= {np.std(samples[0, -5000:, 0],ddof=1)}\n" ) # 标准差
        
        # print(f"y均值={np.mean(samples[0, -5000:, 1])}") # 均值
        # print(f"y标准差= {np.std(samples[0, -5000:, 1],ddof=1)}\n" ) # 标准差
        
        # print(f"s均值={np.mean(samples[0, -5000:, 2])}") # 均值
        # print(f"s标准差= {np.std(samples[0, -5000:, 2],ddof=1)}\n" ) # 标准差
        # print(f"最终接受率 = {round((count/(n_samples*n_chains))*100, 4)}%")
        
        # acctep_ration = []
        # for i in range(T):  
        #     acctep_ration.append(sum(iteration[:i])/(i+1))
            
        # # 接受率历史变化绘图
        # plt.plot(list(range(0, T)), acctep_ration)

# %% 统一绘图

f, ax = plt.subplots(2, 3, figsize = (16, 8))

# 设置字体
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

#直方图统计x
plt.subplot(2,3,1)
sns.histplot(samples[0, -5000:, 0],  bins=50, kde=True, color='red')
# plt.hist(X[-20000:], bins=100 , density=1, facecolor='red')
plt.title('X-plot', font)
plt.xlabel('X(m)',font)
plt.ylabel('KDE',font)
plt.axvline(X_true, color='k')
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

# 直方图统计y
plt.subplot(2,3,2)
# plt.hist(Y[-20000:], bins=100, density=1, facecolor='green')
sns.histplot(samples[0, -5000:, 1], bins=50, kde=True, color='green')
plt.title('Y-plot',font)
plt.xlabel('Y(m)',font)
plt.ylabel('KDE',font)
plt.axvline(Y_true, color='k')
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')


# 直方图统计s
plt.subplot(2,3,3)
# plt.hist(Y[-20000:], bins=100, density=1, facecolor='green')
sns.histplot(samples[0, -5000:, 2], bins=50, kde=True, color='blue')
plt.title('S-plot',font)
plt.xlabel('S(m)',font)
plt.ylabel('KDE',font)
plt.axvline(S_true, color='k')
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

#采样值变化x
plt.subplot(2,3,4)
plt.plot(list(range(n_samples-5000, n_samples)), samples[0, -5000:, 0])
plt.title('X-plot',font)
plt.xlabel('iteration',font)
plt.ylabel('X(m)',font)
plt.axhline(X_true, color='k')
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

#采样值变化y
plt.subplot(2,3,5)
plt.plot(list(range(n_samples-5000, n_samples)), samples[0, -5000:, 1])
plt.title('Y-plot',font)
plt.xlabel('iteration',font)
plt.ylabel('Y(m)',font)
plt.axhline(Y_true, color='k')
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

#采样值变化s
plt.subplot(2,3,6)
plt.plot(list(range(n_samples-5000, n_samples)), samples[0, -5000:, 2])
plt.title('s-plot',font)
plt.xlabel('iteration',font)
plt.ylabel('S(m)',font)
plt.axhline(S_true, color='k')
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

plt.tight_layout()
plt.show()

# %% x和y的采样结果同时显示

fig = plt.figure(figsize=(10,10))

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 24,
}

ax = fig.add_subplot(1,1,1)
# ax.plot(accepted[:,0], accepted[:,1], label="Path")
ax.plot(rejected[-5000:,0], rejected[-5000:,1], 'rx', label='Rejected',alpha=0.5)
ax.plot(accepted[-5000:,0], accepted[-5000:,1], 'b.', label='Accepted',alpha=0.5)
plt.xlabel("$X$(m)",font,labelpad=(0))
plt.ylabel("$Y$(m)",font,labelpad=(0))
x_major_locator=MultipleLocator(200)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(200)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数

ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细
ax.xaxis.set_major_formatter('{x:.00f}')
ax.yaxis.set_major_formatter('{x:.00f}')

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.xlim(0,1000)
plt.ylim(0,1000)

# plt.annotate('$S$', xy=(651,322), weight='heavy', bbox=dict(boxstyle='circle,pad=0.001', fc='yellow', ec='k', lw=1, alpha=1))

plt.tick_params(labelsize=24,which='major',width=2,colors='k')
plt.tick_params(labelsize=24,which='minor',width=2,colors='k')
plt.legend(loc=3,frameon=True, framealpha=1, edgecolor = 'k', prop = {"family": "Times New Roman", "size":24})
plt.tight_layout()
# ax.set_title("MCMC sampling for $x$ and $y$ with Metropolis-Hastings. All samples are shown.") 

# %% X采样分布绘图

f, ax = plt.subplots(3, 1, figsize = (10, 12))
# 设置字体
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

#直方图统计X
plt.subplot(3,1,1)
sns.distplot(samples[0, -20000:, 0], hist=True, bins=60, kde=True, fit=None, norm_hist=False,color="b",hist_kws={'alpha':0.6,'color':'gray'},kde_kws={'alpha':1,'color':'blue','linewidth':1.5})
# plt.hist(X_history[-20000:], bins=100)
# sns.kdeplot(X_history[-20000:],shade=True,color="g")
# plt.hist(X[-20000:], bins=100 , density=1, facecolor='red')

plt.xlabel('$X$(m)',font,labelpad=(5))
plt.ylabel('$P$',font,labelpad=(5))



plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

x_major_locator=MultipleLocator(100)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.01)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数

ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.02f}')

plt.xlim(0,1000)
plt.ylim(0,0.05)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.axvline(X_true, color='red',linestyle='--',linewidth=1.5)
plt.grid(b=1, which='both',color='k',linestyle='--', linewidth=0.5,alpha=0.6)
plt.legend([r'$KDE$', r'$x$ location'], loc=2,frameon=True, framealpha=1, edgecolor = 'k', prop = {"family": "Times New Roman", "size":18})
plt.tight_layout()

#  Y采样分布绘图


#直方图统计Y
plt.subplot(3,1,2)
sns.distplot(samples[0, -20000:, 1], hist=True, bins=60, kde=True, fit=None, norm_hist=False,color="b",hist_kws={'alpha':0.6,'color':'gray'},kde_kws={'alpha':1,'color':'blue','linewidth':1.5})
# plt.hist(X_history[-20000:], bins=100)
# sns.kdeplot(X_history[-20000:],shade=True,color="g")
# plt.hist(X[-20000:], bins=100 , density=1, facecolor='red')
plt.xlabel('$Y$(m)',font,labelpad=(5))
plt.ylabel('$P$',font,labelpad=(5))
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

x_major_locator=MultipleLocator(100)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.01)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数

ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.02f}')

plt.xlim(0,1000)
plt.ylim(0,0.05)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.axvline(Y_true, color='red',linestyle='--',linewidth=1.5)
plt.grid(b=1, which='both',color='k',linestyle='--', linewidth=0.5,alpha=0.6)
plt.legend([r'$KDE$', r'$y$ location'], loc=2,frameon=True, framealpha=1, edgecolor = 'k', prop = {"family": "Times New Roman", "size":18})
plt.tight_layout()

#  S采样分布绘图



#直方图统计S
plt.subplot(3,1,3)
sns.distplot(samples[0, -20000:, 2], hist=True, bins=60, kde=True, fit=None, norm_hist=False,color="b",hist_kws={'alpha':0.6,'color':'gray'},kde_kws={'alpha':1,'color':'blue','linewidth':1.5})
# plt.hist(X_history[-20000:], bins=100)
# sns.kdeplot(X_history[-20000:],shade=True,color="g")
# plt.hist(X[-20000:], bins=100 , density=1, facecolor='red')
plt.xlabel('$S$ ($\mathregular{unit/m^{3}·kg}$)',font,labelpad=(5))
plt.ylabel('$P$',font,labelpad=(5))
plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')

x_major_locator=MultipleLocator(0.5)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(1)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数

ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.01f}')

plt.xlim(0,5)
plt.ylim(0,5)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.axvline(S_true, color='red',linestyle='--',linewidth=1.5)
plt.grid(b=1, which='both',color='k',linestyle='--', linewidth=0.5,alpha=0.6)
plt.legend([r'$KDE$', r'$s$ strength'], loc=2, frameon=True, framealpha=1, edgecolor = 'k', prop = {"family": "Times New Roman", "size":18})
plt.tight_layout()

# %% 显示采样历史

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(2,1,1)
to_show=-int(0.5*accepted.shape[0])
ax.plot( rejected[to_show:, 0], 'rx', label='Rejected',alpha=0.5)
ax.plot( accepted[to_show:, 0], 'b.', label='Accepted',alpha=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("$x$")
ax.set_title("MCMC sampling for $x$ with Metropolis-Hastings. Half samples are shown.")
ax.grid()
ax.legend()

ax2 = fig.add_subplot(2,1,2)
to_show=-int(0.5*accepted.shape[0])
ax2.plot( rejected[to_show:, 1], 'rx', label='Rejected',alpha=0.5)
ax2.plot( accepted[to_show:, 1], 'b.', label='Accepted',alpha=0.5)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("$y$")
ax2.set_title("MCMC sampling for $y$ with Metropolis-Hastings. Half samples are shown.")
ax2.grid()
ax2.legend()

fig.tight_layout()
accepted.shape


# %% 显示自相关性

show=-int(0.5*accepted.shape[0])
mean_acc_0=accepted[show:,0].mean()
mean_acc_1=accepted[show:,1].mean()
mean_acc_2=accepted[show:,2].mean()
lag=np.arange(1,500)

def autocorr(accepted,lag):
    num_0=0
    denom_0=0
    num_1=0
    denom_1=0
    num_2=0
    denom_2=0
    for i in range(accepted.shape[0]-lag):
        num_0+=(accepted[i,0]-mean_acc_0)*(accepted[i+lag,0]-mean_acc_0)
        num_1+=(accepted[i,1]-mean_acc_1)*(accepted[i+lag,1]-mean_acc_1)
        num_2+=(accepted[i,2]-mean_acc_2)*(accepted[i+lag,2]-mean_acc_2)
        denom_0+=(mean_acc_0-accepted[i,0])**2
        denom_1+=(mean_acc_1-accepted[i,1])**2
        denom_2+=(mean_acc_2-accepted[i,2])**2
    rk_0=num_0/denom_0
    rk_1=num_1/denom_1
    rk_2=num_2/denom_2  
    return rk_0, rk_1, rk_2


accepted_reversed=accepted[show:,:]
result=np.zeros((3,lag.shape[0]))
#print(lag)
for l in lag:
    result[:,l-1]=autocorr(accepted_reversed,l)
    
    
###Instead of writing an autocorrelation function, one could simply use thee autocorr function provided in pymc3    
#from pymc3.stats import autocorr
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1,1,1)
#ax.plot(lag, [autocorr(accepted[show:,1], l) for l in lags], label='auto b')
#ax.plot(lag, [autocorr(accepted[show:,0], l) for l in lags], label='auto a')
ax.plot(lag, result[1,:], linestyle='--',color='r', linewidth=2, label='Auto correlation for $X$')
ax.plot(lag, result[0,:], linestyle='-', color='b',linewidth=2, label='Auto correlation for $Y$')
ax.plot(lag, result[2,:], linestyle='-.',color='k', linewidth=2, label='Auto correlation for $S$')
ax.legend(loc=0)

plt.xlabel("Iteration",font,labelpad=(10))
plt.ylabel("Autocorrelation",font,labelpad=(10))
x_major_locator=MultipleLocator(100)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.2)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数

ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
ax.xaxis.set_major_formatter('{x:.01f}')
ax.yaxis.set_major_formatter('{x:.01f}')

plt.xlim(0,500)
plt.ylim(-0.1,1)

plt.grid(b=1, which='both',color='k',linestyle='--', linewidth=0.5,alpha=0.6)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.tick_params(labelsize=18,which='major',width=2,colors='k')
plt.tick_params(labelsize=18,which='minor',width=2,colors='k')
plt.legend(loc=1,frameon=True, framealpha=1, edgecolor = 'k', prop = {"family": "Times New Roman", "size":18})
plt.tight_layout()

# %% 二维联合分布图

show=int(-0.5*accepted.shape[0])
hist_show=int(-0.50*accepted.shape[0])
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1,1,1)
xbins, ybins = np.linspace(0,9,300), np.linspace(0,3,300)
counts, xedges, yedges, im = ax.hist2d(accepted[hist_show:,0], accepted[hist_show:,1], density=True, bins=[xbins, ybins])
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(im, ax=ax)
ax.set_title("2D histogram showing the joint distribution of $x$ and $y$")

# %% pymc进行高斯推断

# 显示核密度估计
sns.kdeplot(X_history[-20000:])
plt.xlabel('$x$', fontsize=16)
plt.show()

sns.kdeplot(Y_history[-20000:])
plt.xlabel('$x$', fontsize=16)
plt.show()

# 利用pymc进行x坐标推断
with pm.Model() as model_g_X:
    mu = pm.Uniform('mu', lower=0, upper=9)
    sigma = pm.HalfNormal('sigma', sigma=1)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=X_history[-20000:])
    trace_g_X = pm.sample(1100)
    
az.plot_trace(trace_g_X, combined=True);
a = az.summary(trace_g_X, round_to=2)

# 利用pymc进行y坐标推断
with pm.Model() as model_g_Y:
    mu = pm.Uniform('mu', lower=0, upper=3)
    sigma = pm.HalfNormal('sigma', sigma=1)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=Y_history[-20000:])
    trace_g_Y = pm.sample(1100)
    
az.plot_trace(trace_g_Y, combined=True);
b = az.summary(trace_g_Y, round_to=2)

# 绘图
x = np.linspace(0, 9, 1000)
y = stats.norm(a.loc["mu","mean"], a.loc["sigma","mean"]).pdf(x)
plt.axvline(X_true, color='b')
plt.plot(x, y)

x = np.linspace(0, 3, 1000)
y = stats.norm(b.loc["mu","mean"], b.loc["sigma","mean"]).pdf(x)
plt.axvline(Y_true, color='b')
plt.plot(x, y)

