import numpy as np
from matplotlib import pyplot as plt
import time
import math

"""----------------------结果统计----------------------"""


def statistic(w, xin, yin, xout, yout):
    wrong_cases_train = 0
    wrong_cases_test = 0
    nin = len(xin)
    nout = len(xout)
    for j in range(nin):
        if np.dot(w, xin[j].T) * yin[j] <= 0:
            wrong_cases_train += 1
    wrong_rate_train = wrong_cases_train / nin

    for j in range(nout):
        if np.dot(w, xout[j].T) * yout[j] <= 0:
            wrong_cases_test += 1
    wrong_rate_test = wrong_cases_test / nout

    print("训练集正确率=", 1 - wrong_rate_train)
    print("测试集正确率=", 1 - wrong_rate_test)

    return 0


"""----------------洗牌----------------"""


def shuffle(xin, yin):
    x_in = np.copy(xin)
    y_in = np.copy(yin)
    for j in range(len(x_in) - 1, - 1, -1):
        p = np.random.randint(0, j + 1)
        x_in[j], x_in[p] = x_in[p], x_in[j]
        y_in[j], y_in[p] = y_in[p], y_in[j]
    return x_in, y_in


"""----------------计算损失函数和梯度----------------"""


def deriv_loss(xin, yin, w):  # x列向量，坐标沿行；w行向量
    nx, d = xin.shape
    loss = 0
    d_loss = np.zeros(d)
    for j in range(nx):
        temp = 1 + math.exp(-yin[j] * np.dot(xin[j], w.T))
        loss = loss + math.log(temp)
        d_loss = d_loss + 1 / temp * (-yin[j] * xin[j])
    loss = loss / nx
    d_loss = d_loss / nx  # d_loss为行向量
    d_loss_abs = np.linalg.norm(d_loss)
    return loss, d_loss, d_loss_abs


"""----------------------Logistic回归----------------------"""


def logistic(xin, yin, eta, batch, epoch):
    x = np.copy(xin)  # 列向量，坐标沿行
    y = np.copy(yin)  # 列向量
    nx, d = x.shape
    w = np.zeros(d)  # 行向量
    hist = []
    loss_t = []
    hist.append([w[0], w[1], w[2]])
    terminate = False
    it = 0
    loss_min = 1000
    w_best = w
    for j in range(epoch):
        if terminate:
            break
        xe, ye = x, y  # 不进行洗牌
        # xe, ye = shuffle(x, y)        # 进行洗牌
        for k in range(int(nx / batch)):
            xb = xe[k * batch:(k + 1) * batch]
            yb = ye[k * batch:(k + 1) * batch]
            loss, d_loss, d_loss_abs = deriv_loss(xb, yb, w)
            loss_t.append([it, loss])
            if d_loss_abs <= 0.000000001:
                terminate = True
                break
            it += 1
            w = w - eta * d_loss
            hist.append([w[0], w[1], w[2]])
            if loss <= min(loss, loss_min):
                loss_min = loss
                w_best = w
    w_best = w_best.T
    return w_best, it, loss_t, hist


"""----------------------数据集初始化----------------------"""

# 数据分布与规模
u1 = [-5, 0]
s1 = [[1, 0], [0, 1]]
u2 = [0, 5]
s2 = [[1, 0], [0, 1]]
n = 200
train_rate = 0.8
n_train = int(n * train_rate)
n_test = n - n_train
# 数据填充
x1 = np.empty([n, 2])  # A
x2 = np.empty([n, 2])  # B
x_train = np.empty([n_train * 2, 2])  # 320
x_test = np.empty([n_test * 2, 2])  # 80

for i in range(n):  # 200
    x1[i] = np.random.multivariate_normal(u1, s1)
    x2[i] = np.random.multivariate_normal(u2, s2)

for i in range(n_train):  # 160
    x_train[i] = x1[i]  # A
    x_train[n_train + i] = x2[i]  # B
for i in range(n_test):  # 40
    x_test[i] = x1[i]  # A
    x_test[n_test + i] = x2[i]  # B

aug1 = np.ones((n_train * 2, 1))
x_train = np.hstack((x_train, aug1))
aug2 = np.ones((n_test * 2, 1))
x_test = np.hstack((x_test, aug2))

y_train = np.empty([n_train * 2, 1])
for i in range(n_train):
    y_train[i] = 1
    y_train[n_train + i] = -1
y_test = np.empty([n_test * 2, 1])
for i in range(n_test):
    y_test[i] = 1
    y_test[40 + i] = -1

"""----------------------代码运行----------------------"""

time_start = time.time()
eta = 0.001
batch = 320
epoch = 1000
w, iteration, loss_f, hist = logistic(x_train, y_train, eta, batch, epoch)
time_end = time.time()
time_gd_spend = time_end - time_start
loss_f = np.array(loss_f)

x_min = min(min(x1[:, 0]), min(x2[:, 0]))
x_max = max(max(x1[:, 0]), max(x2[:, 0]))
y_min = min(min(x1[:, 1]), min(x2[:, 1]))
y_max = max(max(x1[:, 1]), max(x2[:, 1]))
x_co = np.linspace(x_min - 1, x_max + 1)

'''print("--------------广义逆结果统计--------------")
print("w=", w_lg)
statistic(w_lg, x_train, y_train, n_train, x_test, y_test, n_test)
print("算法运行时间=", time_lg_spend, "s")

plt.figure("广义逆算法")
str1 = "gen_inverse, x1~N(%s,%s), x2~N(%s,%s)" % (u1, s1, u2, s2)
plt.title(str1)
# z_pla = -(w_lg[0][0] / w_lg[0][1]) * x_co
z_pla = -(w_lg[0][0] / w_lg[0][1]) * x_co - w_lg[0][2] / w_lg[0][1]
plt.scatter(x1[:, 0], x1[:, 1], c='r')
plt.scatter(x2[:, 0], x2[:, 1], c='b')
plt.plot(x_co, z_pla, c='g')
plt.xlim(x_min - 1, x_max + 1)
plt.ylim(y_min - 1, y_max + 1)
'''
print("--------------梯度下降结果统计--------------")
print("w=", w)
print("迭代次数=", iteration)
print("损失函数=", loss_f[len(loss_f) - 1, 1])
statistic(w, x_train, y_train, x_test, y_test)
print("算法运行时间=", time_gd_spend, "s")

plt.figure("梯度下降算法")
str2 = "grandient_descent, x1~N(%s,%s), x2~N(%s,%s)" % (u1, s1, u2, s2)
plt.title(str2)
z_gd = -(w[0] / w[1]) * x_co - w[2] / w[1]
plt.scatter(x1[:, 0], x1[:, 1], c='r')
plt.scatter(x2[:, 0], x2[:, 1], c='b')
plt.plot(x_co, z_gd, c='g')
plt.xlim(x_min - 1, x_max + 1)
plt.ylim(y_min - 1, y_max + 1)

plt.figure("梯度下降损失函数")
plt.title(str2)

plt.plot(loss_f[:, 0], loss_f[:, 1], c='k')

plt.show()
