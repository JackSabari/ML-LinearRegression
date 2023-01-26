import numpy as np


phi = np.loadtxt('data/train.csv', dtype='float', delimiter=',', skiprows=1,usecols=tuple(range(1, 14)))

y = np.loadtxt('data/train.csv', dtype='float', delimiter=',', skiprows=1,usecols=14, ndmin=2)

phi_test = np.loadtxt('data/test.csv', dtype='float', delimiter=',',skiprows=1, usecols=tuple(range(1, 14)))

phi_test = np.concatenate((phi_test, np.ones((105, 1))), axis=1)

phi = np.concatenate((phi, np.ones((400, 1))), axis=1)

for i in range(0, 13):
    column_max = max(phi[:, i])
    column_min = min(phi[:, i])
    phi[:, i] = (phi[:, i] - column_min) / (column_max - column_min)
    phi_test[:, i] = (phi_test[:, i] - column_min) / (column_max - column_min)

y = np.log(y)

def delta_w(p, phi, w):
    if p == 2:
        deltaw = (2 * (np.dot(np.dot(np.transpose(phi), phi), w) - np.dot(np.transpose(phi), y)) +
                  lambd * p * np.power(np.absolute(w), (p - 1)))
    if p < 2 and p > 1:
        deltaw = (2 * (np.dot(np.dot(np.transpose(phi), phi), w) -np.dot(np.transpose(phi), y)) +
                  lambd * p * np.power(np.absolute(w), (p - 1)) * np.sign(w))
    return deltaw


filenames = {'output.csv': 2.0,'output_p1.csv': 1.75,'output_p2.csv': 1.5,'output_p3.csv': 1.3}

for (fname, p) in filenames.items():
    w = np.zeros((14, 1))
    lambd = 0.2
    t = 0.00012

    w_new = w - t * delta_w(p, phi, w)

    i = 0
    while(np.linalg.norm(w_new-w) > 10 ** -10):
        w = w_new
        w_new = w - t * delta_w(p, phi, w)
        i = i + 1

    id_test = np.loadtxt('data/test.csv', dtype='int', delimiter=',',skiprows=1, usecols=0, ndmin=2)
    y_test = np.exp(np.dot(phi_test, w_new))
    np.savetxt(fname, np.concatenate((id_test, y_test), axis=1),delimiter=',', fmt=['%d', '%f'], header='ID,MEDV', comments='')
