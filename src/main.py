#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata

epsilon = 0.1
k_max = 100
probs = []
prob = 1
p = (epsilon / (1 - epsilon))
for i in range(k_max):
    probs.append(prob)
    prob *= p

def hit_and_run(obss, d, N, redundancy=True):
    a = np.zeros([10, d])
    hyps = np.zeros([N, d])
    w = random_point(d)
    for i in range(N):
        theta = random_vector(d)

        # obtain the lower and upper bounds of \rho
        A = np.dot(theta, theta)
        B = 2 * np.dot(theta, w)
        C = np.dot(w, w) - 1
        D = np.sqrt(B ** 2 - 4 * A * C)
        l = - (B + D) / A / 2
        u = (D - B) / A / 2

        # obtain the intersection with boundaries
        n = len(obss)
        ints = []
        plus_count = 0
        for j in range(n):
            if redundancy:
                den = np.dot(obss[j][0], theta)
                intersection = - np.dot(obss[j][0], w) / den
                #a[3+j,:] = w + intersection * theta
                if den * obss[j][1] > 0:
                    ints.append((intersection, 1))
                    plus_count += 1
                else:
                    ints.append((intersection, -1))
            else:
                A = np.dot(theta, theta)
                B = np.dot(theta, obss[j][0] - 2 * w)
                C = np.dot(obss[j][0] - w, w)
                det = B ** 2 - 4 * A * C
                if det <= 0:
                    plus_count += 1
                else:
                    ints.append((- (B - np.sqrt(det)) / A / 2, -1))
                    ints.append((- (B + np.sqrt(det)) / A / 2, +1))
                    plus_count += 1

        dtype = [('rho', float), ('sgn', int)]
        ints.append((l, 0))
        ints.append((u, 0))
        ints = np.array(ints, dtype=dtype)
        ints = np.sort(ints, order='rho')

        # calculate the cumulative probability
        sum_prob = 0
        miss = plus_count
        prev = l
        for rho, sgn in ints:
            if rho >= l and rho <= u:
                sum_prob += (rho - prev) * probs[miss]
                prev = rho
            miss -= sgn

        # sample a hypothesis
        r = np.random.rand()
        r *= sum_prob
        miss = plus_count
        prev = l
        for rho, sgn in ints:
            if rho >= l and rho <= u:
                total = (rho - prev) * probs[miss]
                if total > r:
                    h = prev + r / probs[miss]
                    break
                else:
                    r -= total
                prev = rho
            miss -= sgn

        hyps[i,:] = w + h * theta
        w = hyps[i,:]
    return hyps

def random_point(d):
    w_0 = np.random.normal(0, 1, d) 
    w_0 /= np.sqrt(np.dot(w_0, w_0))
    r = np.random.rand()
    w_0 *= pow(r, 1.0 / d)
    return w_0

def random_vector(d):
    w_0 = np.random.normal(0, 1, d) 
    w_0 /= np.sqrt(np.dot(w_0, w_0))
    return w_0
    
def read_data(name="wdbc"):
    if name == "wdbc":
        url = './data/wdbc/wdbc.data'
        f = open(url, 'r')
        lines = f.readlines()
        f.close()

        X = []
        y = []
        for line in lines:
            s = line.rstrip().split(',')
            X.append([float(i) for i in  s[2:]])
            if s[1] == 'M':
                y.append(1)
            else:
                y.append(-1)
        X = np.array(X)
        y = np.array(y)
        return X, y
    
    elif name == "australian":
        url = "./data/australian/australian.txt"
        a = np.loadtxt(url, delimiter=" ")
        X = a[:,:-1]
        y = a[:,-1]
        print(X)
        print(y)
        y[y == 0] = -1
        print(y)
        return X, y

    elif name == "mnist":
        mnist = fetch_mldata('MNIST original')

        #extract 0s and 1s
        mask = np.any(np.c_[mnist.target == 0, mnist.target == 1], axis=1)
        X = mnist.data[mask]
        y = mnist.target[mask] * 2 - 1
        pca = PCA(n_components = 10)
        pca.fit(mnist.data)
        X = pca.transform(X)
        return X, y

    else:
        print("dataset name error")
        return

def random_select(X, y, k):
    n, d = X.shape
    labeled = random.sample(range(n), k)
    while np.all(y[labeled[:10]] == 1) or np.all(y[labeled[:10]] == -1):
        labeled = random.sample(range(n), k)
    return labeled

def pool_greedy(X, y, k, N):
    n, d = X.shape
    obss = []
    init = 0
    labeled = []
    h = []
    for i in range(k-init):
        hyps = hit_and_run(obss, d, N)
        min_count = N
        for j in range(n):
            count = np.count_nonzero(np.dot(np.array(hyps), X[j]) > 0)
            if abs(N / 2 - count) < min_count:
                min_sample = [j]
                min_count = abs(N / 2 - count)
            elif abs(N / 2 - count) == min_count:
                min_sample.append(j)
        min_sample = np.random.choice(min_sample)
        obss.append((X[min_sample], y[min_sample]))
        labeled.append(min_sample)
    return labeled

def pool_uncertainty(X, y, k):
    n, d = X.shape
    init = 10
    labeled = random.sample(range(n), init)
    while np.all(y[labeled] == 1) or np.all(y[labeled] == -1):
        labeled = random.sample(range(n), init)
    w = random_point(d)
    for i in range(k-init):
        min_dist = 100 # some large number
        for j in range(n):
            dist = np.abs(np.dot(w, X[j]))
            if dist < min_dist:
                min_sample = j
                min_dist = dist
        labeled.append(min_sample)

        clf = SVC(C = 1, kernel='linear')
        clf.fit(X[labeled], y[labeled])
        w = clf.coef_
    return labeled

def stream_uncertainty(X, y, k, arr):
    n, d = X.shape
    init = 10
    kp = k - init
    labeled = random.sample(range(n), init)
    while np.all(y[labeled] == 1) or np.all(y[labeled] == -1):
        labeled = random.sample(range(n), init)
    clf = SVC(C = 1, kernel='linear')
    clf.fit(X[labeled], y[labeled])
    w = clf.coef_
    for i in range(kp):
        min_dist = 100
        for j in list(arr[n/kp*i:n/kp*(i+1)]):
            dist = np.abs(np.dot(w, X[j]))
            if dist < min_dist:
                min_sample = j
                min_dist = dist
        labeled.append(min_sample)
        clf = SVC(C = 1, kernel='linear')
        clf.fit(X[labeled], y[labeled])
        w = clf.coef_
    return labeled

def secretary_uncertainty(X, y, k, arr):
    n, d = X.shape
    init = 10
    kp = k - init
    labeled = random.sample(range(n), init)
    while np.all(y[labeled] == 1) or np.all(y[labeled] == -1):
        labeled = random.sample(range(n), init)
    clf = SVC(C = 1, kernel='linear')
    clf.fit(X[labeled], y[labeled])
    w = clf.coef_
    for i in range(kp):
        min_dist = 100
        for l in range(int(n/kp*i), int(n/kp*(i+1))):
            j = arr[l]
            if l == int(n/kp*(i+1)) - 1:
                labeled.append(j)
                break
            dist = np.abs(np.dot(w, X[j]))
            if dist < min_dist:
                if l < n / kp * (i + 1 / np.e):
                    min_sample = j
                    min_dist = dist
                else:
                    labeled.append(j)
                    break
        clf = SVC(C = 1, kernel='linear')
        clf.fit(X[labeled], y[labeled])
        w = clf.coef_
    return labeled

def pool_uncertainty_log(X, y, k):
    n, d = X.shape
    init = 10
    labeled = random.sample(range(n), init)
    while np.all(y[labeled] == 1) or np.all(y[labeled] == -1):
        labeled = random.sample(range(n), init)
    w = random_point(d)
    for i in range(k-init):
        min_dist = 100 # some large number
        for j in range(n):
            dist = np.abs(np.dot(w, X[j]))
            if dist < min_dist:
                min_sample = j
                min_dist = dist
        labeled.append(min_sample)

        clf = LogisticRegression(C = 1)
        clf.fit(X[labeled], y[labeled])
        w = clf.coef_
    return labeled

def stream_uncertainty_log(X, y, k, arr):
    n, d = X.shape
    init = 10
    kp = k - init
    labeled = random.sample(range(n), init)
    while np.all(y[labeled] == 1) or np.all(y[labeled] == -1):
        labeled = random.sample(range(n), init)
    clf = LogisticRegression(C = 1)
    clf.fit(X[labeled], y[labeled])
    w = clf.coef_
    for i in range(kp):
        min_dist = 100
        for j in list(arr[n/kp*i:n/kp*(i+1)]):
            dist = np.abs(np.dot(w, X[j]))
            if dist < min_dist:
                min_sample = j
                min_dist = dist
        labeled.append(min_sample)
        clf = LogisticRegression(C = 1)
        clf.fit(X[labeled], y[labeled])
        w = clf.coef_
    return labeled

def secretary_uncertainty_log(X, y, k, arr):
    n, d = X.shape
    init = 10
    kp = k - init
    labeled = random.sample(range(n), init)
    while np.all(y[labeled] == 1) or np.all(y[labeled] == -1):
        labeled = random.sample(range(n), init)
    clf = LogisticRegression(C = 1)
    clf.fit(X[labeled], y[labeled])
    w = clf.coef_
    for i in range(kp):
        min_dist = 100
        for l in range(int(n/kp*i), int(n/kp*(i+1))):
            j = arr[l]
            if l == int(n/kp*(i+1)) - 1:
                labeled.append(j)
                break
            dist = np.abs(np.dot(w, X[j]))
            if dist < min_dist:
                if l < n / kp * (i + 1 / np.e):
                    min_sample = j
                    min_dist = dist
                else:
                    labeled.append(j)
                    break
        clf = LogisticRegression(C = 1)
        clf.fit(X[labeled], y[labeled])
        w = clf.coef_
    return labeled

def adaptive_stream(X, y, k, arr, N):
    n, d = X.shape

    obss = []
    labeled = []
    for i in range(k):
        hyps = hit_and_run(obss, d, N)
        min_count = N
        for j in list(arr[n/k*i:n/k*(i+1)]):
            count = np.count_nonzero(np.dot(np.array(hyps), X[j]) > 0)
            if abs(N / 2 - count) < min_count:
                min_sample = j
                min_count = abs(N / 2 - count)
        obss.append((X[min_sample], y[min_sample]))
        labeled.append(min_sample)
    return labeled

def adaptive_secretary(X, y, k, arr, N):
    n, d = X.shape

    obss = []
    labeled = []
    for i in range(k):
        hyps = hit_and_run(obss, d, N)
        min_count = N
        for l in range(int(n/k*i), int(n/k*(i+1))):
            j = arr[l]
            count = np.count_nonzero(np.dot(np.array(hyps), X[j]) > 0)
            if l == int(n/k*(i+1)) - 1:
                obss.append((X[j], y[j]))
                labeled.append(j)
                break
            if abs(N / 2 - count) < min_count:
                if l < n / k * (i + 1 / np.e):
                    min_sample = j
                    min_count = abs(N / 2 - count)
                else:
                    obss.append((X[j], y[j]))
                    labeled.append(j)
                    break
    return labeled

def labeling_with(method, X, y, k, arr, N, i, dataset, test_clf):
    if method == "random":
        return random_select(X, y, k)
    elif method == "pool_greedy":
        if test_clf != "svm" or k < 50:
            return pool_greedy_emulate(X, y, k, N, i, dataset)
        else:
            return pool_greedy(X, y, k, N)
    elif method == "pool_uncertainty":
        if test_clf == "svm":
            return pool_uncertainty(X, y, k)
        else:
            return pool_uncertainty_log(X, y, k)
    elif method == "stream_uncertainty":
        if test_clf == "svm":
            return stream_uncertainty(X, y, k, arr)
        else:
            return stream_uncertainty_log(X, y, k, arr)
    elif method == "secretary_uncertainty":
        if test_clf == "svm":
            return secretary_uncertainty(X, y, k, arr)
        else:
            return secretary_uncertainty_log(X, y, k, arr)
    elif method == "adaptive_stream":
        if test_clf == "svm":
            return adaptive_stream(X, y, k, arr, N)
        else:
            return adaptive_stream_emulate(X, y, k, N, i, dataset)
    elif method == "adaptive_secretary":
        if test_clf == "svm":
            return adaptive_secretary(X, y, k, arr, N)
        else:
            return adaptive_secretary_emulate(X, y, k, N, i, dataset)

def pool_greedy_emulate(X, y, k, N, i, dataset):
    filename = "./result/%s/labeled_%s/labeled_%s_k%02d_N%04d_t%03d.txt" % (dataset, "pool_greedy", "pool_greedy", 50, N, i)
    labeled_50 = np.loadtxt(filename, dtype=int)
    labeled = labeled_50[:k]
    print(labeled)
    return labeled

def adaptive_stream_emulate(X, y, k, N, i, dataset):
    filename = "./result/%s/labeled_%s/labeled_%s_k%02d_N%04d_t%03d.txt" % (dataset, "adaptive_stream", "adaptive_stream", 50, N, i)
    labeled_50 = np.loadtxt(filename, dtype=int)
    labeled = labeled_50[:k]
    print(labeled)
    return labeled

def adaptive_secretary_emulate(X, y, k, N, i, dataset):
    filename = "./result/%s/labeled_%s/labeled_%s_k%02d_N%04d_t%03d.txt" % (dataset, "adaptive_secretary", "adaptive_secretary", 50, N, i)
    labeled_50 = np.loadtxt(filename, dtype=int)
    labeled = labeled_50[:k]
    print(labeled)
    return labeled

def convergence(dataset, test_clf = "svm"):
    X, y = read_data(dataset)
    n, d = X.shape
    for i in range(d):
        X[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i])
    X = np.c_[X, np.ones(n)]
    n, d = X.shape

    methods = ["random", "pool_uncertainty", "stream_uncertainty", "secretary_uncertainty", "pool_greedy", "adaptive_stream", "adaptive_secretary"]
    trial = 100
    N = 1000
    error_arr = np.zeros([len(methods), trial, 41])
    for m in range(len(methods)):
        method = methods[m]
        print("start %s" % method)
        for i in range(trial):
            filename = "./result/%s/labeled_%s/labeled_%s_k%02d_N%04d_t%03d.txt" % (dataset, method, method, 50, N, i)
            labeled = np.loadtxt(filename, dtype=int)
            print(labeled)

            for l in range(9, 50):
                if test_clf == "svm":
                    clf = SVC(C = 1, kernel='linear')
                else:
                    clf = LogisticRegression(C = 1)
                clf.fit(X[labeled[:l]], y[labeled[:l]])
                y_pred = clf.predict(X)
                error = np.count_nonzero(y - y_pred) * 1.0 / n
                error_arr[m, i, l-9] = error
        if test_clf == "svm":
            filename = "./plot/%s/convergence_%s.txt" % (dataset, method)
        else:
            filename = "./plot/%s_log/convergence_%s.txt" % (dataset, method)
        np.savetxt(filename, error_arr[m,:,:], delimiter=",", fmt="%f")

def experiment(dataset, test_clf = "svm"):
    X, y = read_data(dataset)
    n, d = X.shape

    # scaling
    for i in range(d):
        X[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i])

    X = np.c_[X, np.ones(n)]
    n, d = X.shape
    print("n: %d" % n)

    k = 30
    trial = 100
    N = 1000
    methods = ["pool_greedy"]
    error_arr = np.zeros([len(methods), trial])
    for m in range(len(methods)):
        method = methods[m]
        print("start %s" % method)
        for i in range(trial):
            arr = np.loadtxt("./permutation/%s/per_%s%03d.txt" % (dataset, dataset, i), dtype=int)
            labeled = labeling_with(method, X, y, k, arr, N, i, dataset, test_clf)
            
            if test_clf == "svm":
                filename = "./result/%s/labeled_%s/labeled_%s_k%02d_N%04d_t%03d.txt" % (dataset, method, method, k, N, i)
            else:
                filename = "./result/%s_log/labeled_%s/labeled_%s_k%02d_N%04d_t%03d.txt" % (dataset, method, method, k, N, i)
            np.savetxt(filename, labeled, delimiter=",", fmt="%d")

            if test_clf == "svm":
                clf = SVC(C = 1, kernel='linear')
            else:
                clf = LogisticRegression(C = 1)
            clf.fit(X[labeled], y[labeled])
            y_pred = clf.predict(X)
            error = np.count_nonzero(y - y_pred) * 1.0 / n
            error_arr[m, i] = error
            print("trial %02d: error %f" % (i, error))
        print(error_arr[m,:])
        if test_clf == "svm":
            filename = "./result/%s/error_%s/error_%s_k%02d_N%04d.txt" % (dataset, method, method, k, N)
        else:
            filename = "./result/%s_log/error_%s/error_%s_k%02d_N%04d.txt" % (dataset, method, method, k, N)
        np.savetxt(filename, error_arr[m,:], delimiter=",", fmt="%1.5f")

    print("result")
    for m in range(len(methods)):
        print("%20s: %f pm %f" % (methods[m], np.mean(error_arr[m,:]), np.std(error_arr[m,:])))

if __name__ == '__main__':
    experiment("australian")
    #experiment("mnist", "logistic")
    #convergence("wdbc")
