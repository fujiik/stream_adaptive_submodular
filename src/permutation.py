#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import random

dataset = "australian"
if dataset == "wdbc":
    n = 569 # WDBC
elif dataset == "australian":
    n = 690 # australian
elif dataset == "mnist":
    n = 14780
for i in range(100):
    a = np.arange(n)
    np.random.shuffle(a)
    print(a)
    filename = "./permutation/%s/per_%s%03d.txt" % (dataset, dataset, i)
    np.savetxt(filename, a, delimiter=",", fmt="%d")
