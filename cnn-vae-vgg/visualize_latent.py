import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.manifold import TSNE

import pickle

import sys

ll = lambda f: pickle.load(open(f, "rb"))

z_mu_new = pickle.load(open("z_mu.p", "rb"))
y_val = pickle.load(open("y_val.p", "rb"))
artists = pickle.load(open("artists.p", "rb"))

priors = ll("priors.p")

pp = 30
latent_dim = 50
n_samples = 50 # num samples for prior

art_back = {}
for k, v in artists.items():
    art_back[v] = k


#y_val[0] = "test"
USE_TSNE = latent_dim > 2

for i in range(max(y_val)):
    new_y_val = np.copy(y_val)
    for j in range(len(new_y_val)):
        y_j = new_y_val[j]
        if y_j != j:
            new_y_val[j] = i+1

    ps = []
    for k in n_samples:
        z_mu_new = np.append(z_mu_new, np.random.normal(priors[i][0], priors[i][1]), axis=0)
        new_y_val = np.append(new_y_val, i+2)    

    if USE_TSNE:
        if i == 0:
            reducer = TSNE(n_components=2, perplexity=pp,verbose=1)
            z_mu_2 = reducer.fit_transform(z_mu_new)
        plt.figure(figsize=(10, 10))
        x = z_mu_2[:,0]
        y = z_mu_2[:,1]
        plt.scatter(z_mu_2[:, 0], z_mu_2[:, 1], c=new_y_val, cmap='brg')
        plt.colorbar()
    #     plt.show()
    else:
        plt.figure(figsize=(10, 10))
        plt.scatter(z_mu_new[:, 0], z_mu_new[:, 1], c=y_val, cmap='brg')
        plt.colorbar()
    #     plt.show()
        
    plt.savefig(str(art_back[i]) + "-test-pp" + str(pp) +"-lat" + str(latent_dim) + " ".join(sys.argv[1:]) + ".png")


