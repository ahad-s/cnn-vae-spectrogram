import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.manifold import TSNE

import pickle

import sys

z_mu_new = pickle.load(open("z_mu.p", "rb"))
y_val = pickle.load(open("y_val.p", "rb"))
artists = pickle.load(open("artists.p", "rb"))

pp = 30
latent_dim = 50

art_back = {}
for k, v in artists.items():
	art_back[v] = k


#y_val[0] = "test"
USE_TSNE = latent_dim > 2

for i in range(max(y_val)):
	new_y_val = np.copy(y_val)
	for j in range(len(new_y_val)):
		y_j = new_y_val[j]
		if y_i != j:
			new_y_val[j] = i+1
	if USE_TSNE:
	    reducer = TSNE(n_components=2, perplexity=pp,verbose=1)
	    z_mu_2 = reducer.fit_transform(z_mu_new)
	    plt.figure(figsize=(10, 10))
	    plt.scatter(z_mu_2[:, 0], z_mu_2[:, 1], c=new_y_val, cmap='brg')
	    plt.colorbar()
	#     plt.show()
	else:
	    plt.figure(figsize=(10, 10))
	    plt.scatter(z_mu_new[:, 0], z_mu_new[:, 1], c=y_val, cmap='brg')
	    plt.colorbar()
	#     plt.show()
	    
	plt.savefig(str(art_back[i]) + "-test-pp" + str(pp) +"-lat" + str(latent_dim) + " ".join(sys.argv[1:]) + ".png")
