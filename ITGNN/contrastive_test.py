import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def load_embed_label(file_name):
	embed_label = torch.load(file_name, map_location=torch.device('cuda:0'))
	embed_0 = []
	embed_1 = []
	for idx, embed_label in enumerate(embed_label):
		for kk in range(embed_label[0].size(0)):
			if embed_label[1][kk] == 1:
				embed_1.append(embed_label[0][kk])
			else:
				embed_0.append(embed_label[0][kk])
	
	# negative embeddings
	embed_0 = torch.stack(embed_0)
	centroid_0 = torch.mean(embed_0, 0)

	sample2cent0 = []
	for kk in range(len(embed_0)):
		sample2cent0.append(torch.linalg.norm(embed_0[kk] - centroid_0))

	sample2cent0 = torch.stack(sample2cent0)
	med_sample2cent0 = torch.median(sample2cent0)
	MAD_0 = torch.median(sample2cent0-med_sample2cent0)

	# positive_embeddings
	embed_1 = torch.stack(embed_1)
	centroid_1 = torch.mean(embed_1, 0)

	sample2cent1 = []
	for kk in range(len(embed_1)):
		sample2cent1.append(torch.linalg.norm(embed_1[kk] - centroid_1))

	sample2cent1=torch.stack(sample2cent1)
	med_sample2cent1 = torch.median(sample2cent1)
	MAD_1 = torch.median(sample2cent1-med_sample2cent1)
	
	return centroid_0, centroid_1, MAD_0, MAD_1

def draw_figure(file_name="train_saved_embed_label.pt"):

	embed_label = torch.load(file_name, map_location=torch.device('cpu'))
	saved_embed_cpu = [embed.detach().numpy() for embed, label in embed_label]

	saved_embed = []
	for item in saved_embed_cpu:
		for kk in item:
			saved_embed.append(kk)
	saved_embed = saved_embed[0:1500]

	tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, init="pca")
	reduced_data = tsne.fit_transform(saved_embed)
	kmeans = KMeans(init="k-means++", n_clusters=2, n_init=50)
	kmeans.fit(reduced_data.astype('double'))

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
	y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Obtain labels for each point in mesh. Use last trained model.
	Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
	# Z = db.labels_

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)

	plt.clf()
	plt.imshow(
		Z,
		interpolation="nearest",
		extent=(xx.min(), xx.max(), yy.min(), yy.max()),
		cmap=plt.cm.tab20,
		aspect="auto",
		origin="lower",
	)

	plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
	# Plot the centroids as a white X
	centroids = kmeans.cluster_centers_
	plt.scatter(
		centroids[:, 0],
		centroids[:, 1],
		marker="x",
		s=300,
		linewidths=3,
		color="w",
		zorder=10,
	)

	fig = plt.gcf()
	ax = fig.gca()
	circle2 = plt.Circle((10, 95), 10, color='r', fill=False, linewidth=2.5)
	ax.add_patch(circle2)

	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())

	fig = plt.gcf()
	plt.show()
	fig.savefig('constrative_points.pdf', dpi=300, transparent=True, bbox_inches='tight')
