import torch
import numpy as np 
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

####### loading the training sample

# train_saved_embed_label = torch.load("train_saved_embed_label.pt", map_location=torch.device('cuda:0'))
train_saved_embed_label = torch.load("smt_train_saved_embed_label.pt", map_location=torch.device('cuda:0'))


embed_0 = []
embed_1 = []

for idx, embed_label in enumerate(train_saved_embed_label):
	for kk in range(embed_label[0].size(0)):
		if embed_label[1][kk] == 1:
			embed_1.append(embed_label[0][kk])
		else:
			embed_0.append(embed_label[0][kk])

####### the positive sample

embed_0 = torch.stack(embed_0)
centroid_0 = torch.mean(embed_0, 0)

# print(centroid_0)
sample2cent0 = []
for kk in range(len(embed_0)):
	# print(embed_0[kk], "embed_0[kk]")
	sample2cent0.append(torch.linalg.norm(embed_0[kk] - centroid_0))

sample2cent0 = torch.stack(sample2cent0)
med_sample2cent0 = torch.median(sample2cent0)

# print(med_sample2cent0)
# print(sample2cent0[180:200])
# print(sample2cent0-med_sample2cent0)

MAD_0 = torch.median(sample2cent0-med_sample2cent0)
# print(MAD_0)

####### the negative sample

embed_1 = torch.stack(embed_1)
centroid_1 = torch.mean(embed_1, 0)

sample2cent1 = []
for kk in range(len(embed_1)):
	sample2cent1.append(embed_1[kk] - centroid_1)

sample2cent1=torch.stack(sample2cent1)
med_sample2cent1 = torch.median(sample2cent1)

MAD_1 = torch.median(sample2cent1-med_sample2cent1)

####### loading the testing sample

# test_saved_embed_label = torch.load("test_saved_embed_label.pt", map_location=torch.device('cuda:0'))
test_saved_embed_label = torch.load("smt_test_saved_embed_label.pt", map_location=torch.device('cuda:0'))


test_embed_0 = []
test_embed_1 = []

y_true = []
y_pred = []

for idx, embed_label in enumerate(test_saved_embed_label):
	for kk in range(embed_label[0].size(0)):
		if embed_label[1][kk] == 1:
			test_embed_1.append(embed_label[0][kk])
		else:
			test_embed_0.append(embed_label[0][kk])

cnt = 0
for kk in range(len(test_embed_0)):
	aa = torch.linalg.norm(test_embed_0[kk]-centroid_0)
	bb = torch.linalg.norm(test_embed_0[kk]-centroid_1)

	Taa = torch.abs(aa-med_sample2cent0)
	Tbb = torch.abs(bb-med_sample2cent1)

	# print(MAD_0, MAD_1, Taa, Tbb)

	y_true.append(0)

	if aa < bb:
		cnt+=1
		y_pred.append(0)
	else:
		y_pred.append(1)

print(cnt/len(test_embed_0))

cnt = 0
for kk in range(len(test_embed_1)):
	aa = torch.linalg.norm(test_embed_1[kk]-centroid_0)
	bb = torch.linalg.norm(test_embed_1[kk]-centroid_1)
	y_true.append(1)

	if aa > bb:
		cnt+=1
		y_pred.append(1)
	else:
		y_pred.append(0)
print(cnt/len(test_embed_1))

prfs = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

print(prfs)
