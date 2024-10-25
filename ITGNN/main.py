import sys
sys.path.append("..")
from smt_build_dataset.build_smt_graph import SMTGraphDataset
from ifttt_build_dataset.build_ifttt_graph import IFTTTGraphDataset
from heterograph_dataset.build_heter_graph import HeteroGraphDataset

import os, json
from datetime import datetime
from time import time
from copy import deepcopy

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import random_split

from dgl.dataloading import GraphDataLoader

from networks import GraphClassifier
from utils import get_stats, parse_args
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

def contrastive_loss(logits_embedding: Tensor, labels: Tensor, device:torch.device):
    loss = 0
    threshold = 80
    len_embedding = len(logits_embedding)

    dis_matrix = torch.cdist(logits_embedding, logits_embedding)
    eye_matrix = torch.eye(len_embedding).to(device)
    inv_matrix = 1 - eye_matrix
    positive_loss = torch.sum(torch.square(torch.mul(dis_matrix, eye_matrix)))
    negative_loss = torch.sum(torch.relu(torch.square(threshold - torch.mul(dis_matrix, inv_matrix))))

    loss = (positive_loss+negative_loss)/(len_embedding*len_embedding)

    return loss


def compute_loss(cls_logits:Tensor, labels:Tensor,
                 logits_s1:Tensor, logits_s2:Tensor,
                 epoch:int, total_epochs:int, device:torch.device):
    # classification loss
    classify_loss = F.nll_loss(cls_logits, labels.to(device))

    # loss for vertex infomax pooling
    scale1, scale2 = logits_s1.size(0) // 2, logits_s2.size(0) // 2
    s1_label_t, s1_label_f = torch.ones(scale1), torch.zeros(scale1)
    s2_label_t, s2_label_f = torch.ones(scale2), torch.zeros(scale2)
    s1_label = torch.cat((s1_label_t, s1_label_f), dim=0).to(device)
    s2_label = torch.cat((s2_label_t, s2_label_f), dim=0).to(device)

    pool_loss_s1 = F.binary_cross_entropy_with_logits(logits_s1, s1_label)
    pool_loss_s2 = F.binary_cross_entropy_with_logits(logits_s2, s2_label)
    pool_loss = (pool_loss_s1 + pool_loss_s2) / 2
    
    loss = classify_loss + (2 - epoch / total_epochs) * pool_loss

    return loss


def train(model:torch.nn.Module, optimizer, trainloader,
          device, curr_epoch, total_epochs, mode = "classification"):
    model.train()

    total_loss = 0.
    num_batches = len(trainloader)

    train_saved_embed_label = []

    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out, logits_embedding, l1, l2 = model(batch_graphs, 
                            batch_graphs.ndata["embedding"])
        if mode == "contrastive":
            loss = contrastive_loss(logits_embedding, batch_labels, device)
            train_saved_embed_label.append((logits_embedding, batch_labels))
        else:
            loss = compute_loss(out, batch_labels, l1, l2,
                    curr_epoch, total_epochs, device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if len(train_saved_embed_label)!=0:
        torch.save(train_saved_embed_label,  'train_saved_embed_label.pt')    
    
    return total_loss / num_batches


@torch.no_grad()
def test(model:torch.nn.Module, loader, device, mode = "classification"):
    model.eval()

    correct = 0.
    num_graphs = 0
    y_true = []
    y_pred = []
    out_cpu_list = []

    test_saved_embed_label=[]

    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        out, logits_embedding, _, _ = model(batch_graphs, batch_graphs.ndata["embedding"])

        test_saved_embed_label.append((logits_embedding, batch_labels))
 
        out_cpu = out.cpu()
        pred = out.argmax(dim=1)
        pred = pred.cpu()
        targets_copy = deepcopy(batch_labels)
        y_true.extend(targets_copy)
        y_pred.extend(pred)
        out_cpu_list.extend(out_cpu[:,1])

        del batch_graphs
        del batch_labels
        del batch

    torch.save(test_saved_embed_label, 'test_saved_embed_label.pt')

    acc = accuracy_score(y_true, y_pred)
    prfs = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_true, out_cpu_list, average='weighted')
    print('acc, weighted prfs and auc on test dataset', acc, prfs, auc)

    return acc, prfs, auc

def main(args):

    if args.dataset == "IFTTT":
        dataset = IFTTTGraphDataset()
        saved_model_name = "./ifttt_saved_model.pt"
        print("I am using IFTTT Dataset")
    elif args.dataset == "SMT":
        dataset = SMTGraphDataset()
        saved_model_name = "./smt_saved_model.pt"
        print("I am using SMT Dataset")
    elif args.dataset == "Hetero":
        dataset = HeteroGraphDataset()
        saved_model_name = "./hetero_saved_model.pt"
        print("I am using Heter Dataset")            

    num_training = int(len(dataset) * float(args.split_ratio))
    num_test = len(dataset) - num_training
    train_set, test_set = random_split(dataset, [num_training, num_test], generator=torch.Generator().manual_seed(42))

    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, num_workers=16)
    test_loader = GraphDataLoader(test_set, batch_size=args.batch_size, num_workers=16)

    device = torch.device(args.device)

    num_feature, num_classes = 300, 2
    args.in_dim = int(num_feature)
    args.out_dim = int(num_classes)
    args.edge_feat_dim = 0 # No edge feature in datasets that we use.
    
    model = GraphClassifier(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)

    best_test_acc = 0.0
    best_prfs = []
    best_auc = 0
    best_epoch = -1
    train_times = []

    if os.path.isfile(saved_model_name):
        model.load_state_dict(torch.load(saved_model_name))
        model.eval()
        test_acc, prfs, auc = test(model, test_loader, device)
        return test_acc, prfs, auc, 0
    else:
        for e in range(args.epochs):
            s_time = time()
            train_loss = train(model, optimizer, train_loader, device,
                            e, args.epochs, args.mode)
            train_times.append(time() - s_time)

            test_acc, prfs, auc = test(model, test_loader, device, args.mode)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = e + 1
                best_prfs = prfs
                best_auc = auc

            if (e + 1) % args.print_every == 0:
                log_format = "Epoch {}: loss={:.4f}, test_acc={:.4f}, best_test_acc={:.4f}"
                print(log_format.format(e + 1, train_loss, test_acc, best_test_acc))
        print("Best Epoch {}, final test acc {:.4f}".format(best_epoch, best_test_acc))

        torch.save(model.state_dict(), saved_model_name)
    return best_test_acc, best_prfs, best_auc, sum(train_times) / len(train_times)


if __name__ == "__main__":
    args = parse_args()
    res = []
    train_times = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        acc, prfs, auc, train_time = main(args)
        res.append(acc)
        res.append(prfs)
        res.append(auc)
        train_times.append(train_time)

    mean, err_bd = get_stats([res[0]], conf_interval=False)
    print("mean acc: {:.4f}, error bound: {:.4f}".format(mean, err_bd))

    out_dict = {"hyper-parameters": vars(args),
                "result_date": str(datetime.now()),
                "result": "{:.4f}".format(mean),
                "train_time": "{:.4f}".format(sum(train_times) / len(train_times)),
                "details": res}

    with open(os.path.join(args.output_path, "{}.log".format(args.dataset)), "w") as f:
        json.dump(out_dict, f, sort_keys=True, indent=4)
