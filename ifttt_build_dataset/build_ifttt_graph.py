import numpy as np 
import pandas as pd
import json, dgl, os, torch
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

embedding_size = 300
dataset_name = 'ifttt_dataset_dgl.bin'
data_source = '../ifttt_build_dataset/'


class IFTTTGraphDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='ifttt_graph')

    def process(self):
        edges = pd.read_csv(data_source + 'graph_edges.csv')
        properties = pd.read_csv(data_source + 'graph_properties.csv')
        with open(data_source + 'embedding.json','r',encoding='utf8') as fp:
            embedding_dict = json.load(fp)[0]

        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)

            g.ndata['embedding'] = torch.zeros(g.num_nodes(), embedding_size)
            g.ndata['embedding'][src] = torch.Tensor([embedding_dict[str(i)] for i in src])
            g.ndata['embedding'][dst] = torch.Tensor([embedding_dict[str(i)] for i in dst])
            # g = dgl.add_self_loop(g)
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)
        graph_labels = {"glabel": self.labels}

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(data_source, dataset_name)
        save_graphs(graph_path, self.graphs, {'labels': self.labels})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(data_source, dataset_name)
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(data_source, dataset_name)
        return os.path.exists(graph_path)

if __name__ == "__main__":
    IFTTTGraphDataset=IFTTTGraphDataset()
