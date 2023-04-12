import numpy as np 
import pandas as pd
import json, dgl, torch, os
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
# dgl: 0.7.2

data_source = "../heterograph_dataset/"

ifttt2smartthings = data_source+"ifttt_interact_smartthings.csv"
alexa2smartthings = data_source+"alexa_interact_smartthings.csv" 
alexa2ifttt = data_source+"alexa_interact_ifttt.csv" 
alexa2alexa = data_source+"alexa_interact_alexa.csv" 
ifttt2ifttt = data_source+"ifttt_interact_ifttt.csv" 
smart2smart = data_source+"smt_interact_smt.csv"
graph_properties = data_source+"graph_properties.csv"

class HeteroGraphDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='heterograph')
        
    def process(self):
        ift2smt = pd.read_csv(ifttt2smartthings)
        alx2smt = pd.read_csv(alexa2smartthings)
        alx2ift = pd.read_csv(alexa2ifttt)
        alx2alx = pd.read_csv(alexa2alexa)
        ift2ift = pd.read_csv(ifttt2ifttt)
        smt2smt = pd.read_csv(smart2smart)
        properties = pd.read_csv(graph_properties)

        with open(data_source+"ifttt_embedding.json",'r',encoding='utf8') as fp:
            ifttt_embedding_dict = json.load(fp)[0]
            for i in range(307, 1000):
                ifttt_embedding_dict[str(i)] = np.zeros(300)

        with open(data_source+'smt_embedding.json','r',encoding='utf8') as fp1:
            smt_embedding_dict = json.load(fp1)[0]

        with open(data_source+'alexa_embedding.json','r',encoding='utf8') as fp2:
            alexa_embedding_dict = json.load(fp2)[0]
            for i in range(3274, 4000):
                alexa_embedding_dict[str(i)] = np.zeros(512)

        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        ift2smt_edge_group = ift2smt.groupby('graph_id')
        alx2smt_edge_group = alx2smt.groupby('graph_id')
        alx2ift_edge_group = alx2ift.groupby('graph_id')
        alx2alx_edge_group = alx2alx.groupby('graph_id')
        ift2ift_edge_group = ift2ift.groupby('graph_id')
        smt2smt_edge_group = smt2smt.groupby('graph_id')

        # For each graph ID...
        graph_data={}
        ift2smt_data={}
        alx2smt_data={}
        alx2ift_data={}
        alx2alx_data={}
        ift2ift_data={}
        smt2smt_data={}

        for graph_id in ift2smt_edge_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = ift2smt_edge_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            ift2smt_data[graph_id]={('I', 'U', 'S'): (src, dst)}


        for graph_id in alx2smt_edge_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = alx2smt_edge_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            alx2smt_data[graph_id]={('A', 'V', 'S'): (src, dst)}

        for graph_id in alx2ift_edge_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = alx2ift_edge_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            alx2ift_data[graph_id]={('A', 'W', 'I'): (src, dst)}

        for graph_id in alx2alx_edge_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = alx2alx_edge_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            alx2alx_data[graph_id]={('A', 'X', 'A'): (src, dst)}

        for graph_id in ift2ift_edge_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = ift2ift_edge_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            ift2ift_data[graph_id]={('I', 'Y', 'I'): (src, dst)}

        for graph_id in smt2smt_edge_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = smt2smt_edge_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            smt2smt_data[graph_id]={('S', 'Z', 'S'): (src, dst)}

        for graph_id in properties['graph_id']:
            if graph_id in ift2smt_data.keys():
                for k, v in ift2smt_data[graph_id].items():
                    graph_data[k]=v
            if graph_id in alx2smt_data.keys():
                for k, v in alx2smt_data[graph_id].items():
                    graph_data[k]=v
            if graph_id in alx2ift_data.keys():
                for k, v in alx2ift_data[graph_id].items():
                    graph_data[k]=v
            if graph_id in alx2alx_data.keys():
                for k, v in alx2alx_data[graph_id].items():
                    graph_data[k]=v
            if graph_id in ift2ift_data.keys():
                for k, v in ift2ift_data[graph_id].items():
                    graph_data[k]=v
            if graph_id in smt2smt_data.keys():
                for k, v in smt2smt_data[graph_id].items():
                    graph_data[k]=v

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.heterograph(graph_data)
            label = label_dict[graph_id]
            
            if 'S' in g.ntypes:
                g.nodes['S'].data['embedding'] = torch.Tensor(np.array([smt_embedding_dict[str(i.item())] for i in g.nodes('S')]))
            if 'I' in g.ntypes:
                g.nodes['I'].data['embedding'] = torch.Tensor(np.array([ifttt_embedding_dict[str(i.item())] for i in g.nodes('I')]))
            if 'A' in g.ntypes:
                g.nodes['A'].data['embedding'] = torch.Tensor(np.array([alexa_embedding_dict[str(i.item())] for i in g.nodes('A')]))
            
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        # save graphs and labels
        graph_path = os.path.join(data_source, 'heterograph_dataset_dgl.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(data_source, 'heterograph_dataset_dgl.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(data_source, 'heterograph_dataset_dgl.bin')
        return os.path.exists(graph_path)


if __name__ == "__main__":
    HeteroGraphData=HeteroGraphDataset()