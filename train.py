import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
import argparse
from torch_geometric.loader import DataLoader
from models import GCN, NodeDropping, RandomChoice, Identity, EdgeRemoving
from models import GConv, Encoder, DualBranchContrast, InfoNCE, FGNConv
from sklearn.svm import LinearSVC, SVC
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }

def from_predefined_split(data):
    assert all([mask is not None for mask in [data.train_mask, data.test_mask, data.val_mask]])
    num_samples = data.num_nodes
    indices = torch.arange(num_samples)
    return {
        'train': indices[data.train_mask],
        'valid': indices[data.val_mask],
        'test': indices[data.test_mask]
    }


def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split

        result = self.evaluate(x, y, split)
        return result

class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params):
        self.evaluator = evaluator
        self.params = params

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
        test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')
        test_acc = accuracy_score(y_test, classifier.predict(x_test))

        return {
            'micro_f1': test_micro,
            'macro_f1': test_macro,
            "acc": test_acc
        }

class SVMEvaluator(BaseSKLearnEvaluator):
    def __init__(self, linear=True, params=None):
        if linear:
            self.evaluator = LinearSVC()
        else:
            self.evaluator = SVC()
        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        super(SVMEvaluator, self).__init__(self.evaluator, params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='graph', help="graph or node") 
    parser.add_argument('--dataset_name', type=str, default='MUTAG', help="PROTEINS, MUTAG, COLLAB, DD, NCI1, GITHUB, REDDIT-BINARY \
                        REDDIT-MULTI-5K, Cora, CiteSeer, PubMed") 
    parser.add_argument('--epochs', type=int, default=200, metavar='E', help='number of epochs')
    parser.add_argument('--embed_size', type=int, default=32, help='hidden dimensions')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden dimensions')
    parser.add_argument('--num_layers', type=int, default=2, help='number of GCN layers')
    parser.add_argument('--seq_length', type=int, default=7, help='input feature')
    parser.add_argument('--pre_length', type=int, default=7, help='prediction')
    parser.add_argument('--device', type=str, default=1, metavar='D', help='device')
    args = parser.parse_args()
    print(args)
    n_epochs = args.epochs
    device = torch.device('cuda:{}'.format(args.device))
    if args.tasks == "graph":
        dataset = TUDataset(name=args.dataset_name, root="/data2/syt")
    else:
        dataset = Planetoid(root='/data2/syt', name=args.dataset_name)

    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    dataloader = DataLoader(dataset, batch_size=128)
    input_dim = max(dataset.num_features, 1)

    aug1 = Identity()
    aug2 = RandomChoice([NodeDropping(pn=0.1),
                         EdgeRemoving(pe=0.1)], 2)

    # gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
    gconv = FGNConv(args=args, pre_length=args.pre_length, input_dim=input_dim, embed_size=args.embed_size, hidden_dim=args.hidden_size, \
                    seq_length=args.seq_length, num_layers=args.num_layers).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.01)
    for epoch in range(n_epochs):
        encoder_model.train()
        epoch_loss = 0
        correct = 0
        for data in dataloader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
            g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
            loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            epoch_loss += loss.item()

        print("loss:", epoch_loss)

        encoder_model.eval()
        x = []
        y = []
        for data in dataloader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
            x.append(g)
            y.append(data.y)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
        result = SVMEvaluator(linear=True)(x, y, split)
        print(f'(E): Best test F1Mi={result["micro_f1"]:.4f}, F1Ma={result["macro_f1"]:.4f}, Acc={result["acc"]:.4f}')





