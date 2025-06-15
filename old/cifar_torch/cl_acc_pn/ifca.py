import os
import time
import pickle
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

import numpy as np

from util import *

LR_DECAY = True

class PerImageStandardize(object):
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-7)

class TrainCIFARCluster(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device

        assert self.config['m'] % self.config['p'] == 0

    def setup(self):

        os.makedirs(self.config['project_dir'], exist_ok = True)

        self.result_fname = os.path.join(self.config['project_dir'], 'results.pickle')
        self.checkpoint_fname = os.path.join(self.config['project_dir'], 'checkpoint.pt')

        self.setup_datasets()
        self.setup_models()

        self.epoch = None
        self.lr = None
        #self.cluster_switch = None


    def setup_datasets(self):

        np.random.seed(self.config['data_seed'])

        # generate indices for each dataset
        # also write cluster info

        CIFAR_TRAINSET_DATA_SIZE = 50000
        CIFAR_TESTSET_DATA_SIZE = 10000

        np.random.seed(self.config['data_seed'])

        cfg = self.config

        self.dataset = {}

        if cfg['uneven'] == True:
            dataset = {}
            dataset['data_indices'], dataset['cluster_assign'] = \
                self._setup_dataset_random_n(CIFAR_TRAINSET_DATA_SIZE, cfg['p'], cfg['m'], cfg['n'])
            (X, y) = self._load_CIFAR(train=True)
            dataset['X'] = X
            dataset['y'] = y
            self.dataset['train'] = dataset

            dataset = {}
            dataset['data_indices'], dataset['cluster_assign'] = \
                self._setup_dataset_random_n(CIFAR_TESTSET_DATA_SIZE, cfg['p'], cfg['m_test'], cfg['n'], random=True)
            (X, y) = self._load_CIFAR(train=False)
            dataset['X'] = X
            dataset['y'] = y
            self.dataset['test'] = dataset

        else:
            dataset = {}
            dataset['data_indices'], dataset['cluster_assign'] = \
                self._setup_dataset(CIFAR_TRAINSET_DATA_SIZE, cfg['p'], cfg['m'], cfg['n'])
            (X, y) = self._load_CIFAR(train=True)
            dataset['X'] = X
            dataset['y'] = y
            self.dataset['train'] = dataset

            dataset = {}
            dataset['data_indices'], dataset['cluster_assign'] = \
                self._setup_dataset(CIFAR_TESTSET_DATA_SIZE, cfg['p'], cfg['m_test'], cfg['n'], random=True)
            (X, y) = self._load_CIFAR(train=False)
            dataset['X'] = X
            dataset['y'] = y
            self.dataset['test'] = dataset

        # import ipdb; ipdb.set_trace()


    def _setup_dataset(self, num_data, p, m, n, random = True):

        # print("m:",m)
        # print("p:",p)
        # print("n:",n)
        print("num_data:",num_data)
        assert (m // p) * n == num_data

        dataset = {}

        cfg = self.config

        data_indices = []
        cluster_assign = []

        m_per_cluster = m // p

        for p_i in range(p):

            if random:
                ll = list(np.random.permutation(num_data))
            else:
                ll = list(range(num_data))

            ll2 = chunkify(ll, m_per_cluster) # splits ll into m lists with size n
            data_indices += ll2

            cluster_assign += [p_i for _ in range(m_per_cluster)]

        data_indices = np.array(data_indices)
        cluster_assign = np.array(cluster_assign)
        assert data_indices.shape[0] == cluster_assign.shape[0]
        assert data_indices.shape[0] == m


        return data_indices, cluster_assign


    def _setup_dataset_random_n(self, num_data, p, m, n, random = True):

        # print("m:",m)
        # print("p:",p)
        # print("num_data:",num_data)

        dataset = {}

        cfg = self.config

        data_indices = []
        cluster_assign = []

        m_per_cluster = m // p

        for p_i in range(p):

            ll = list(np.random.permutation(num_data))

            ll2 = chunkify_uneven(ll, m_per_cluster) # splits ll into m lists
            data_indices += ll2

            cluster_assign += [p_i for _ in range(m_per_cluster)]

        data_indices = np.array(data_indices, dtype=object)
        cluster_assign = np.array(cluster_assign)
        assert data_indices.shape[0] == cluster_assign.shape[0]
        assert data_indices.shape[0] == m


        return data_indices, cluster_assign


    def _load_CIFAR(self, train=True):
        # transform = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize((24, 24)),
        #     torchvision.transforms.ToTensor()
        # ])

        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=None)

        X = [img for img, _ in dataset]  # img is a PIL Image
        y = [label for _, label in dataset]
        y = torch.tensor(y)

        # dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        # X, y = next(iter(dl))  # This applies the transform pipeline

        # X = X.to(self.device)
        # y = y.to(self.device)

        # print(X.shape)  # [50000, 3, 24, 24] or [10000, 3, 24, 24]

        self.train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(24),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(brightness=63/255, contrast=(0.2, 1.8)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                PerImageStandardize()
            ])
        
        self.test_transform = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(24),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                PerImageStandardize()
            ])

        return X, y


    # Need p models for each client

    def setup_models(self):
        np.random.seed(self.config['train_seed'])
        # torch.manual_seed(self.config['train_seed'])

        p = self.config['p']
        m = self.config['m']
        local_model_init = self.config['local_model_init']
        lr = self.config['lr']

        self.models = [SimpleCNN().to(self.device) for p_i in range(p)]  # Create p models

        self.criterion = torch.nn.CrossEntropyLoss()
        self.cluster_assign = []

        # import ipdb; ipdb.set_trace()


    def run(self):
        num_epochs = self.config['num_epochs']
        lr = self.config['lr']

        #self.cluster_switch = [[0 for _ in range(self.config['p'])] for m_i in range(self.config['m'])] 

        results = []

        # epoch -1
        self.epoch = -1

        self.find_good_initializer()

        result = {}
        result['epoch'] = -1

        t0 = time.time()
        self.set_participating_nodes()
        res = self.test(train=True)
        t1 = time.time()
        res['infer_time'] = t1-t0
        result['train'] = res

        self.print_epoch_stats(res)

        t0 = time.time()
        res = self.test(train=False)
        t1 = time.time()
        res['infer_time'] = t1-t0
        result['test'] = res
        self.print_epoch_stats(res)
        results.append(result)


        for epoch in range(num_epochs):
            self.epoch = epoch

            result = {}
            result['epoch'] = epoch

            lr = self.lr_schedule(epoch)
            result['lr'] = lr

            t0 = time.time()
            result['train'] = self.train(lr = lr)
            t1 = time.time()
            train_time = t1-t0

            t0 = time.time()
            self.set_participating_nodes()
            res = self.test(train=True)
            t1 = time.time()
            res['infer_time'] = t1-t0
            res['train_time'] = train_time
            res['lr'] = lr
            result['train'] = res

            self.print_epoch_stats(res)

            t0 = time.time()
            res = self.test(train=False)
            t1 = time.time()
            res['infer_time'] = t1-t0
            result['test'] = res
            self.print_epoch_stats(res)

            results.append(result)

            if LR_DECAY:
                self.lr = self.lr * 0.9999

            # this will be used in next epoch's gradient update

            if epoch % 10 == 0 or epoch == num_epochs - 1 :
                with open(self.result_fname, 'wb') as outfile:
                    pickle.dump(results, outfile)
                    print(f'result written at {self.result_fname}')
#                self.save_checkpoint()
                print(f'checkpoint written at {self.checkpoint_fname}')


        # plt.figure(figsize=(10,5))
        # plt.plot([r['train']['loss'] for r in results], label='train')
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.title('Training Loss per Epoch')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(os.path.join(self.config['project_dir'], 'train_loss.png'))
        # # import ipdb; ipdb.set_trace()

        # plt.figure(figsize=(10,5))
        # plt.plot([r['test']['acc'] for r in results], label='test')
        # plt.xlabel('epoch')
        # plt.ylabel('test accuracy')
        # plt.title('Test Accuracy per Epoch')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(os.path.join(self.config['project_dir'], 'test_acc.png'))

        # plt.figure(figsize=(10,5))
        # plt.plot([r['train']['cl_acc'] for r in results], label='train')
        # plt.xlabel('epoch')
        # plt.ylabel('cluster acc')
        # plt.title('Cluster Accuracy per Epoch')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(os.path.join(self.config['project_dir'], 'cluster_acc.png'))

        return results


    def find_good_initializer(self):
        print("finding good initializer from train data")

        cfg = self.config

        if cfg['p'] == 4:
            th = 0.1
        elif cfg['p'] == 2:
            th = 0.35
        elif cfg['p'] == 1:
            th = 0.0
        else:
            raise NotImplementedError("only p=1,2,4 supported")

        is_not_good = True
        while is_not_good:
            self.setup_models()
            t0 = time.time()
            self.set_participating_nodes()
            # res = self.test(train=True, force_full_nodes = True)
            res = self.test(train=True)
            t1 = time.time()
            res['infer_time'] = t1-t0
            self.print_epoch_stats(res)

            cl_ct = res['cl_ct']

            num_nodes = np.sum(cl_ct)
            is_not_good = False
            for ct in cl_ct:
                if ct / num_nodes < th:
                    is_not_good = True

        print("found good initializer")

    def set_participating_nodes(self):
        cfg = self.config
        m = cfg['m']
        p = cfg['p']
        p_rate = cfg['participation_rate']

        self.participating_nodes = np.random.choice(m, int(m * p_rate), replace = False)
        # self.participating_nodes = range(m)

        return self.participating_nodes


    def lr_schedule(self, epoch):
        if self.lr is None:
            self.lr = self.config['lr']

        if epoch == 350 and LR_DECAY:
            self.lr = self.lr * 0.1

        return self.lr        


    def print_epoch_stats(self, res):
        if res['is_train']:
            data_str = 'tr'
        else:
            data_str = 'tst'

        if 'train_time' in res:
            time_str = f"{res['train_time']:.3f}sec(train) {res['infer_time']:.3f}sec(infer)"
        else:
            time_str = f"{res['infer_time']:.3f}sec"

        if 'lr' in res:
            lr_str = f" lr {res['lr']:4f}"
        else:
            lr_str = ""

        str0 = f"Epoch {self.epoch} {data_str}: l {res['loss']:.3f} a {res['acc']:.3f} clct{res['cl_ct']} clct_ans{res['cl_ct_ans']} {lr_str} {time_str}"

        print(str0)

    def train(self, lr):
        VERBOSE = 0

        cfg = self.config
        m = cfg['m']
        p = cfg['p']
        tau = cfg['tau']
        batch_size = cfg['batch_size']
        cluster_assign = self.cluster_assign

        participating_nodes = self.participating_nodes

        # run local update
        t0 = time.time()

        updated_models = []

        for m_i in participating_nodes:
            if VERBOSE and m_i % 100 == 0: print(f'm {m_i}/{m} processing \r', end ='')

            (X, y) = self.load_data(m_i)

            ds = TensorDataset(X, y)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

            p_i = cluster_assign[m_i]
            model = copy.deepcopy(self.models[p_i])

            optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.004)

            for step_i in range(tau):
                for x, y in dl:
                    y_logit = model(x)
                    loss = self.criterion(y_logit, y)

                    optim.zero_grad()
                    loss.backward()
                    # self.local_param_update(model, lr)
                    optim.step()

            optim.zero_grad()

            updated_models.append(model)


        t02 = time.time()
        # print(f'running single ..took {t02-t01:.3f}sec')


        t1 = time.time()
        if VERBOSE: print(f'local update {t1-t0:.3f}sec')

        # apply gradient update
        t0 = time.time()

        # CLUSTER MACHINES INTO p_i's
        local_models = [[] for p_i in range(p)]
        for m_i2, m_i in enumerate(participating_nodes):
            p_i = cluster_assign[m_i]
            local_models[p_i].append(updated_models[m_i2])

        for p_i, models in enumerate(local_models):
            if len(models) > 0:
                self.global_param_update(models, self.models[p_i])
        t1 = time.time()

        t1 = time.time()

        if VERBOSE: print(f'global update {t1-t0:.3f}sec')

    def check_local_model_loss(self, local_models):
        # for debugging
        m = self.config['m']

        losses = []
        for m_i in range(m):
            (X, y) = self.load_data(m_i)
            y_logit = local_models[m_i](X)
            loss = self.criterion(y_logit, y)

            losses.append(loss.item())

        return np.array(losses)


    def get_inference_stats(self, train = True):
        cfg = self.config
        if train:
            m = cfg['m']
            dataset = self.dataset['train']
            participating_nodes = self.participating_nodes
        else:
            m = cfg['m_test']
            dataset = self.dataset['test']
            participating_nodes = list(range(m))

        p = cfg['p']
        p_rate = cfg['participation_rate']


        num_data = 0
        losses = {}
        corrects = {}
        for m_i in participating_nodes:
            (X, y) = self.load_data(m_i, train=train) # load batch data rotated

            for p_i in range(p):
                y_logit = self.models[p_i](X)
                loss = self.criterion(y_logit, y) # loss of
                n_correct = self.n_correct(y_logit, y)

                # if torch.isnan(loss):
                #     print("nan loss: ", dataset['data_indices'][m_i])

                losses[(m_i,p_i)] = loss.item()
                corrects[(m_i,p_i)] = n_correct

            num_data += X.shape[0]

        # calculate loss and cluster the machines
        cluster_assign = [-1 for _ in range(m)]
        for m_i in participating_nodes:
            machine_losses = [ losses[(m_i,p_i)] for p_i in range(p) ]
            #print("Machine Losses:", machine_losses)
            min_p_i = np.argmin(machine_losses)
            cluster_assign[m_i] = min_p_i

        # calculate optimal model's loss, acc over all models
        min_corrects = []
        min_losses = []
        for m_i in participating_nodes:
            p_i = cluster_assign[m_i]

            min_loss = losses[(m_i,p_i)]
            min_losses.append(min_loss)

            min_correct = corrects[(m_i,p_i)]
            min_corrects.append(min_correct)

        # print("losses: ", min_losses)

        if train:
            self.cluster_assign = cluster_assign

        loss = np.mean(min_losses)
        acc = np.sum(min_corrects) / num_data


        # check cluster assignment acc
        # cl_acc = self.get_cluster_accuracy(dataset['cluster_assign'], cluster_assign)
        cl_ct = [np.sum(np.array(cluster_assign) == p_i ) for p_i in range(p)]

        # cl_acc = self.get_cluster_accuracy(cluster_assign, train)

        # improved cluster assignment acc (model 2 can work better on clients with p=3)
        cluster_assign_ans = dataset['cluster_assign']
        cluster_assign_ans_part = np.array(cluster_assign_ans)[participating_nodes]
        cl_ct_ans = [np.sum(np.array(cluster_assign_ans_part) == p_i ) for p_i in range(p)]
        

        res = {} # results
        # res['losses'] = losses
        # res['corrects'] = corrects
        res['cluster_assign'] = cluster_assign
        res['num_data'] = num_data
        res['loss'] = loss
        res['acc'] = acc
        # res['cl_acc'] = cl_acc
        res['cl_ct'] = cl_ct
        res['is_train'] = train
        res['cl_ct_ans'] = cl_ct_ans

        # import ipdb; ipdb.set_trace()

        return res

    def n_correct(self, y_logit, y):
        _, predicted = torch.max(y_logit.data, 1)
        correct = (predicted == y).sum().item()

        return correct

    def load_data(self, m_i, train=True):
        # this part is very fast since its just rearranging models
        cfg = self.config

        if train:
            dataset = self.dataset['train']

        else:
            dataset = self.dataset['test']

        indices = dataset['data_indices'][m_i]
        p_i = dataset['cluster_assign'][m_i]

        X_batch = [dataset['X'][i] for i in indices]  # PIL Images
        y_batch = [dataset['y'][i] for i in indices]

        # k : how many times rotate 90 degree
        # k =1 : 90 , k=2 180, k=3 270

        if cfg['p'] == 4:
            k = p_i
        elif cfg['p'] == 2:
            k = (p_i % 2) * 2
        elif cfg['p'] == 1:
            k = 0
        else:
            raise NotImplementedError("only p=1,2,4 supported")

        if train:
            X_batch2 = [self.train_transform(img) for img in X_batch]  # Now tensors
        else:
            X_batch2 = [self.test_transform(img) for img in X_batch]  # Now tensors

        X_batch2 = torch.stack(X_batch2).to(self.device)
        y_batch = torch.tensor(y_batch).to(self.device)
        
        X_batch2 = torch.rot90(X_batch2, k=int(k), dims = (2,3))

        # import ipdb; ipdb.set_trace()

        return X_batch2, y_batch


    def local_param_update(self, model, lr):

        # gradient update manually

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data -= lr * param.grad

        model.zero_grad()

        # import ipdb; ipdb.set_trace() # we need to check the output of name, check if duplicate exists

    def global_param_update(self, local_models, global_model):

        # average of each weight

        weights = {}

        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in weights:
                    weights[name] = torch.zeros_like(param.data)

                weights[name] += param.data

        for name, param in global_model.named_parameters():
            weights[name] /= len(local_models)
            param.data = weights[name]

        # import ipdb; ipdb.set_trace()



    def test(self, train=False):
        return self.get_inference_stats(train=train)


    def save_checkpoint(self):
        models_to_save = [model.state_dict() for model in self.models]
        torch.save({'models':models_to_save}, self.checkpoint_fname)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.fc1 = nn.Linear(64 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.lrn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.lrn2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)