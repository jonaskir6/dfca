import os
import time
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import copy

import numpy as np

from util import *

LR_DECAY = False


class TrainEMNISTCluster(object):
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

        EMNIST_TRAINSET_DATA_SIZE = 60000
        EMNIST_TESTSET_DATA_SIZE = 10000

        np.random.seed(self.config['data_seed'])

        cfg = self.config

        self.dataset = {}

        if cfg['uneven'] == True:
            dataset = {}
            dataset['data_indices'], dataset['cluster_assign'] = \
                self._setup_dataset_random_n(EMNIST_TRAINSET_DATA_SIZE, cfg['p'], cfg['m'], cfg['n'])
            (X, y) = self._load_EMNIST(train=True)
            dataset['X'] = X
            dataset['y'] = y
            self.dataset['train'] = dataset

            dataset = {}
            dataset['data_indices'], dataset['cluster_assign'] = \
                self._setup_dataset_random_n(EMNIST_TESTSET_DATA_SIZE, cfg['p'], cfg['m_test'], cfg['n'], random=True)
            (X, y) = self._load_EMNIST(train=False)
            dataset['X'] = X
            dataset['y'] = y
            self.dataset['test'] = dataset

        else:
            dataset = {}
            dataset['data_indices'], dataset['cluster_assign'] = \
                self._setup_dataset(EMNIST_TRAINSET_DATA_SIZE, cfg['p'], cfg['m'], cfg['n'])
            (X, y) = self._load_EMNIST(train=True)
            dataset['X'] = X
            dataset['y'] = y
            self.dataset['train'] = dataset

            dataset = {}
            dataset['data_indices'], dataset['cluster_assign'] = \
                self._setup_dataset(EMNIST_TESTSET_DATA_SIZE, cfg['p'], cfg['m_test'], cfg['n'], random=True)
            (X, y) = self._load_EMNIST(train=False)
            dataset['X'] = X
            dataset['y'] = y
            self.dataset['test'] = dataset

        # import ipdb; ipdb.set_trace()


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


    def _load_EMNIST(self, train=True):
        transforms = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               # torchvision.transforms.Normalize(
                               #   (0.1307,), (0.3081,))
                             ])
        if train:
            emnist_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transforms)
        else:
            emnist_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transforms)

        dl = DataLoader(emnist_dataset)

        X = dl.dataset.data.unsqueeze(1)  # Add channel dimension for CNN
        y = dl.dataset.targets #(60000)

        # normalize to have 0 ~ 1 range in each pixel

        X = X / 255.0
        X = X.to(self.device)
        y = y.to(self.device)

        return X, y


    # Need p models for each client

    def setup_models(self):
        np.random.seed(self.config['train_seed'])
        torch.manual_seed(self.config['train_seed'])

        p = self.config['p']
        m = self.config['m']
        local_model_init = self.config['local_model_init']

        if local_model_init:
            self.models = [[SimpleCNN(h1 = self.config['h1']).to(self.device) for p_i in range(p)] for m_i in range(m)]

        else:
            global_models = [SimpleCNN(h1 = self.config['h1']).to(self.device) for p_i in range(p)]  # Create p models
            self.models = [[copy.deepcopy(model) for model in global_models] for m_i in range(m)]  # Each client gets the same list of p models

        self.criterion = torch.nn.CrossEntropyLoss()

        # import ipdb; ipdb.set_trace()


    def run(self):
        num_epochs = self.config['num_epochs']
        lr = self.config['lr']

        #self.cluster_switch = [[0 for _ in range(self.config['p'])] for m_i in range(self.config['m'])] 

        results = []

        # epoch -1
        self.epoch = -1

        result = {}
        result['epoch'] = -1

        t0 = time.time()
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

        # this will be used in next epoch
        cluster_assign = result['train']['cluster_assign']

        for epoch in range(num_epochs):
            self.epoch = epoch

            result = {}
            result['epoch'] = epoch

            lr = self.lr_schedule(epoch)
            result['lr'] = lr

            t0 = time.time()
            result['train'] = self.train(cluster_assign, lr = lr)
            t1 = time.time()
            train_time = t1-t0

            t0 = time.time()
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

            # this will be used in next epoch's gradient update
            cluster_assign = result['train']['cluster_assign']

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





    def lr_schedule(self, epoch):
        if self.lr is None:
            self.lr = self.config['lr']

        if epoch % 50 == 0 and epoch != 0 and LR_DECAY:
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

        str0 = f"Epoch {self.epoch} {data_str}: l {res['loss']:.3f} a {res['acc']:.3f} clct{res['cl_ct']} cl_acc {res['cl_acc']:.3f} {lr_str} {time_str}"

        print(str0)

    def train(self, cluster_assign, lr):
        VERBOSE = 0

        cfg = self.config
        m = cfg['m']
        p = cfg['p']
        tau = cfg['tau']

        # run local update
        t0 = time.time()


        for m_i in range(m):
            if VERBOSE and m_i % 100 == 0: print(f'm {m_i}/{m} processing \r', end ='')

            (X, y) = self.load_data(m_i)

            p_i = cluster_assign[m_i]
            model = self.models[m_i][p_i]

            # LOCAL UPDATE PER MACHINE tau times
            for step_i in range(tau):

                y_logit = model(X)
                loss = self.criterion(y_logit, y)

                model.zero_grad()
                loss.backward()
                self.local_param_update(model, lr)

            model.zero_grad()


        t02 = time.time()
        # print(f'running single ..took {t02-t01:.3f}sec')


        t1 = time.time()
        if VERBOSE: print(f'local update {t1-t0:.3f}sec')

        # apply gradient update
        t0 = time.time()

        # NEEDS TO BE DECENTRALIZED
        self.dec_param_update(cluster_assign)
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
    
    def get_cluster_accuracy(self, actual, pred):
        
        cm = confusion_matrix(actual, pred)

        row_ind, col_ind = linear_sum_assignment(-cm)
        matching = dict(zip(col_ind, row_ind))

        remapped_preds = [matching[p] for p in pred]

        cl_acc = np.mean(np.array(remapped_preds) == np.array(actual))

        return cl_acc


    def get_inference_stats(self, train = True):
        cfg = self.config
        if train:
            m = cfg['m']
            dataset = self.dataset['train']
        else:
            m = cfg['m_test']
            dataset = self.dataset['test']

        p = cfg['p']


        num_data = 0
        losses = {}
        corrects = {}
        for m_i in range(m):
            (X, y) = self.load_data(m_i, train=train) # load batch data rotated

            for p_i in range(p):
                y_logit = self.models[m_i][p_i](X)
                loss = self.criterion(y_logit, y) # loss of
                n_correct = self.n_correct(y_logit, y)

                # if torch.isnan(loss):
                #     print("nan loss: ", dataset['data_indices'][m_i])

                losses[(m_i,p_i)] = loss.item()
                corrects[(m_i,p_i)] = n_correct

            num_data += X.shape[0]

        # calculate loss and cluster the machines
        cluster_assign = []
        for m_i in range(m):
            machine_losses = [ losses[(m_i,p_i)] for p_i in range(p) ]
            #print("Machine Losses:", machine_losses)
            min_p_i = np.argmin(machine_losses)
            cluster_assign.append(min_p_i)

        # calculate optimal model's loss, acc over all models
        min_corrects = []
        min_losses = []
        for m_i, p_i in enumerate(cluster_assign):

            min_loss = losses[(m_i,p_i)]
            min_losses.append(min_loss)

            min_correct = corrects[(m_i,p_i)]
            min_corrects.append(min_correct)

        # print("losses: ", min_losses)

        if train:
            loss = np.mean(min_losses)
            acc = np.sum(min_corrects) / num_data

        else:
            loss, acc = self.test_all()


        # check cluster assignment acc
        cl_acc = self.get_cluster_accuracy(dataset['cluster_assign'], cluster_assign)
        cl_ct = [np.sum(np.array(cluster_assign) == p_i ) for p_i in range(p)]

        # improved cluster assignment acc (model 2 can work better on clients with p=3)
        

        res = {} # results
        # res['losses'] = losses
        # res['corrects'] = corrects
        res['cluster_assign'] = cluster_assign
        res['num_data'] = num_data
        res['loss'] = loss
        res['acc'] = acc
        res['cl_acc'] = cl_acc
        res['cl_ct'] = cl_ct
        res['is_train'] = train

        # import ipdb; ipdb.set_trace()

        return res

    def n_correct(self, y_logit, y):
        _, predicted = torch.max(y_logit.data, 1)
        correct = (predicted == y).sum().item()

        return correct

    # TODO Does every Cluster get 4 clients with the same data, but rotated differently?

    def load_data(self, m_i, train=True):
        # this part is very fast since its just rearranging models
        cfg = self.config

        if train:
            dataset = self.dataset['train']
        else:
            dataset = self.dataset['test']

        indices = dataset['data_indices'][m_i]
        p_i = dataset['cluster_assign'][m_i]

        X_batch = dataset['X'][indices]
        y_batch = dataset['y'][indices]

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

        X_batch2 = torch.rot90(X_batch, k=int(k), dims = (2,3))

        # import ipdb; ipdb.set_trace()

        return X_batch2, y_batch


    def local_param_update(self, model, lr):

        # gradient update manually

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data -= lr * param.grad

        model.zero_grad()

        # import ipdb; ipdb.set_trace() # we need to check the output of name, check if duplicate exists


    def dec_param_update(self, cluster_assign):

        num_clients = self.config['m']

        if num_clients <= 1:
            return

        client_indices = list(range(num_clients)) 

        for m_i in range(num_clients):

            counts = {i: 0 for i in range(self.config['p'])}
            for value in cluster_assign:
                counts[value] += 1

            num_cluster_i = counts[cluster_assign[m_i]]
            num_cluster_rest = num_clients - num_cluster_i

            threshold_j = min(num_cluster_rest, 100)
            threshold_i = min(num_cluster_i, 100)

            # threshold_j = min(num_cluster_rest, int(np.floor(e/2)))
            # threshold_i = min(num_cluster_i, int(np.floor(e/2))) - 1

            if threshold_i <=1 or threshold_j<= 1:
                continue

            selected_clients = random.sample([i for i in client_indices if i != m_i], torch.randint(1, min(threshold_i,threshold_j), (1,)))
            # selected_clients += random.sample([i for i in client_indices if i != m_i and cluster_assign[m_i] == cluster_assign[i]], threshold_i)
            m_i_cluster = cluster_assign[m_i]
            for m_j in selected_clients:
                m_j_cluster = cluster_assign[m_j]

                m_j_params = dict(self.models[m_j][m_j_cluster].named_parameters())

                if m_i_cluster == m_j_cluster:
                    for name, param in self.models[m_i][m_i_cluster].named_parameters():
                        m_i_param = param.data.clone()
                        m_j_param = m_j_params[name].data.clone()
                        alpha = 0.5
                        param.data = (m_i_param + m_j_param) / 2     

                else:
                    for name, param in self.models[m_i][m_j_cluster].named_parameters():
                        m_i_param = param.data.clone()
                        m_j_param = m_j_params[name].data.clone()
                        alpha = 0.5
                        param.data = alpha * m_i_param + (1 - alpha) * m_j_param

        # import ipdb; ipdb.set_trace()


    def test(self, train=False):
        return self.get_inference_stats(train=train)

    def load_test_data(self, m_i, train=False):
        cfg = self.config

        p = cfg['p']

        if train:
            dataset = self.dataset['train']
        else:
            dataset = self.dataset['test']

        indices = dataset['data_indices'][m_i]
        p_i = dataset['cluster_assign'][m_i]

        X_batch = dataset['X'][indices]
        y_batch = dataset['y'][indices]

        data = []

        if p == 4:
            rotation_list = [0, 1, 2, 3] 
        elif p == 2:
            rotation_list = [0, 2]        
        elif p == 1:
            rotation_list = [0]          
        else:
            raise NotImplementedError("Only p=1,2,4 supported")

        for k in rotation_list:
            X_batch2 = torch.rot90(X_batch, k=int(k), dims=(2, 3))
            data.append(X_batch2)

        return data, y_batch

    def test_all(self, train=False):
        cfg = self.config
        m = cfg['m_test']
        dataset = self.dataset['test']

        p = cfg['p']

        num_data = 0
        losses = []
        corrects = []
        for m_i in range(m):
            
            (data, y) = self.load_test_data(m_i, train=train)

            for p_i in range(p):
                X = data[p_i]
                loss_m_i = []
                correct_m_i = []
                for model in range(p):
                    y_logit = self.models[m_i][model](X)
                    loss_m_i.append(self.criterion(y_logit, y))
                    correct_m_i.append(self.n_correct(y_logit, y))

                loss = np.min([l.item() for l in loss_m_i])
                n_correct = np.max(correct_m_i)

                # if torch.isnan(loss):
                #     print("nan loss: ", dataset['data_indices'][m_i])

                losses.append(loss)
                corrects.append(n_correct)

                num_data += X.shape[0]

        loss = np.mean(losses)
        acc = np.sum(corrects) / num_data

        # print(f"Average loss over all clients and models: {loss:.3f}")
        # print(f"Average accuracy over all clients and models: {acc:.3f}")    

        return loss, acc



    def save_checkpoint(self):
        models_to_save = [model.state_dict() for model in self.models]
        torch.save({'models':models_to_save}, self.checkpoint_fname)

class SimpleCNN(nn.Module):
    def __init__(self, h1=128):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, h1)
        self.fc2 = nn.Linear(h1, 47)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)