import numpy as np


import torch
import torch.nn as nn
import numpy as np

import random
dtype = torch.cuda.FloatTensor

from torch_geometric.data import DataLoader


from mpnn import utils
import sklearn as sk

#fix random seeds
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)


class DebugSession:
    def __init__(self, model_type, model_class_ls, data_set, zero_data_set, 
                 device, target_mean_test=False, choose_model=True, LR=.001, BS=124, CHOOSE_MODEL_EPOCHS=1000):
        self.model_class_ls = model_class_ls
        self.model_type = model_type #should be 'gnn' for graph neural network or 'mlp' for multi-layer perceptron
        self.data_set = data_set
        self.zero_data_set = zero_data_set
        self.device = device
        self.target_mean_test = target_mean_test
        if self.target_mean_test is False:
            print('Skipping test_target_mean', flush=True)
        self.choose_model = choose_model
        if self.choose_model is False:
            print('Skipping chart_dependencies', flush=True)
        self.LR = LR
        self.BS = BS
        self.CHOOSE_MODEL_EPOCHS = CHOOSE_MODEL_EPOCHS

    def test_target_mean(self, model, target_mean):
        model.train()
        model.init_bias(target_mean)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
        train_loader = DataLoader(self.data_set['train'], batch_size=10, shuffle=True)
        model.train()
        for data in train_loader: #loop through training batches
            pass
        self.data = data.to(self.device)
        optimizer.zero_grad()
        self.output = model(self.data).view(self.data.num_graphs,)
        if self.target_mean_test:
            print('\nChecking that all outputs are near to the mean', flush=True)
            assert (np.max(np.abs(target_mean - self.output.detach().cpu().numpy())) 
                                  / target_mean) < .1 #the absolute deviation from the mean should be <.1
            print('Verified that all outputs are near to the mean\n', flush=True)
        loss_fn = nn.MSELoss()
        loss = loss_fn(self.output, data.y)
        loss.backward()

    def test_output_shape(self):
        assert self.output.shape == self.data.y.shape
        print('\nVerified that shape of model predictions is equal to shape of labels\n', flush=True)

    def grad_check(self, model, file_name):
        try:
            utils.plot_grad_flow(model.named_parameters(),filename=file_name)
        except:
            print('Error: None-type gradients detected', flush=True)
            raise
    
    def test_input_independent_baseline(self, model):
        print('\nChecking input-independent baseline', flush=True)
        train_loader = DataLoader(self.data_set['train'], batch_size=self.BS, shuffle=True)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
        for epoch in range(5):
            real_data_loss = 0
            for data in train_loader: #loop through training batches
                data = data.to(self.device)
                optimizer.zero_grad()
                output = model(data).view(data.num_graphs,)
                loss_fn = nn.MSELoss()
                loss = loss_fn(output, data.y)
                real_data_loss += loss.detach().cpu().numpy()
                loss.backward()
                optimizer.step()
        print('..real_data_loss', real_data_loss, flush=True) #loss for all points in 5th epoch gets printed

        zero_loader = DataLoader(self.zero_data_set, batch_size=self.BS, shuffle=True)
        for epoch in range(5):
            zero_data_loss = 0
            for data in zero_loader: #loop through training batches
                data = data.to(self.device)
                optimizer.zero_grad()
                output = model(data).view(data.num_graphs,)
                loss_fn = nn.MSELoss()
                loss = loss_fn(output, data.y)
                zero_data_loss += loss.detach().cpu().numpy()
                loss.backward()
                optimizer.step()
        print('..zero_data_loss', zero_data_loss, flush=True) #loss for all points in 5th epoch gets printed
        if zero_data_loss < real_data_loss:
            raise ValueError('The loss of zeroed inputs is less than the loss of \
                    real input. This may indicate that your model is not learning anything.')
        print('Input-independent baseline is verified\n', flush=True)
    
    def test_overfit_small_batch(self, model):
        print('\nChecking if a small batch can be overfit', flush=True)
        epsilon = .01 #the loss below which we consider a batch fully-learned
        train_loader = DataLoader(self.data_set['train'][0:5], batch_size=5, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
        model.train()
        overfit = False
        for epoch in range(200):
            if not overfit:
                for data in train_loader: #loop through training batches
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    output = model(data).view(data.num_graphs,)
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(output, data.y)
                    loss.backward()
                    optimizer.step()
                    if np.sqrt(loss.item()) < epsilon:
                        print('..Loss:', np.sqrt(loss.item()))
                        print('..Outputs', output.detach().cpu().numpy().round(4))
                        print('..Labels', data.y.detach().cpu().numpy().round(4), flush=True)
                        overfit = True
        if not overfit:
            raise ValueError('Error: Your model was not able to overfit a small batch of data')
        print('Verified that a small batch can be overfit\n', flush=True)

    def visualize_large_batch_training(self, model):
        print('\nStarting visualization', flush=True)
        train_loader = DataLoader(self.data_set['train'], batch_size=124, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
        model.train()
        min_loss = np.inf
        n_epochs = self.CHOOSE_MODEL_EPOCHS // 2
        for epoch in range(n_epochs):
            for ind,data in enumerate(train_loader): #loop through training batches
                data = data.to(self.device)
                optimizer.zero_grad()
                output = model(data).view(data.num_graphs,)
                loss_fn = nn.MSELoss()
                loss = loss_fn(output, data.y)
                loss.backward()
                optimizer.step()
                if ind == 0:
                    print('..Epoch %s' %epoch)
                    print('....Output:', output.detach().cpu().numpy()[0:10].round(decimals=3))
                    print('....Labels:', data.y.detach().cpu().numpy()[0:10].round(decimals=3))
                    print('....Loss:', np.sqrt(loss.item()))
                    if loss < min_loss:
                        min_loss = loss
                    print('....Best loss:', np.sqrt(min_loss.item()), flush=True)
        print('Visualization complete \n', flush=True)

    def chart_dependencies(self, model):
        print('\nBeginning to chart dependencies', flush=True)
    
        train_loader = DataLoader(self.data_set['train'][0:5], batch_size=5, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
        model.train()
        i = 0
        for epoch in range(1):
            for data in train_loader: #loop through training batches
                data = data.to(self.device)
                data.x.requires_grad = True
                optimizer.zero_grad()
                output = model(data).view(data.num_graphs,)
                loss = output[0]
                loss.backward()
                print('..Epoch %s' %epoch)
                print('....Output:', output.detach().cpu().numpy()[0:10].round(decimals=3))
                print('....Labels:', data.y.detach().cpu().numpy()[0:10].round(decimals=3))
                print('....Loss:', loss.item(), flush=True)

        if self.model_type is 'gnn':
            start_ind = self.data_set['train'][0].x.shape[0] #the number of nodes in the first connected graph
        elif self.model_type is 'mlp':
            start_ind = 1
        else:
            raise ValueError("Invalid 'model_type' selected")
        if data.x.grad[start_ind:,:].sum().item() != 0:
            raise ValueError('Data is getting passed along the batch dimension.')
        
        print('Finished charting dependencies. Data is not getting passed along the batch dimension.\n', flush=True)

    def choose_model_size_by_overfit(self):
        print('\nBeginning model size search', flush=True)

        N_TRAIN_DATA = len(self.data_set['train'])
        train_loader = DataLoader(self.data_set['train'], batch_size=self.BS, shuffle=False)

        min_best_loss = np.inf
        best_model_class = None #index of best model
        for model_n,model_class in enumerate(self.model_class_ls):
            print('\n..Training model %s \n' %model_n)
            model = model_class()
            try:
                model.init_bias(self.target_mean)
            except:
                pass
            optimizer = torch.optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
            model = model.to(self.device)
            model.train()
            min_loss = np.inf #epoch-wise loss
            max_r2 = 0
            for epoch in range(self.CHOOSE_MODEL_EPOCHS):
                epoch_loss = 0
                y = []
                y_hat = []
                for ind,data in enumerate(train_loader): #loop through training batches
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    output = model(data).view(data.num_graphs,)
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(output, data.y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()*data.num_graphs
                    y += data.y.cpu().numpy().tolist()
                    y_hat += output.detach().cpu().numpy().tolist()
                r2 = sk.metrics.r2_score(y, y_hat)
                print('\n....Epoch %s' %epoch)
                print('......[loss] %s [r2] %s' %(np.sqrt(sk.metrics.mean_squared_error(y, y_hat)), r2))
                print('......Output:', np.array(y_hat[0:10]).round(decimals=3))
                print('......Labels:', np.array(y[0:10]).round(decimals=3))
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    max_r2 = r2

                print('......[best loss] %s [best r2] %s' %(np.sqrt(min_loss/N_TRAIN_DATA), max_r2), flush=True)
            utils.plot_grad_flow(model.named_parameters(),filename='big_model_%s_grad_check.png' %model_n)
            print('..Set of gradients plotted to big_model_%s_grad_check.png' %model_n, flush=True)
            if min_loss > min_best_loss:
                break
            else:
                min_best_loss = min_loss
                best_model_class = model_class
        print('Finished model size search. Best model is %s\n' %best_model_class, flush=True)
        return best_model_class
    
    def main(self):
        
        min_model = self.model_class_ls[0]() #instantiate model
        min_model.to(self.device)

        self.target_mean = np.mean([x.y for x in self.data_set['train']])
        print('\ntarget_mean %s \n' %self.target_mean, flush=True)
        self.test_target_mean(min_model, self.target_mean)
        self.test_output_shape()
        self.grad_check(min_model, file_name='first_grad_check.png')
        print('\nSet of gradients plotted to first_grad_check.png\n', flush=True)

        min_model = self.model_class_ls[0]() #re-instantiate model
        min_model.to(self.device)

        self.test_input_independent_baseline(min_model)

        min_model = self.model_class_ls[0]() #re-instantiate model
        min_model.to(self.device)

        self.test_overfit_small_batch(min_model)

        min_model = self.model_class_ls[0]() #re-instantiate model
        min_model.to(self.device)

        self.visualize_large_batch_training(min_model)
        self.grad_check(min_model, file_name='second_grad_check.png')
        print('\nSet of gradients plotted to second_grad_check.png\n', flush=True)

        min_model = self.model_class_ls[0]() #re-instantiate model
        min_model.to(self.device)

        self.chart_dependencies(min_model)

        if self.choose_model:
            best_model = self.choose_model_size_by_overfit()
        else:
            best_model = None
        print('\nDebug session complete.', flush=True)
        return best_model
