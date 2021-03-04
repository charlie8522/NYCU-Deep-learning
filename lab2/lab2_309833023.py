# -*- coding: utf-8 -*- 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataloader import read_bci_data
import matplotlib
matplotlib.use('Agg')

batch_size = 64
lr = 0.001
epochs = 3000
print_interval = 750
activation_type = {'Relu': nn.ReLU(),'LeakyRelu': nn.LeakyReLU(),'ELU': nn.ELU()}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(87)
torch.cuda.manual_seed(123)

class BCIDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
        self.label = torch.from_numpy(label).type(torch.LongTensor).to(device)
        return
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

train_data, train_label, test_data, test_label = read_bci_data()

train_dataset = BCIDataset(train_data, train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = BCIDataset(test_data, test_label)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)


class DeepConvNet(nn.Module):
    def __init__(self, activation_func):
        super(DeepConvNet, self).__init__()
        Conv1 = nn.Conv2d(1, 25, kernel_size = (1,5), stride = (1,1), bias = False)
        self.Model1 = nn.Sequential(Conv1)

        Conv2 = nn.Conv2d(25, 25, kernel_size = (2,1), stride = (1,1), bias = False)
        Batchnorm2 = nn.BatchNorm2d(25, eps = 1e-05, momentum = 0.1)   
        Maxpool2 = nn.MaxPool2d(kernel_size = (1,2), stride = (1,2), padding = 0)
        Dropout2 = nn.Dropout(p = 0.5)
        self.Model2 = nn.Sequential(Conv2, Batchnorm2, activation_type[activation_func], Maxpool2, Dropout2)
        
        Conv3 = nn.Conv2d(25, 50, kernel_size = (1,5), stride = (1,1), bias = False)#
        Batchnorm3 = nn.BatchNorm2d(50, eps = 1e-05, momentum = 0.1)
        Maxpool3 = nn.MaxPool2d(kernel_size = (1,2), stride = (1,2), padding = 0)
        Dropout3 = nn.Dropout(p = 0.5)
        self.Model3 = nn.Sequential(Conv3, Batchnorm3, activation_type[activation_func], Maxpool3, Dropout3)
        
        Conv4 = nn.Conv2d(50, 100, kernel_size = (1,5))
        Batchnorm4 = nn.BatchNorm2d(100, eps = 1e-05, momentum = 0.1)    
        Maxpool4 = nn.MaxPool2d(kernel_size = (1,2), stride = (1,2), padding = 0)
        Dropout4 = nn.Dropout(p = 0.5)
        self.Model4 = nn.Sequential(Conv4, Batchnorm4, activation_type[activation_func], Maxpool4, Dropout4)
        
        Conv5 = nn.Conv2d(100, 200, kernel_size = (1,5))
        Batchnorm5 = nn.BatchNorm2d(200, eps = 1e-05, momentum = 0.1)    
        Maxpool5 = nn.MaxPool2d(kernel_size = (1,2), stride = (1,2), padding = 0)
        Dropout5 = nn.Dropout(p = 0.5)
        self.Model5 = nn.Sequential(Conv5, Batchnorm5, activation_type[activation_func], Maxpool5, Dropout5)

        self.fc1 = nn.Linear(in_features = 8600, out_features = 2, bias = True)
        
    def forward(self,out):
        out = self.Model1(out)
        out = self.Model2(out)
        out = self.Model3(out)
        out = self.Model4(out)
        out = self.Model5(out)
        out = out.view(-1,8600)
        out = self.fc1(out)
        return out

        
class EEGNet(nn.Module):
    def __init__(self, activation_func):
        super(EEGNet, self).__init__()

        Conv1 = nn.Conv2d(1, 16, kernel_size = (1, 51), stride = (1, 1), padding = (0, 25), bias=False)
        Batchnorm1 = nn.BatchNorm2d(16, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.Model1 = nn.Sequential(Conv1, Batchnorm1)
        
        Conv2 = nn.Conv2d(16, 32, kernel_size = (2, 1), stride = (1, 1), groups = 16, bias = False)
        Batchnorm2 = nn.BatchNorm2d(32, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        Avgpool2 = nn.AvgPool2d(kernel_size = (1, 4), stride = (1, 4), padding = 0)
        Dropout2 = nn.Dropout(p = 0.5)
        self.Model2 = nn.Sequential(Conv2, Batchnorm2, activation_type[activation_func], Avgpool2, Dropout2)
        
        Conv3 = nn.Conv2d(32, 32, kernel_size = (1, 15), stride = (1, 1),  padding = (0, 7), bias = False)
        Batchnorm3 = nn.BatchNorm2d(32, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        Avgpool3 = nn.AvgPool2d(kernel_size = (1, 8), stride = (1, 8), padding = 0)
        Dropout3 = nn.Dropout(p = 0.5)
        self.Model3 = nn.Sequential(Conv3, Batchnorm3, activation_type[activation_func], Avgpool3, Dropout3)
       
        self.fc1 = nn.Linear(in_features = 736, out_features = 2, bias = True)

    def forward(self, out):
        out = self.Model1(out)
        out = self.Model2(out)
        out = self.Model3(out)
        out = out.view(-1, 736)
        out = self.fc1(out)
        return out
    
def train(model, optimizer, loss_func, scheduler=None):
    acc_train, acc_test = list(), list()
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        
        for i, batch_data in enumerate(train_loader):
            data, label = batch_data
            optimizer.zero_grad()
            
            predict = model(data)
            loss = loss_func(predict, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if scheduler is not None: 
            scheduler.step()
        
        model.eval()
        acc_train.append(test(model, train_loader))
        acc_test.append(test(model, test_loader))
        
        if epoch % print_interval == 0:
            print(f'-------------------------[Epoch {epoch}]-------------------------')
            print(f'loss: {epoch_loss}')
            print(f'Training Acc: {acc_train[-1]}')
            print(f'Testing Acc: {acc_test[-1]}\n')   
   
    return acc_train, acc_test


def test(model, test_loader):
    total, correct = 0, 0
    with torch.no_grad():
        for test_data in test_loader:
            data, label = test_data
            predict = model.forward(data)
            _, predict = torch.max(predict, dim=1)
            total += label.size(0)
            correct += (predict == label).sum().item()
    
    acc = correct / total * 100
    
    return acc
 

def train_entry(model_name, activation):
    if model_name == 'EEG':
        print(f'Now Running with "{activation}" Activation!')
        model = EEGNet(activation).to(device)
    else:
        print(f'Now Running with "{activation}" Activation!')
        model = DeepConvNet(activation).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    # Decaying lr with a factor 0.95 every 25 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)

    acc_train, acc_test = train(model, optimizer, loss_func, scheduler)
    print('Final Acc:', acc_test[-1])
        
    return acc_train, acc_test


'''def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output=model(data)
       
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t loss{:.6f}'.format(
                epoch, batch_idx* len(data), len(train_loader.dataset),
                100. *batch_idx /   len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test-loader:
        if args[ 'cuda' ]:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatiles=True), Variable(target)
        output = model(data)
        test_loss += F.nll-loss(output, target, size_average= False).item() 
        pred = output.data.max( 1, keepdim=True )[1] 
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= (test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test-loss, correct, len(test-loader.dataset),
        100. *correct / len(test-loader.dataset)))'''



def plot_comparison(model_name):
    plt.figure(figsize=(12, 8), dpi=300)
    plt.title(f'Activation Function Comparison of {model_name} Net')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
        
    for activation_func in activation_type:
        if model_name == 'EEG':
            acc_train, acc_test = train_entry('EEG', activation_func)
        else:
            acc_train, acc_test = train_entry('DCN', activation_func)
    
        plt.plot(acc_train, label=f'{activation_func}-Train', linewidth=0.8)
        plt.plot(acc_test, label=f'{activation_func}-Test', linewidth=0.8)
    plt.legend()
    plt.savefig(f'{model_name}NET_Comparison_Graph.jpg')
    plt.show()
plot_comparison('EEG')
plot_comparison('DCN')

