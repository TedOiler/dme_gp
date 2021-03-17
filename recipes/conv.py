import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from ipywidgets import IntProgress
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
from sklearn.decomposition import PCA # Import the PCA module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random

random_state = 10 # Ensure reproducible results



class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        n_feature = int(np.sqrt(X_data.shape[1]))
        self.X_data = torch.tensor(X_data).view(-1,n_feature,n_feature).unsqueeze(dim=1).float()
        self.y_data = torch.tensor(y_data).long()

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class Autoencoder(nn.Module):
    def __init__(self,n_features):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=n_features, out_features=256)
        # self.enc2 = nn.Linear(in_features=256, out_features=128)
        # self.enc3 = nn.Linear(in_features=128, out_features=64)
        # self.enc4 = nn.Linear(in_features=64, out_features=32)
        # self.enc5 = nn.Linear(in_features=32, out_features=16)
        # self.enc6 = nn.Linear(in_features=16, out_features=2)
        #
        # # decoder

        # self.dec1 = nn.Linear(in_features=2, out_features=16)
        # self.dec2 = nn.Linear(in_features=16, out_features=32)
        # self.dec3 = nn.Linear(in_features=32, out_features=64)
        # self.dec4 = nn.Linear(in_features=64, out_features=128)
        # self.dec5 = nn.Linear(in_features=128, out_features=256)
        # self.dec1 = nn.Linear(in_features=128, out_features=256)
        self.dec1 = nn.Linear(in_features=256, out_features=n_features)
        # self.dec1 = nn.Linear(in_features=32, out_features=64)
        # self.dec1 = nn.Linear(in_features=64, out_features=128)
        # self.dec1 = nn.Linear(in_features=128, out_features=256)
        # self.dec1 = nn.Linear(in_features=256, out_features=n_features)
    def forward(self, x):
        # x = F.relu(self.enc1(x))
        # x = F.relu(self.enc2(x))
        # x = F.relu(self.enc3(x))
        # x = F.relu(self.enc4(x))
        # x = F.relu(self.enc5(x))
        # x = F.relu(self.enc6(x))
        # x = F.relu(self.dec1(x))
        # x = F.relu(self.dec2(x))
        # x = F.relu(self.dec3(x))
        # x = F.relu(self.dec4(x))
        # x = F.relu(self.dec5(x))
        # x = F.relu(self.dec6(x))

        x = F.relu(self.enc1(x))
        # x = F.relu(self.enc2(x))
        # # x = F.relu(self.enc3(x))
        # # x = F.relu(self.enc4(x))
        # # x = F.relu(self.enc5(x))
        x = F.relu(self.dec1(x))
        # x = F.relu(self.dec2(x))
        # # x = F.relu(self.dec3(x))
        # # x = F.relu(self.dec4(x))
        # # x = F.relu(self.dec5(x))
        return x
    def encode(self,X):
        x = F.relu(self.enc1(X))
        # x = F.relu(self.enc2(x))
        # x = F.relu(self.enc3(x))
        # x = F.relu(self.enc4(x))
        # x = F.relu(self.enc5(x))
        # x = F.relu(self.enc6(x))
        return x

class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        #
        # self.layer_1 = nn.Linear(num_feature, 512)
        # self.layer_2 = nn.Linear(512, 128)
        # self.layer_3 = nn.Linear(128, 64)
        # self.layer_out = nn.Linear(64, num_class)
        #
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)
        # self.batchnorm1 = nn.BatchNorm1d(512)
        # self.batchnorm2 = nn.BatchNorm1d(128)
        # self.batchnorm3 = nn.BatchNorm1d(64)


        self.layer_1 = nn.Linear(num_feature, 128)
        # self.layer_2 = nn.Linear(128, 64)

        self.layer_out = nn.Linear(128, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(128)
        # self.batchnorm2 = nn.BatchNorm1d(64)


    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        # x = self.layer_2(x)
        # x = self.batchnorm2(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        # x = self.layer_3(x)
        # x = self.batchnorm3(x)
        # x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x



class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False
        if torch.isnan(metrics):
            return True
        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def getEncoder(X_train,X_val,n_feature):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = Autoencoder(n_feature).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()
    train_loader = DataLoader(dataset=X_train,
                              batch_size=16,
                              )
    val_loader = DataLoader(dataset=X_val, batch_size=1)
    es = EarlyStopping(patience=5)
    EPOCHS = 300

    for e in tqdm(range(1, EPOCHS + 1)):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, n_feature).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        with torch.no_grad():
            val_epoch_loss = 0
            model.eval()
            for val_batch_feature, _ in val_loader:
                val_batch_feature = batch_features.view(-1, n_feature).to(device)
                val_outputs = model(val_batch_feature)
                val_loss = criterion(val_outputs, val_batch_feature)
                val_epoch_loss += val_loss.item()

        # compute the epoch training loss
            if es.step(torch.tensor(val_epoch_loss)):
                break
        loss = loss / len(train_loader)

        # display the epoch training loss
        print(
            f'Epoch {e + 0:03}: | Train Loss: {loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} ')
    return model


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def scatter_2d_label(X_2d, y, ax=None, s=2, alpha=0.5, lw=2):
    """Visualise a 2D embedding with corresponding labels.

    X_2d : ndarray, shape (n_samples,2)
        Low-dimensional feature representation.

    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_2d.

    ax : matplotlib axes.Axes
         axes to plot on

    s : float
        Marker size for scatter plot.

    alpha : float
        Transparency for scatter plot.

    lw : float
        Linewidth for scatter plot.
    """

    targets = np.unique(y)  # extract unique labels
    colors = sns.color_palette(palette='hsv_r', n_colors=targets.size)

    if ax is None:
        fig, ax = plt.subplots()

    # scatter plot
    for color, target in zip(colors, targets):
        ax.scatter(X_2d[y == target, 0], X_2d[y == target, 1], color=color, label=target, s=s, alpha=alpha, lw=lw)

    # add legend
    ax.legend(loc='center left', bbox_to_anchor=[1.01, 0.5], scatterpoints=3,
              frameon=False);  # Add a legend outside the plot at specified point
    ax.set_xlim(X_2d[:,0].min(), X_2d[:,0].max())
    ax.set_ylim(X_2d[:,1].min(), X_2d[:,1].max())
    fig.savefig('naive.png')

ing_mat = loadmat('MATLAB/ingredients.mat')['ingredients']
cityDist_mat = loadmat('MATLAB/citiesDistMat.mat')['citiesDistMat']
labelName_mat = loadmat('MATLAB/labelNames.mat')['labelNames']
labels_mat = loadmat('MATLAB/labels.mat')['labels']
recipe_mat = loadmat('MATLAB/recipes.mat')['recipes']

dataset_full = np.concatenate((recipe_mat, labels_mat-1), axis=1)
train_dataset, X_test = train_test_split(dataset_full,random_state=random_state, test_size=0.2)
X_train, X_val = train_test_split(train_dataset,random_state=random_state, test_size=0.2)
ing_headline = []
for i in ing_mat[0]:
    ing_headline.append(i[0])
ing_headline.append('label')
X_train = pd.DataFrame(X_train, columns=ing_headline)
X_val = pd.DataFrame(X_val, columns=ing_headline)
X_test = pd.DataFrame(X_test, columns=ing_headline)
X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
X_test = X_test.to_numpy()

# train_encoded, decoded, final_loss  = QuickEncode(X_train[:,:-1], embedding_dim=2)
train_pca = PCA(n_components=529).fit(X_train[:,:-1]) # we can chain the instantiation and the fitting
X_train_pca = train_pca.transform(X_train[:,:-1])
X_val_pca = train_pca.transform(X_val[:,:-1])
X_test_pca = train_pca.transform(X_test[:,:-1])

# train_dataset = ClassifierDataset(X_train[:,:-1], X_train[:,-1])
# val_dataset = ClassifierDataset(X_val[:,:-1], X_val[:,-1])
# test_dataset = ClassifierDataset(X_test[:,:-1], X_test[:,-1])
train_dataset = ClassifierDataset(X_train_pca, X_train[:,-1])
val_dataset = ClassifierDataset(X_val_pca, X_val[:,-1])
test_dataset = ClassifierDataset(X_test_pca, X_test[:,-1])
target_list = []
for _, t in train_dataset:
    target_list.append(t)

target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]

class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3)
        self.fc1 = nn.Linear(3*3*128, 64)
        self.fc2 = nn.Linear(64, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print ("1", x.size())

        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        # print ("2", x.size())

        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        # print ("3", x.size())
        x = x.view(-1,3*3*128 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    # def __init__(self,num_class):
    #     super(CNN, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(32*2*2, 64)
    #     self.fc2 = nn.Linear(64, num_class)
    #
    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     # print ("1",x.size())
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     # print ("2",x.size())
    #
    #     x = x.view(-1, 2*2*32)
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return F.log_softmax(x)

# encoder = getEncoder(train_dataset,val_dataset,709)
# X_train_encode = encoder.encode(torch.tensor(X_train[:,:-1]).float())
# #--- top 10 most used ingredients
# # scatter_2d_label(X_train_encode.detach().numpy(), X_train[:,-1])
# # scatter_2d_label(train_encoded, X_train[:,-1])
#
# X_val_encode = encoder.encode(torch.tensor(X_val[:,:-1]).float())
# X_test_encode = encoder.encode(torch.tensor(X_test[:,:-1]).float())
#
# encoder = getEncoder(train_dataset,val_dataset,300)
# X_train_encode = encoder.encode(torch.tensor(X_train_pca).float())
#--- top 10 most used ingredients
# scatter_2d_label(X_train_encode.detach().numpy(), X_train[:,-1])
# scatter_2d_label(train_encoded, X_train[:,-1])
#
# X_val_encode = encoder.encode(torch.tensor(X_val_pca).float())
# X_test_encode = encoder.encode(torch.tensor(X_test_pca).float())

train_dataset = ClassifierDataset(X_train_pca, X_train[:,-1])
val_dataset = ClassifierDataset(X_val_pca, X_val[:,-1])
test_dataset = ClassifierDataset(X_test_pca, X_test[:,-1])
# train_dataset = ClassifierDataset(X_train_encode, X_train[:,-1])
# val_dataset = ClassifierDataset(X_val_encode, X_val[:,-1])
# test_dataset = ClassifierDataset(X_test_encode, X_test[:,-1])


EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0007
# NUM_FEATURES = X_train_encode.size()[1]
NUM_FEATURES = 500

NUM_CLASSES = 12
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN(num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)


accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}



print("Begin training.")
es = EarlyStopping(mode='max',patience=25)

for e in tqdm(range(1, EPOCHS + 1)):

    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0
        val_epoch_acc = 0

        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
        #
        if es.step(torch.tensor(val_epoch_acc)):
            break
    loss_stats['train'].append(train_epoch_loss / len(train_loader))
    loss_stats['val'].append(val_epoch_loss / len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

    print(
        f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')



