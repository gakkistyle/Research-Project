import scipy.io as sc
import numpy as np
from functions import *
#from act_models import *
#from gram_model import *
# from gcram_model import *
#from grcam_model import *
from puretran import *
from copy import deepcopy
import torch.nn.functional as F
import torch
import random
from glob import glob



#do evaluation after each task
@torch.no_grad()
def validation(model, quary_x, quary_y):
    model.eval()
    outputs = F.softmax(model(quary_x),dim=2)
    outputs = outputs.argmax(2).reshape(-1)
    acc = (outputs == quary_y).float().mean().item()
    return np.mean(acc)

path = '/content/drive/MyDrive/research/Multi-agent-Attentional-Activity-Recognition/Data_/'

# dict: [filename, sub, class]
datasets_dict = {'MH8': ['MHEALTH_8class', 10, 8],
                 'MH6': ['MHEALTH_balance', 10, 6],  # better
                 # 'OPP': ['Opp_Gesture', 4, 17],
                 # 'OPP5': ['Opp_loco_5', 4, 5],
                 'PMP8': ['PAMAP2_8class', 8, 8],
                 'PMP6': ['PAMAP2_Protocol_6_classes_balance', 8, 6],  # better
                 'UCI': ['UCI_train_raw', 10, 6],
                 # 'EEG': ['EEG_10sub_4act', 10, 4],
                 'MARS': ['MARS', 8, 5]}


# get data
dataset = 'MH6'
target_id = 3
source_size = 8000
target_size = 2000
batch_size = 5000
training_epochs = 2000

# data process
# window 20, step 10 is best for MHEALTH
window_size = 20
step = 10

# training
learning_rate = 5e-3
learning_rate_decay_factor = 0.97
min_learning_rate = 1e-5
max_gradient_norm = 5.0

# model
g_size = 128  # Size of theta_g^0
l_size = 128  # Size of theta_g^1
glimpse_output_size = 220  # Output size of Glimpse Network
cell_size = 220  # Size of LSTM cell
nb_glimpses = 30  # Number of glimpses: 30
variance = 0.22  # Gaussian variance for Location Network
# M = 1 # Monte Carlo sampling, see Eq(2) (not used)


file_name = datasets_dict[dataset][0]
file_data = sc.loadmat(path + file_name + '.mat')
file_data = file_data[file_name]

nb_subjects = datasets_dict[dataset][1]
nb_classes = datasets_dict[dataset][2]

source_range = list(range(nb_subjects))
source_range.remove(target_id)
target_range = [target_id]

source_data = get_act_data(file_data, source_range)
target_data = get_act_data(file_data, target_range)

if dataset == 'UCI':
    print('dataset is UCI')
    source_data = sc.loadmat(path + 'UCI_train_raw' + '.mat')
    source_data = source_data['UCI_train_raw']

    target_data = sc.loadmat(path + 'UCI_test_raw' + '.mat')
    target_data = target_data['UCI_test_raw']

# print(source_data.shape)

source_X, source_y, source_u = get_act_time_sequences(source_data, window_size, step)
target_X, target_y, target_u = get_act_time_sequences(target_data, window_size, step)

nb_feature = source_X.shape[-1]
# source_X = source_X.reshape([source_X.shape[0], -1])
# target_X = target_X.reshape([target_X.shape[0], -1])

source_X, source_y, source_u = seed_shuffle(source_X, 1), seed_shuffle(source_y, 1), seed_shuffle(source_u, 1)
target_X, target_y, target_u = seed_shuffle(target_X, 1), seed_shuffle(target_y, 1), seed_shuffle(target_u, 1)

source_X, source_y, source_u = source_X[: source_size], source_y[: source_size], source_u[: source_size]
target_X, target_y, target_u = target_X[: target_size], target_y[: target_size], target_u[: target_size]

#split to task
task_nb = 100    #100 tasks in training set
each_task = int(source_size/task_nb)  #each task has this num of 

support_size =  nb_classes # n-way k-shot
quary_size = target_size

#support_x , support_y , quary_x, quary_y = get_meta_data(source_X,source_y,support_size,quary_size,task_nb,each_task)


# model
img_height = window_size
img_width = nb_feature


#transformer
ntokens = nb_classes  # the size of vocabulary
emsize = 120  # embedding dimension
nhid = 2048   # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value
device = 'cuda'

model = PureTran_torch(height = img_height, nb_features = nb_feature,ntoken = ntokens, ninp = emsize, nhead = nhead, nhid = nhid, nlayers = nlayers,
                 dropout=dropout).to(device)

innerstepsize = 1e-5 # stepsize in inner loop
innerepochs = 50  # number of epochs of each inner loop

#outerstepsize = 0.1

epoch = 30

#weights_dir = '/content/drive/MyDrive/research/Multi-agent-Attentional-Activity-Recognition/model_weights_HAR'
weights_dir = './tradition_tran'
best_val_acc = 0
task_ids = np.arange(task_nb)

def cond(x): return float(x.split('/')[-1].split('_')[-1][:-4])
#all_model_fn = sorted(glob('/content/drive/MyDrive/research/Multi-agent-Attentional-Activity-Recognition/model_weights_HAR/*.pth'), key=cond)[-2:]
all_model_fn = sorted(glob('./tradition_tran/*.pth'), key=cond)[-8:]

#the quary set is simply the target set from above
quary_x,quary_y = target_X, target_y
quary_x = torch.tensor(quary_x).float().to(device)
quary_y = torch.tensor(quary_y).long().to(device)

for fn in all_model_fn:
    print('Processing fn', fn)
    state = torch.load(fn)

    all_train_acc = []
    all_test_acc = []
    #every epoch is another sample and test
    for epoch in range(1, epoch+1):
        # get a random support set per task every epoch
        # support_x , support_y= random_get_Nway_Kshot(target_X, target_y,N_way,K_shot)
        
        # #need to reload the state every epoch
        model.load_state_dict(state)
        # x, y = support_x, support_y
         
        # train_acc = []
        # train_losses = []
        # x = torch.tensor(x).float() 
        # x = torch.reshape(x,(-1,img_height,img_width)).to(device)
        # y = torch.tensor(y).long().to(device)

        # #print("y:shape",y.shape)
          
        # optimizer = torch.optim.AdamW(model.parameters(), lr=innerstepsize)

        # model.train()
        # for _ in range(innerepochs):
        #     optimizer.zero_grad()
        #     #print("x shape: ",x.shape)
        #     outputs = model(x)
        #     outputs = outputs.reshape(-1, nb_classes).float()
        #     loss = F.cross_entropy(outputs, y)
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        #     optimizer.step()
        #     train_losses.append(loss.item())
        #     acc = (outputs.argmax(1) == y).float().mean().item()
        #     train_acc.append(acc)

        # print('\ttraining loss:',
        #       np.mean(train_losses), '\ttraining acc:', np.mean(train_acc))
        #all_train_acc.append(np.mean(train_acc))

        test_acc = validation(model, quary_x, quary_y)
        all_test_acc.append(test_acc)
        print('epoch:', epoch,  '\ttest acc:',
                  f'{np.mean(test_acc):.3}')

    print('model: ',fn,'\taverage training loss:',
                    f'{np.mean(all_train_acc):.3}','\taverage test acc:', f'{np.mean(all_test_acc):.3}')
           

            #record the weight after task training
        