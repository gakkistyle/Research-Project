
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
import os

# get support and quary data within each task
def get_meta_data(source_X, source_y, support_size, quary_size, task_nb, each_task):
    support_x = []   # [task_nb,support_size,nb_feature]
    support_y = []  
    quary_x= []
    quary_y = []
    start_support = 0
    start_quary = support_size
    for i in range(task_nb):
        support_x.append(source_X[start_support:start_support+support_size])
        support_y.append(source_y[start_support:start_support+support_size])
        quary_x.extend(source_X[start_quary:start_quary+quary_size])
        quary_y.extend(source_y[start_quary:start_quary+quary_size])
        start_support += each_task
        start_quary += each_task

    return support_x,support_y,quary_x,quary_y

#split to nb_task number of tasks, let 70% task be train, 30% be validation
def get_meta_split(source_X,source_y, nb_all, nb_train = 0.7,nb_task =20 ):
  nb_train_task = int(nb_train * nb_task)
  nb_each_task = int(nb_all//nb_task)
  nb_support = int(nb_all  * nb_train)
  support_x = []
  support_y = []
  quary_x = []
  quary_y = []

  for i in range(nb_train_task):
    support_x.append(source_X[i*nb_each_task:(i+1)*nb_each_task])
    support_y.append(source_y[i*nb_each_task:(i+1)*nb_each_task])

  quary_x.extend(source_X[nb_support:])
  quary_y.extend(source_y[nb_support:])

  #print(np.array(quary_x).shape)
  return support_x,support_y,quary_x,quary_y

#get randomly selected support set and evaluation set
def random_get_support_quary(source_X, source_y, support_size, quary_size, task_nb, each_task):
    support_x = []   # [task_nb,support_size,nb_feature]
    support_y = []  
    quary_x= []
    quary_y = []
    start = 0

    for i in range(task_nb):
        support_choose = random.sample(range(0, each_task), support_size)
        quary_choose = random.sample(range(0, each_task), quary_size)

        sub_x = source_X[start:start+each_task]
        sub_y = source_y[start:start+each_task]

        support_x.append(sub_x[support_choose])
        support_y.append(sub_y[support_choose])
        #quary set as a whole to evaluate the step
        quary_x.extend(sub_x[quary_choose])
        quary_y.extend(sub_y[quary_choose])
        start += each_task
    return support_x,support_y,quary_x,quary_y

#get training and validation from each training user, used in this report
def get_data_each_user(source_X, source_y, source_u, nb_subjects, training_percent = 0.7):
    support_x = []   # [task_nb,support_size,nb_feature]
    support_y = []  
    quary_x= []
    quary_y = []

    for i in range(nb_subjects):
        index = np.where(source_u == i)
        nb_choice = int (len(index) * training_percent)

        support_x.append(source_X[index[:nb_choice]])
        support_y.append(source_y[index[:nb_choice]])

        quary_x.extend(source_X[index[nb_choice:]])
        quary_y.extend(source_y[index[nb_choice:]])


    return support_x,support_y,quary_x,quary_y

#do evaluation after each task
@torch.no_grad()
def validation(model, quary_x, quary_y):
    model.eval()

    outputs = F.softmax(model(quary_x),dim=2)
   # print("outputs shape: ",outputs.shape)
    outputs = outputs.argmax(2).reshape(-1)
  #  print("outputs shape after argmax: ",outputs.shape)
   # print(outputs)
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
dataset = 'PMP6'
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

max_gradient_norm = 5.0

# model
# g_size = 128  # Size of theta_g^0
# l_size = 128  # Size of theta_g^1
# glimpse_output_size = 220  # Output size of Glimpse Network
# cell_size = 220  # Size of LSTM cell
# nb_glimpses = 30  # Number of glimpses: 30
# variance = 0.22  # Gaussian variance for Location Network 
# # M = 1 # Monte Carlo sampling, see Eq(2) (not used)


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



#support_x , support_y , quary_x, quary_y = get_meta_data(source_X,source_y,support_size,quary_size,task_nb,each_task)


# model
img_height = window_size
img_width = nb_feature

#transformer
ntokens = nb_classes  # the size of vocabulary
emsize = 120  # embedding dimension
nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value
device = 'cuda'

model = PureTran_torch(height = img_height, nb_features = nb_feature,ntoken = ntokens, ninp = emsize, nhead = nhead, nhid = nhid, nlayers = nlayers,
                 dropout=dropout).to(device)

innerstepsize = 1e-4  # stepsize in inner loop, 1e-2 really bad
innerepochs = 30  # number of epochs of each inner loop

outerstepsize = 0.01
decay_rate = 0.999
min_learning_rate = 1e-3
epoch = 100

weights_dir = './model_weights_HAR_MH'
os.makedirs(weights_dir, exist_ok=True)
best_val_acc = 0
#split to task
task_nb = 100    #100 tasks in training set
each_task = int(source_size/task_nb)  #each task has this num of 

support_size = 56
quary_size = 24
nb_train = 0.7
#task_ids = np.arange(int(nb_train *task_nb))
task_ids = np.arrange(task_nb)

task_nb = nb_subjects-1

#support_x , support_y , quary_x, quary_y = get_meta_split(source_X,source_y,  nb_all=len(source_X),nb_train =nb_train,nb_task=task_nb)

support_x , support_y , quary_x, quary_y = get_data_each_user(source_X, source_y, source_u, nb_subjects, training_percent =nb_train)

for epoch in range(1, epoch+1):
    # get a random support set per task every epoch
    #support_x , support_y , quary_x, quary_y = random_get_support_quary(source_X,source_y,support_size,quary_size,task_nb,each_task)
    quary_x = torch.tensor(quary_x).float().to(device)
    quary_y = torch.tensor(quary_y).long().to(device)
    # randomly shuffle the task orders.
    np.random.shuffle(task_ids)
    for step, task_num in enumerate(task_ids):
        x, y = support_x[task_num], support_y[task_num]
        model.train()
        train_acc = []
        train_losses = []
        x = torch.tensor(x).float().to(device) #shape: [batch_size, 225]
        y = torch.tensor(y).long().to(device) #shape: [batch_size * 225]
        #x = x.long() 
        #y = y.long() 
        #print("y:shape",y.shape)
            #record the weight before the task training
        weights_before = deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=innerstepsize)
        # starts training over the task
        for _ in range(innerepochs):
            optimizer.zero_grad()
            outputs = model(x)
            outputs = outputs.reshape(-1, nb_classes).float()
           # print(outputs.shape)
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            train_losses.append(loss.item())
          # print("outputs:",outputs.argmax(1))
            acc = (outputs.argmax(1) == y).float().mean().item()
            train_acc.append(acc)

        #valid_before_acc = validation(model, arc_dataset)
        valid_before_acc = validation(model,quary_x,quary_y)
            # if (step % 20 == 0):
            #     outerstepsize = outerstepsize * \
            #         (1 - epoch / args.epoch)  # linear schedule
            #     print('outerstepsize:', outerstepsize)

            # print('Interpolating weights.')
            # Interpolate between current weights and trained weights from this task
            # I.e. (weights_before - weights_after) is the meta-gradient

            #record the weight after task training
        weights_after = model.state_dict()

            #get the reptile
        model.load_state_dict({name:
                                weights_before[name] + (weights_after[name] -
                                                        weights_before[name]) * outerstepsize
                                for name in weights_before})

        #update the outerstepsize
        outerstepsize *= decay_rate
        if outerstepsize < min_learning_rate:
          outerstepsize = min_learning_rate
        #outerstepsize  =  (outerstepsize >= min_learning_rate)? outerstepsize*learning_rate_decay_factor :min_learning_rate
       # valid_after_acc = validation(model, arc_dataset)
        valid_after_acc = validation(model,quary_x,quary_y)
        print('epoch:', epoch, 'step:', step, '\ttraining loss:',
                  f'{np.mean(train_losses):.3}', '\ttraining acc:',
                  f'{np.mean(train_acc):.3}', '\tvalidation before acc:',
                  f'{np.mean(valid_before_acc):.3}', '\tvalidation after acc:',
                  f'{np.mean(valid_after_acc):.3}')
        if valid_after_acc > best_val_acc:  # evaluation
            best_val_acc = np.copy(valid_after_acc)
            fn = f'./{weights_dir}/epoch_{epoch}_step_{step}_acc_{valid_after_acc:.3}.pth'
            torch.save(model.state_dict(), fn)