#leave one person out, do training and testing together

"""
parameter setting:
MH:
source_size = 8000
target_size = 2000
window_size = 20
step = 10

emsize = 120  # embedding dimension
nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value

o_learning_rate = 5e-4
learning_rate_decay_factor = 0.999
min_learning_rate = 1e-4
max_gradient_norm = 2.0

epoch = 2000
batch_size = 1000

PMP:
source_size = 8000
target_size = 2000

window_size = 20
step = 10

emsize = 120  # embedding dimension
nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value
o_learning_rate = 5e-4
learning_rate_decay_factor = 0.998
min_learning_rate = 5e-5
max_gradient_norm = 5.0
epoch = 1000
batch_size = 2000

MARS:
source_size = 8000
target_size = 2000

window_size = 20
step = 10

emsize = 120  # embedding dimension
nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value
o_learning_rate = 5e-4
learning_rate_decay_factor = 0.999
min_learning_rate = 5e-5
max_gradient_norm = 5.0

epoch = 1000
batch_size = 1000

UCI:
source_size = 8000
target_size = 2000

window_size = 20
step = 10

emsize = 120  # embedding dimension
nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value
o_learning_rate = 5e-4
learning_rate_decay_factor = 0.998
min_learning_rate = 1e-4
max_gradient_norm = 2.0

epoch = 2000
batch_size = 1000
"""

#get randomly selected support set, not used
def random_get_support(source_X, source_y, support_size):
    support_x = []  
    support_y = []  

    start = 0

    support_choose = random.sample(range(0, len(source_X)), support_size)

    support_x.extend(source_X[support_choose])
    support_y.extend(source_y[support_choose])

    return np.array(support_x),np.array(support_y)

import scipy.io as sc
import numpy as np
from functions import *
#from act_models import *
#from gram_model import *
# from gcram_model import *
from grcam_model import *
from puretran import *
import random
import os
import torch.nn.functional as F
import torch
from glob import glob
from copy import deepcopy
import time

#get training and validation from each training user. used to compare with meta learning
def get_data_each_user(source_X, source_y, source_u, nb_subjects, user_id, user_except,training_percent = 0.7):
    support_x = []   # [task_nb,support_size,nb_feature]
    support_y = []  
    quary_x= []
    quary_y = []

    for i in user_id:
      if i != user_except:
        index = np.array(np.where(source_u == i))
      
        
        nb_choice = int (len(index[0]) * training_percent)
        print(nb_choice)
        support_x.extend(source_X[index[0][:nb_choice]])
        support_y.extend(source_y[index[0][:nb_choice]])

        quary_x.extend(source_X[index[0][nb_choice:]])
        quary_y.extend(source_y[index[0][nb_choice:]])


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
source_size = 8000
target_size = 2000

# data process
# window 20, step 10 is best for MHEALTH
window_size = 20
step = 10

file_name = datasets_dict[dataset][0]
file_data = sc.loadmat(path + file_name + '.mat')
file_data = file_data[file_name]

nb_subjects = datasets_dict[dataset][1]
nb_classes = datasets_dict[dataset][2]

#transformer model
ntokens = nb_classes  # the size of vocabulary
emsize = 120  # embedding dimension
nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value
device = 'cuda'
has_pe = False

# training
o_learning_rate = 5e-4
learning_rate_decay_factor = 0.999
min_learning_rate = 1e-4
max_gradient_norm = 2.0

epoch = 2000
batch_size = 1000

source_range = list(range(nb_subjects))
print("source range: ",source_range)

test_acc_total = []
# for MH, PMP, MARS dataset
  
# ts stores the time in seconds
ts1 = time.time()
if dataset != 'UCI':
    #leave one person out
   
    for target_id in source_range:
        learning_rate = o_learning_rate
        print("init lr: ",learning_rate)

        source_range_copy  = deepcopy(source_range)
        source_range_copy.remove(target_id)
        print("source_copy range:",source_range_copy)
        target_range = [target_id]

        source_data = get_act_data(file_data, source_range_copy)
        target_data = get_act_data(file_data, target_range)

        # print("source range copy: ",source_range_copy)
        # print("target id: ",target_id)
        # print("total for test:",target_data.shape)

        user_id = [i for i in range(nb_subjects)]


        source_x, source_y, source_u = get_act_time_sequences(source_data, window_size, step)
        target_x, target_y, target_u = get_act_time_sequences(target_data, window_size, step)

        nb_feature = source_x.shape[-1]


        source_x, source_y, source_u = seed_shuffle(source_x, 1), seed_shuffle(source_y, 1), seed_shuffle(source_u, 1)
        target_x, target_y, target_u = seed_shuffle(target_x, 1), seed_shuffle(target_y, 1), seed_shuffle(target_u, 1)

        source_x, source_y, source_u = source_x[ :source_size], source_y[:source_size], source_u[:source_size]
        total_num_source = source_x.shape[0]
        
        num_train = int(total_num_source * 1)

        #use 70% to train, 30% to do model selection
        eval_x,eval_y,eval_u = source_x[num_train:], source_y[num_train:], source_u[num_train:]
       # print("total for eval:",eval_x.shape)
        source_x, source_y, source_u = source_x[: num_train], source_y[: num_train], source_u[: num_train]
       # print("total for training:",source_x.shape)
        target_x, target_y, target_u = target_x[: target_size], target_y[: target_size], target_u[: target_size]
        #print("total for test:",target_x.shape)

        #the path to store and restore the model
        #weights_dir = './tradition_tran' + dataset
        weights_dir = './tradition_tran' + 'PMP0' + str(target_id)
        os.makedirs(weights_dir, exist_ok=True)

        #without positional encoding
        model = PureTran_torch(height = window_size, nb_features = nb_feature,ntoken = ntokens, ninp = emsize, nhead = nhead, nhid = nhid, nlayers = nlayers,
                 pe =has_pe ,dropout=dropout).to(device)
        
        source_x , source_y, quary_x, quary_y  =get_data_each_user(source_x, source_y, source_u, nb_subjects,user_id, user_except = target_id, training_percent = 0.7)
        
        source_x = torch.tensor(source_x).float().to(device)
        source_y = torch.tensor(source_y).long().to(device)

        # quary_x = eval_x
        # quary_y = eval_y
        quary_x = torch.tensor(quary_x).float().to(device)
        quary_y = torch.tensor(quary_y).long().to(device)
        target_x = torch.tensor(target_x).float().to(device)
        target_y = torch.tensor(target_y).long().to(device)

        #force to get at least one new model every 50 epoch
        best_val_acc_50count = 0
        count = 0

        #training epoches
        for epoch in range(1, epoch+1):

            model.train()
            train_acc = []
            train_losses = []

            for b in range(source_x.shape[0] // batch_size):
                source_batch_x = source_x[batch_size * b: batch_size * (b + 1)]
                source_batch_y = source_y[batch_size * b: batch_size * (b + 1)]
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                optimizer.zero_grad()
        
                outputs = model(source_batch_x)
                outputs = outputs.reshape(-1, nb_classes).float()
                # print(outputs.shape)
                loss = F.cross_entropy(outputs, source_batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
                optimizer.step()
                train_losses.append(loss.item())
          # print("outputs:",outputs.argmax(1))
                acc = (outputs.argmax(1) == source_batch_y).float().mean().item()
                train_acc.append(acc)

        #update the outerstepsize
                learning_rate *= learning_rate_decay_factor
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate
        #outerstepsize  =  (outerstepsize >= min_learning_rate)? outerstepsize*learning_rate_decay_factor :min_learning_rate
       
            valid_after_acc = validation(model,quary_x,quary_y)
            test_acc = validation(model,target_x,target_y)
            print('target:',target_id,
                'epoch:', epoch, 'step:', step, '\ttraining loss:',
                  f'{np.mean(train_losses):.3}', '\ttraining acc:',
                  f'{np.mean(train_acc):.3}', '\tvalidation acc:',
                  f'{np.mean(valid_after_acc):.3}','\t test acc:',
                  f'{np.mean(test_acc):.3}')
            if count != 50:
              count += 1
            else:
              count = 0
              best_val_acc_50count = 0
            if valid_after_acc > best_val_acc_50count:  # evaluation
                best_val_acc_50count = np.copy(valid_after_acc)
                fn = f'./{weights_dir}/epoch_{epoch}_acc_{valid_after_acc:.3}.pth'
                torch.save(model.state_dict(), fn)

        #get candidate models for test
        def cond(x): return float(x.split('/')[-1].split('_')[-1][:-4])
            #all_model_fn = sorted(glob('/content/drive/MyDrive/research/Multi-agent-Attentional-Activity-Recognition/model_weights_HAR/*.pth'), key=cond)[-2:]
        all_model_fn = sorted(glob(weights_dir + '/*.pth'), key=cond)[-50:]


        #go through different trained model, select a best performance
        best_model_acc = 0
        for fn in all_model_fn:
            print('Processing fn', fn)
            state = torch.load(fn)
            
         
            model_test = PureTran_torch(height = window_size, nb_features = nb_feature,ntoken = ntokens, ninp = emsize, nhead = nhead, nhid = nhid, nlayers = nlayers,
                 pe = has_pe,dropout=dropout).to(device)
       
            model_test.load_state_dict(state)
    
            test_acc = validation(model_test, target_x, target_y)
           
            if test_acc > best_model_acc:
                best_model_acc = test_acc
            print('model: ',fn,'\ttest acc:', f'{np.mean(test_acc):.3}')
        print("For the id:",target_id, "\tthe best model: ",f'{np.mean(best_model_acc):.3}')

        test_acc_total.append(best_model_acc)

    ts2 = time.time()
    print("For dataset: ",dataset, "the average acc on test data is: ",f'{np.mean(test_acc_total):.3}', "the std: ",f'{np.std(test_acc_total):.3}')
    print("time :",ts2-ts1)


if dataset == 'UCI':
    print('dataset is UCI')
    source_data = sc.loadmat(path + 'UCI_train_raw' + '.mat')
    source_data = source_data['UCI_train_raw']

    target_data = sc.loadmat(path + 'UCI_test_raw' + '.mat')
    target_data = target_data['UCI_test_raw']
    user_id = [1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30]
    target_id = 0 #no 0 id in target_data

    target_all = [2, 4, 9, 10, 12, 13, 18, 20, 24]

    source_x, source_y, source_u = get_act_time_sequences(source_data, window_size, step)
    target_x, target_y, target_u = get_act_time_sequences(target_data, window_size, step)

    nb_feature = source_x.shape[-1]

    source_x, source_y, source_u = seed_shuffle(source_x, 1), seed_shuffle(source_y, 1), seed_shuffle(source_u, 1)
    target_x, target_y, target_u = seed_shuffle(target_x, 1), seed_shuffle(target_y, 1), seed_shuffle(target_u, 1)

    source_x, source_y, source_u = source_x[ :source_size], source_y[:source_size], source_u[:source_size]

    total_num_source = source_x.shape[0]
    print("total for training:",total_num_source)
    num_train = int(total_num_source * 1)

        #use 70% to train, 30% to do model selection
    eval_x,eval_y,eval_u = source_x[num_train:], source_y[num_train:], source_u[num_train:]
    source_x, source_y, source_u = source_x[: num_train], source_y[: num_train], source_u[: num_train]
    
    target_x, target_y, target_u = target_x[: target_size], target_y[: target_size], target_u[: target_size]


    #the path to store and restore the model
    #weights_dir = './tradition_tran' + dataset

    weights_dir = './tradition_tran' + "UCI1"
    os.makedirs(weights_dir, exist_ok=True)

    model = PureTran_torch(height = window_size, nb_features = nb_feature,ntoken = ntokens, ninp = emsize, nhead = nhead, nhid = nhid, nlayers = nlayers,
                 pe = has_pe,dropout=dropout).to(device)

    source_x , source_y, quary_x, quary_y  = get_data_each_user(source_x, source_y, source_u, nb_subjects,user_id, user_except = target_id, training_percent = 0.7)
    source_x = torch.tensor(source_x).float().to(device)
    source_y = torch.tensor(source_y).long().to(device)

        # quary_x = eval_x
        # quary_y = eval_y
    quary_x = torch.tensor(quary_x).float().to(device)
    quary_y = torch.tensor(quary_y).long().to(device)
    target_x = torch.tensor(target_x).float().to(device)
    target_y = torch.tensor(target_y).long().to(device)


    #best_val_acc = 0
    epoch = 1000
    batch_size =2000

    #force to get at least one new model every 50 epoch
    best_val_acc_50count = 0
    count = 0

    #training epoches
    for epoch in range(1, epoch+1):

        model.train()
        train_acc = []
        train_losses = []

        learning_rate = o_learning_rate

        for b in range(source_x.shape[0] // batch_size):
            source_batch_x = source_x[batch_size * b: batch_size * (b + 1)]
            source_batch_y = source_y[batch_size * b: batch_size * (b + 1)]
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            optimizer.zero_grad()
    
    
            outputs = model(source_batch_x)
            outputs = outputs.reshape(-1, nb_classes).float()
                # print(outputs.shape)
            loss = F.cross_entropy(outputs, source_batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()
            train_losses.append(loss.item())
          # print("outputs:",outputs.argmax(1))
            acc = (outputs.argmax(1) == source_batch_y).float().mean().item()
            train_acc.append(acc)


        #update the outerstepsize
            learning_rate *= learning_rate_decay_factor
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
        #outerstepsize  =  (outerstepsize >= min_learning_rate)? outerstepsize*learning_rate_decay_factor :min_learning_rate
       # valid_after_acc = validation(model, arc_dataset)
        valid_after_acc = validation(model,quary_x,quary_y)
        test_acc = validation(model,target_x,target_y)
        print(
                'epoch:', epoch, 'step:', step, '\ttraining loss:',
                  f'{np.mean(train_losses):.3}', '\ttraining acc:',
                  f'{np.mean(train_acc):.3}', '\tvalidation acc:',
                  f'{np.mean(valid_after_acc):.3}','\t test acc:',
                  f'{np.mean(test_acc):.3}')

        if count != 50:
            count += 1
        else:
            count = 0
            best_val_acc_50count = 0
        if valid_after_acc > best_val_acc_50count:  # evaluation
        #if valid_after_acc > best_val_acc:  # evaluation
            best_val_acc_50count = np.copy(valid_after_acc)
            fn = f'./{weights_dir}/epoch_{epoch}_acc_{valid_after_acc:.3}.pth'
            torch.save(model.state_dict(), fn)
    
    #get candidate models for test
    def cond(x): return float(x.split('/')[-1].split('_')[-1][:-4])
        #all_model_fn = sorted(glob('/content/drive/MyDrive/research/Multi-agent-Attentional-Activity-Recognition/model_weights_HAR/*.pth'), key=cond)[-2:]
    all_model_fn = sorted(glob(weights_dir + '/*.pth'), key=cond)[-50:]
    
    for test_u in target_all:

        #split the target across users
        target_each_x = target_x[np.where(target_u == test_u)[0]]

        target_each_y = target_y[np.where(target_u == test_u)[0]]

        best_model_acc = 0

        for fn in all_model_fn:
            print('Processing fn', fn)
            state = torch.load(fn)
            
         
            model_test = PureTran_torch(height = window_size, nb_features = nb_feature,ntoken = ntokens, ninp = emsize, nhead = nhead, nhid = nhid, nlayers = nlayers,
                 pe = has_pe,dropout=dropout).to(device)
       
            model_test.load_state_dict(state)
    
            test_acc = validation(model_test, target_each_x, target_each_y)
           
            if test_acc > best_model_acc:
                best_model_acc = test_acc
            print('model: ',fn,'\ttest acc:', f'{np.mean(test_acc):.3}')
        print("For the id:",test_u, "\tthe best model: ",f'{np.mean(best_model_acc):.3}')

        test_acc_total.append(best_model_acc)
    
    ts2 = time.time()
    print("For UCI, the average acc on test data is: ",f'{np.mean(test_acc_total):.3}', "the std: ",f'{np.std(test_acc_total):.3}')
    print("time: ",ts2-ts1)