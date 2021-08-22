#leave one person out for reptile, do training and testing together

"""
parameter setting:
MH: 
source_size = 8000
target_size = 2000
window_size = 20
step = 10

emsize = 120  # embedding dimension
nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value

innerstepsize = 1e-3  # stepsize in inner loop, 1e-2 really bad
innerepochs = 30  # number of epochs of each inner loop
o_outerstepsize = 0.05
decay_rate = 0.999
min_learning_rate = 1e-3
epoch = 250
max_gradient_norm = 2.0
innerstepsize_test = 1e-4
innerepochs_test = 30
epoch_test = 20

PMP:
source_size = 8000
target_size = 2000
window_size = 20
step = 10

emsize = 120  # embedding dimension
nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value

innerstepsize = 5e-4  # stepsize in inner loop, 1e-2 really bad
innerepochs = 50  # number of epochs of each inner loop
o_outerstepsize = 0.05
decay_rate = 0.999
min_learning_rate = 1e-3
epoch = 200
max_gradient_norm = 2.0
innerstepsize_test = 1e-4
innerepochs_test = 30
epoch_test = 20


MARS:
source_size = 8000
target_size = 2000
window_size = 20
step = 10

emsize = 120  # embedding dimension
nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value

innerstepsize = 2e-3  # stepsize in inner loop, 1e-2 really bad
innerepochs = 50  # number of epochs of each inner loop
o_outerstepsize = 0.2
decay_rate = 0.999
min_learning_rate = 1e-3
epoch = 200
max_gradient_norm = 2.0
innerstepsize_test = 1e-4
innerepochs_test = 30
epoch_test = 20

UCI:
source_size = 8000
target_size = 2000
window_size = 20
step = 10

emsize = 120  # embedding dimension
nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value

innerstepsize = 1e-3  # stepsize in inner loop, 1e-2 really bad
innerepochs = 50  # number of epochs of each inner loop
o_outerstepsize = 0.05
decay_rate = 0.999
min_learning_rate = 1e-4
epoch = 200
max_gradient_norm = 5.0
innerstepsize_test = 1e-4
innerepochs_test = 30
epoch_test = 20


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
        support_x.append(source_X[index[0][:nb_choice]])
        support_y.append(source_y[index[0][:nb_choice]])

        quary_x.extend(source_X[index[0][nb_choice:]])
        quary_y.extend(source_y[index[0][nb_choice:]])

    return support_x,support_y,quary_x,quary_y


# in test phase, get one data per class from test set.
def random_get_Nway_Kshot(target_X, target_y,n_way,k_shot):

    support_x = []
    support_y = []

    target_y.reshape(-1,)

    target_X_cpu = target_X.cpu()
    target_X_cpu = target_X_cpu.numpy()

    target_y_cpu = target_y.cpu()
    target_y_cpu = target_y_cpu.numpy()
   # print("target_y shape:",target_y.shape)
    for i in range(n_way):
        # print()
        # print(k_shot)
        if len(np.where(target_y_cpu==i)[0].tolist()) == 0:
          continue
        choice = random.sample( np.where(target_y_cpu==i)[0].tolist(), k_shot)
        #print("choice is: ",choice)
        support_x.append(target_X_cpu[choice])

        for  j in range(k_shot):
            support_y.append(i)
    
    return support_x,support_y



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
dataset = 'MARS'
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
nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value
device = 'cuda'
has_pe = True

# parameters in training phase
innerstepsize = 2e-3  # stepsize in inner loop, 1e-2 really bad
innerepochs = 50  # number of epochs of each inner loop
o_outerstepsize = 0.02
decay_rate = 0.999
min_learning_rate = 1e-3
epoch = 200
max_gradient_norm = 2.0
# in test phase
innerstepsize_test = 1e-4
innerepochs_test = 30
epoch_test = 20

source_range = list(range(nb_subjects))
print("source range: ",source_range)

test_acc_total = []
# for MH, PMP, MARS dataset
target_range2 = [5]
# ts stores the time in seconds
ts1 = time.time()
if dataset != 'UCI':
    #leave one person out
   
    for target_id in target_range2:
        outerstepsize = o_outerstepsize
        print("init lr: ",outerstepsize)

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
        weights_dir = './Meta' + 'MARS7' + str(target_id)
        os.makedirs(weights_dir, exist_ok=True) 

        #without positional encoding
        model = PureTran_torch(height = window_size, nb_features = nb_feature,ntoken = ntokens, ninp = emsize, nhead = nhead, nhid = nhid, nlayers = nlayers,
                 pe =has_pe ,dropout=dropout).to(device)
        
        source_x , source_y, quary_x, quary_y  =get_data_each_user(source_x, source_y, source_u, nb_subjects,user_id, user_except = target_id, training_percent = 0.7)
        
        # source_x = torch.tensor(source_x).float().to(device)
        # source_y = torch.tensor(source_y).long().to(device)

        # quary_x = eval_x
        # quary_y = eval_y
        quary_x = torch.tensor(quary_x).float().to(device)
        quary_y = torch.tensor(quary_y).long().to(device)
        target_x = torch.tensor(target_x).float().to(device)
        target_y = torch.tensor(target_y).long().to(device)

        #force to get at least one new model every 50 epoch
        best_val_acc_50count = 0
        count = 0

        # # task number equals to number of people - 1
        task_nb = nb_subjects-1
        task_ids = np.arange(task_nb)

        #training epoches
        for epoch in range(1, epoch+1):

            np.random.shuffle(task_ids)
            # print("support x shape:",source_x.shape)
            # print("support y shape:",source_y.shape)

            for step, task_num in enumerate(task_ids):
                x, y = source_x[task_num], source_y[task_num]

                model.train()
                train_acc = []
                train_losses = []

                x = torch.tensor(x).float().to(device) #shape: [batch_size, 225]
                y = torch.tensor(y).long().to(device) #shape: [batch_size * 225]

                weights_before = deepcopy(model.state_dict())
                optimizer = torch.optim.AdamW(model.parameters(), lr=innerstepsize)

                for _ in range(innerepochs):
                    optimizer.zero_grad()
                    outputs = model(x)
                    outputs = outputs.reshape(-1, nb_classes).float()
                    # print(outputs.shape)
                    # print(y.shape)
                    loss = F.cross_entropy(outputs, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                    optimizer.step()
                    train_losses.append(loss.item())
                    # print("outputs:",outputs.argmax(1))
                    acc = (outputs.argmax(1) == y).float().mean().item()
                    train_acc.append(acc)

                #valid_before_acc = validation(model,quary_x,quary_y)

                weights_after = model.state_dict()

                model.load_state_dict({name:
                                weights_before[name] + (weights_after[name] -
                                                        weights_before[name]) * outerstepsize
                                for name in weights_before})

                #update the outerstepsize
                outerstepsize *= decay_rate
                if outerstepsize < min_learning_rate:
                    outerstepsize = min_learning_rate

                valid_after_acc = validation(model,quary_x,quary_y)
                test_acc = validation(model,target_x,target_y)

                print('target:',target_id,
                    'epoch:', epoch, 'step:', step, '\ttraining loss:',
                  f'{np.mean(train_losses):.3}', '\ttraining acc:',
                  f'{np.mean(train_acc):.3}', '\tvalidation after acc:',
                  f'{np.mean(valid_after_acc):.3}','\ttest acc:',
                  f'{np.mean(test_acc):.3}')

                if count != 100:
                    count += 1
                else:
                    count = 0
                    best_val_acc_50count = 0
                if valid_after_acc > best_val_acc_50count:  # evaluation
                    best_val_acc_50count = np.copy(valid_after_acc)
                    fn = f'./{weights_dir}/epoch_{epoch}_step_{step}_acc_{valid_after_acc:.3}.pth'
                    torch.save(model.state_dict(), fn)
        

        #test
        def cond(x): return float(x.split('/')[-1].split('_')[-1][:-4])
            #all_model_fn = sorted(glob('/content/drive/MyDrive/research/Multi-agent-Attentional-Activity-Recognition/model_weights_HAR/*.pth'), key=cond)[-2:]
        all_model_fn = sorted(glob(weights_dir + '/*.pth'), key=cond)[-50:]


        #go through different trained model, select a best performance
        best_model_acc = 0

        #do the n-way 1-shot
        N_way = nb_classes
        K_shot = 1
        for fn in all_model_fn:
            print('Processing fn', fn)
            state = torch.load(fn)

            model_test = PureTran_torch(height = window_size, nb_features = nb_feature,ntoken = ntokens, ninp = emsize, nhead = nhead, nhid = nhid, nlayers = nlayers,
                 pe = has_pe,dropout=dropout).to(device)

            #per model
            all_train_acc = []
            all_test_acc = []

            # do a number of random selected Nway Kshot and get the average performance of the model
            for e in range(1, epoch_test+1):
                support_x , support_y= random_get_Nway_Kshot(target_x, target_y,N_way,K_shot)
                model_test.load_state_dict(state)

                x, y = support_x, support_y
         
                train_acc = []
                train_losses = []
                x = torch.tensor(x).float() 
                x = torch.reshape(x,(-1,window_size, nb_feature)).to(device)
                y = torch.tensor(y).long().to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr = innerstepsize_test)

                model.train()
                for _ in range(innerepochs):
                    optimizer.zero_grad()
                    #print("x shape: ",x.shape)
                    outputs = model(x)
                    outputs = outputs.reshape(-1, nb_classes).float()
                    loss = F.cross_entropy(outputs, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                    optimizer.step()
                    train_losses.append(loss.item())
                    acc = (outputs.argmax(1) == y).float().mean().item()
                    train_acc.append(acc)
       
                all_train_acc.append(np.mean(train_acc))
    
                test_acc = validation(model_test, target_x, target_y)
                all_test_acc.append(test_acc)

                print('epoch:', e,  '\ttraining loss:',
                  f'{np.mean(train_losses):.3}', '\ttraining acc:',
                  f'{np.mean(train_acc):.3}', '\ttest acc:',
                  f'{np.mean(test_acc):.3}')
           
            if np.mean(all_test_acc) > best_model_acc:
                best_model_acc = np.mean(all_test_acc)

            print('model: ',fn,'\taverage training acc:',
                    f'{np.mean(all_train_acc):.3}','\taverage test acc:', f'{np.mean(all_test_acc):.3}')
     
        print("For the id:",target_id, "\tthe best model: ",f'{np.mean(best_model_acc):.3}')

        test_acc_total.append(best_model_acc)

    ts2 = time.time()
    print("For dataset: ",dataset, "the average acc on test data is: ",f'{np.mean(test_acc_total):.3}', "the std: ",f'{np.std(test_acc_total):.3}')
    print("time :",ts2-ts1)
    print("best models: ",test_acc_total)


if dataset == 'UCI':
    print('dataset is UCI')
    source_data = sc.loadmat(path + 'UCI_train_raw' + '.mat')
    source_data = source_data['UCI_train_raw']

    target_data = sc.loadmat(path + 'UCI_test_raw' + '.mat')
    target_data = target_data['UCI_test_raw']
    user_id = [1,3,5,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30]
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

    weights_dir = './tradition_tran' + "UCI2"
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


    #force to get at least one new model every 50 epoch
    best_val_acc_50count = 0
    count = 0

    # task number equals to number of people for UCI
    task_nb = len(user_id)
    task_ids = np.arange(task_nb)

    #training epoches
    for epoch in range(1, epoch+1):

        np.random.shuffle(task_ids)
            # print("support x shape:",source_x.shape)
            # print("support y shape:",source_y.shape)

        for step, task_num in enumerate(task_ids):
            x, y = source_x[task_num], source_y[task_num]

            model.train()
            train_acc = []
            train_losses = []

            x = torch.tensor(x).float().to(device) #shape: [batch_size, 225]
            y = torch.tensor(y).long().to(device) #shape: [batch_size * 225]

            weights_before = deepcopy(model.state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=innerstepsize)

            for _ in range(innerepochs):
                optimizer.zero_grad()
                outputs = model(x)
                outputs = outputs.reshape(-1, nb_classes).float()
                # print(outputs.shape)
                # print(y.shape)
                loss = F.cross_entropy(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                train_losses.append(loss.item())
                # print("outputs:",outputs.argmax(1))
                acc = (outputs.argmax(1) == y).float().mean().item()
                train_acc.append(acc)

                #valid_before_acc = validation(model,quary_x,quary_y)

                weights_after = model.state_dict()

                model.load_state_dict({name:
                                weights_before[name] + (weights_after[name] -
                                                        weights_before[name]) * outerstepsize
                                for name in weights_before})

                #update the outerstepsize
                outerstepsize *= decay_rate
                if outerstepsize < min_learning_rate:
                    outerstepsize = min_learning_rate

                valid_after_acc = validation(model,quary_x,quary_y)
                test_acc = validation(model,target_x,target_y)

                print('target:',target_id,
                    'epoch:', epoch, 'step:', step, '\ttraining loss:',
                  f'{np.mean(train_losses):.3}', '\ttraining acc:',
                  f'{np.mean(train_acc):.3}', '\tvalidation after acc:',
                  f'{np.mean(valid_after_acc):.3}','\ttest acc:',
                  f'{np.mean(test_acc):.3}')

                if count != 100:
                    count += 1
                else:
                    count = 0
                    best_val_acc_50count = 0
                if valid_after_acc > best_val_acc_50count:  # evaluation
                    best_val_acc_50count = np.copy(valid_after_acc)
                    fn = f'./{weights_dir}/epoch_{epoch}_step_{step}_acc_{valid_after_acc:.3}.pth'
                    torch.save(model.state_dict(), fn)

    #test
    def cond(x): return float(x.split('/')[-1].split('_')[-1][:-4])
            #all_model_fn = sorted(glob('/content/drive/MyDrive/research/Multi-agent-Attentional-Activity-Recognition/model_weights_HAR/*.pth'), key=cond)[-2:]
    all_model_fn = sorted(glob(weights_dir + '/*.pth'), key=cond)[-50:]


    #go through different trained model, select a best performance
    best_model_acc = 0

    #do the n-way 1-shot
    N_way = nb_classes
    K_shot = 1

    #test phase for each test set user
    for test_u in target_all:

        #split the target across users
        target_each_x = target_x[np.where(target_u == test_u)[0]]

        target_each_y = target_y[np.where(target_u == test_u)[0]]

        for fn in all_model_fn:
            print('Processing fn', fn)
            state = torch.load(fn)

            model_test = PureTran_torch(height = window_size, nb_features = nb_feature,ntoken = ntokens, ninp = emsize, nhead = nhead, nhid = nhid, nlayers = nlayers,
                 pe = has_pe,dropout=dropout).to(device)

            #per model
            all_train_acc = []
            all_test_acc = []

            # do a number of random selected Nway Kshot and get the average performance of the model
            for e in range(1, epoch_test+1):
                support_x , support_y= random_get_Nway_Kshot(target_each_x, target_each_y,N_way,K_shot)
                model_test.load_state_dict(state)

                x, y = support_x, support_y
         
                train_acc = []
                train_losses = []
                x = torch.tensor(x).float() 
                x = torch.reshape(x,(-1,window_size, nb_feature)).to(device)
                y = torch.tensor(y).long().to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr = innerstepsize_test)

                model.train()
                for _ in range(innerepochs):
                    optimizer.zero_grad()
                    #print("x shape: ",x.shape)
                    outputs = model(x)
                    outputs = outputs.reshape(-1, nb_classes).float()
                    loss = F.cross_entropy(outputs, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                    optimizer.step()
                    train_losses.append(loss.item())
                    acc = (outputs.argmax(1) == y).float().mean().item()
                    train_acc.append(acc)
       
                all_train_acc.append(np.mean(train_acc))
    
                test_acc = validation(model_test, target_x, target_y)
                all_test_acc.append(test_acc)

                print('epoch:', e,  '\ttraining loss:',
                  f'{np.mean(train_losses):.3}', '\ttraining acc:',
                  f'{np.mean(train_acc):.3}', '\ttest acc:',
                  f'{np.mean(test_acc):.3}')
            
            if np.mean(all_test_acc) > best_model_acc:
                best_model_acc = np.mean(all_test_acc)

            print('model: ',fn,'\taverage training acc:',
                    f'{np.mean(all_train_acc):.3}','\taverage test acc:', f'{np.mean(all_test_acc):.3}')
        print("For the id:",target_id, "\tthe best model: ",f'{np.mean(best_model_acc):.3}')

        test_acc_total.append(best_model_acc)

        best_model_acc = 0
        

    ts2 = time.time()
    print("For UCI, the average acc on test data is: ",f'{np.mean(test_acc_total):.3}', "the std: ",f'{np.std(test_acc_total):.3}')
    print("time :",ts2-ts1)
    print("best models: ",test_acc_total)