
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


#get randomly selected support set, not used
def random_get_support(source_X, source_y, support_size):
    support_x = []  
    support_y = []  

    start = 0

    support_choose = random.sample(range(0, len(source_X)), support_size)

    support_x.extend(source_X[support_choose])
    support_y.extend(source_y[support_choose])

    return np.array(support_x),np.array(support_y)

#get training and validation from each training user, used to compare with meta learning.
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
dataset = 'MH6'
target_id = 3
source_size = 8000
target_size = 2000
training_epochs = 3000

# data process
# window 20, step 10 is best for MHEALTH
window_size = 20
step = 10



# model
g_size = 128  # Size of theta_g^0
l_size = 128  # Size of theta_g^1
glimpse_output_size = 220  # Output size of Glimpse Network
cell_size = 220  # Size of LSTM cell
nb_glimpses = 40  # Number of glimpses: 30
variance = 0.22  # Gaussian variance for Location Network
# M = 1 # Monte Carlo sampling, see Eq(2) (not used)
glimpse_time_down_scale = 8
glimpse_location_down_scale = 8

file_name = datasets_dict[dataset][0]
file_data = sc.loadmat(path + file_name + '.mat')
file_data = file_data[file_name]

nb_subjects = datasets_dict[dataset][1]
nb_classes = datasets_dict[dataset][2]

source_range = list(range(nb_subjects))
source_range.remove(target_id)
#remove more training
rem = [1,2]
# for i in rem: 
#   source_range.remove(i)
target_range = [target_id]

source_data = get_act_data(file_data, source_range)
target_data = get_act_data(file_data, target_range)

user_id = [i for i in range(nb_subjects)]
if dataset == 'UCI':
    print('dataset is UCI')
    source_data = sc.loadmat(path + 'UCI_train_raw' + '.mat')
    source_data = source_data['UCI_train_raw']

    target_data = sc.loadmat(path + 'UCI_test_raw' + '.mat')
    target_data = target_data['UCI_test_raw']
    user_id = [1,3,5,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29.30]
source_x, source_y, source_u = get_act_time_sequences(source_data, window_size, step)
target_x, target_y, target_u = get_act_time_sequences(target_data, window_size, step)

nb_feature = source_x.shape[-1]

# source_X = source_X.reshape([source_X.shape[0], -1])
# target_X = target_X.reshape([target_X.shape[0], -1])

source_x, source_y, source_u = seed_shuffle(source_x, 1), seed_shuffle(source_y, 1), seed_shuffle(source_u, 1)
target_x, target_y, target_u = seed_shuffle(target_x, 1), seed_shuffle(target_y, 1), seed_shuffle(target_u, 1)

source_x, source_y, source_u = source_x[: source_size], source_y[: source_size], source_u[: source_size]
target_x, target_y, target_u = target_x[: target_size], target_y[: target_size], target_u[: target_size]
# print(target_X.shape)

# model
img_height = window_size
img_width = nb_feature

glimpse_width = max(img_width // glimpse_location_down_scale, 1)
glimpse_height = max(img_height // glimpse_time_down_scale, 1)


# ram = PureTrans( img_width=img_width, img_height=img_height,
#                 nb_classes=nb_classes, 
#                  learning_rate=learning_rate, learning_rate_decay_factor=learning_rate_decay_factor,
#                  min_learning_rate=min_learning_rate,
#                  nb_training_batch=source_X.shape[0]//batch_size,max_gradient_norm=max_gradient_norm, is_training=True)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True


weights_dir = './tradition_tran'
os.makedirs(weights_dir, exist_ok=True)



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


# best_acc = 0
# acc_averge = []

# with tf.Session(config=config) as session:
#     session.run(tf.global_variables_initializer())

#     best_acc_per_run = 0

#     for epoch in range(training_epochs):
#        # source_X,source_y = random_get_support(source_X,source_y,source_size)
#         # training process
#         for b in range(source_X.shape[0] // batch_size):
#             source_batch_x = source_X[batch_size * b: batch_size * (b + 1)]
#             source_batch_y = source_y[batch_size * b: batch_size * (b + 1)]

#             # c = session.run(
#             #     ram.init_glimpse,
#             #     feed_dict={ram.img_ph: source_batch_x,
#             #                ram.lbl_ph: source_batch_y,
#             #                })
#             # print('c', c.shape)
#             #
#             # a = session.run(
#             #     ram.init_glimpse_cooperate,
#             #     feed_dict={ram.img_ph: source_batch_x,
#             #                ram.lbl_ph: source_batch_y,
#             #                })
#             # print(a.shape)

#             # simgs_ph, simgs_ph_re, sh_fc1, sconv_2d_1st, sconv_2d_2nd, sconv_2d_flat = session.run(
#             #     [ram.imgs_ph, ram.imgs_ph_re, ram.h_fc1, ram.conv_2d_1st, ram.conv_2d_2nd, ram.conv_2d_flat],
#             #     feed_dict={ram.img_ph: source_batch_x,
#             #                ram.lbl_ph: source_batch_y,
#             #                })
#             # print('\n\n\n cnn shape:\n', simgs_ph.shape, simgs_ph_re.shape, sh_fc1.shape, sconv_2d_1st.shape, sconv_2d_2nd.shape, sconv_2d_flat.shape)

#             _, loss_source_y, accuracy_source_y = session.run(
#                 [ram.train_op, ram.cross_entropy, ram.accuracy],
#                 feed_dict={ram.img_ph: source_batch_x,
#                            ram.lbl_ph: source_batch_y,
#                            })

#         # test

#         loss_target_y, accuracy_target_y, prediction = session.run(
#             [ram.cross_entropy, ram.accuracy, ram.pred],
#             feed_dict={ram.img_ph: target_X,
#                        ram.lbl_ph: target_y,
#                        })
#        # print("pred is:",prediction)
#         print("acc: ",accuracy_target_y)
#         print(loss_target_y)
#         if(accuracy_target_y>0.86):
#           acc_averge.append(accuracy_target_y)
#         if(best_acc < accuracy_target_y):
#           best_acc = accuracy_target_y
#         # confusion_matrix
#         # confusion_matrix = [[0] * nb_classes for _ in range(nb_classes)]
#         # for i in range(target_X.shape[0]):
#         #     confusion_matrix[int(target_y[i])][int(prediction[i])] += 1
#         # for i in range(nb_classes):
#         #     confusion_matrix[i] = [100 * j / sum(confusion_matrix[i]) for j in confusion_matrix[i]]
#         # print('confusion_matrix:')
#         # for i in range(len(confusion_matrix)):
#         #     print(confusion_matrix[i])
# print("average: ",np.mean(acc_averge))
# print("best: ",best_acc)
print("source_x shape: ",source_x.shape)
source_x , source_y, quary_x, quary_y  =get_data_each_user(source_x, source_y, source_u, nb_subjects,user_id, user_except = target_id, training_percent = 0.7)
print("source_x shape: ",np.array(source_x).shape)
source_x = torch.tensor(source_x).float().to(device)
source_y = torch.tensor(source_y).long().to(device)
quary_x = torch.tensor(quary_x).float().to(device)
quary_y = torch.tensor(quary_y).long().to(device)

# training
learning_rate = 5e-4
learning_rate_decay_factor = 0.998
min_learning_rate = 1e-4
max_gradient_norm = 5.0


best_val_acc = 0
epoch = 2000
batch_size = 2000

for epoch in range(1, epoch+1):
    # get a random support set per task every epoch
    #support_x , support_y , quary_x, quary_y = random_get_support_quary(source_X,source_y,support_size,quary_size,task_nb,each_task)
    

    
       
    model.train()
    train_acc = []
    train_losses = []

    for b in range(source_x.shape[0] // batch_size):
        source_batch_x = source_x[batch_size * b: batch_size * (b + 1)]
        source_batch_y = source_y[batch_size * b: batch_size * (b + 1)]

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.zero_grad()
    # starts training over the task
    
    
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
    print('epoch:', epoch, 'step:', step, '\ttraining loss:',
                  f'{np.mean(train_losses):.3}', '\ttraining acc:',
                  f'{np.mean(train_acc):.3}', '\tvalidation acc:',
                  f'{np.mean(valid_after_acc):.3}')
    if valid_after_acc > best_val_acc:  # evaluation
        best_val_acc = np.copy(valid_after_acc)
        fn = f'./{weights_dir}/epoch_{epoch}_acc_{valid_after_acc:.3}.pth'
        torch.save(model.state_dict(), fn)