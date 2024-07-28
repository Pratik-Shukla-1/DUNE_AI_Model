# In[None]:


# In[1]:
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import xarray as xr
from timeit import default_timer
from utilities3 import *
import torchvision.transforms.functional
torch.manual_seed(0)
np.random.seed(0)

# In[2]:
PATH = 'PATH_TO_YOUR_DATAFILE'

ds = xr.open_dataset(PATH, engine='netcdf4')

ds

# In[3]:
class MinMaxScaler(object):
    def __init__(self, x):
        super(MinMaxScaler, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.min = torch.min(x)
        self.max = torch.max(x)
        self.range = self.max - self.min

    def encode(self, x):
        x = (x - self.min) / self.range
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            min_val = self.min
            range_val = self.range
        else:
            if len(self.min.shape) == len(sample_idx[0].shape):
                min_val = self.min[sample_idx]
                range_val = self.range[sample_idx]
            if len(self.min.shape) > len(sample_idx[0].shape):
                min_val = self.min[:, sample_idx]
                range_val = self.range[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * range_val) + min_val
        return x

    def cuda(self):
        self.min = self.min.cuda()
        self.range = self.range.cuda()

    def cpu(self):
        self.min = self.min.cpu()
        self.range = self.range.cpu()

# In[4]:
ds['time'][480:]

# In[5]:
t2m = ds['sst_t2m_anomalies'][480:]
t2m = np.array(t2m)
t2m = torch.tensor(t2m)
t2m.shape

# In[6]:
lsm = ds['lsm'][480:]
lsm = np.array(lsm)
lsm = torch.tensor(lsm)
lsm.shape

# In[7]:
slt = ds['slt'][480:]
slt = np.array(slt)
slt = torch.tensor(slt)
slt.shape

# In[8]:
orography = ds['orography'][480:]
orography = np.array(orography)
orography = torch.tensor(orography)
orography.shape

# In[9]:
tisr = ds['tisr'][480:]
tisr = np.array(tisr)
tisr = torch.tensor(tisr)
tisr.shape

# In[10]:
cvh = ds['cvh'][480:]
cvh = np.array(cvh)
cvh = torch.tensor(cvh)
cvh.shape

# In[11]:
cvl = ds['cvl'][480:]
cvl = np.array(cvl)
cvl = torch.tensor(cvl)
cvl.shape

# In[12]:
t2m = t2m.reshape(528,1,721,1440)
lsm = lsm.reshape(528,1,721,1440)
slt = slt.reshape(528,1,721,1440)
orography = orography.reshape(528,1,721,1440)
tisr = tisr.reshape(528,1,721,1440)
cvh = cvh.reshape(528,1,721,1440)
cvl = cvl.reshape(528,1,721,1440)

# In[13]:
data = torch.cat((t2m, 
                  lsm,
                  slt, 
                  orography, 
                  tisr,
                  cvh,
                  cvl
                  ), dim=1)
                  
data.shape

# In[14]:
data = data[:, :, :720, :]
data.shape

# In[15]:
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# In[16]:
ntrain = 444
nval = 24
ntest = 60

batch_size = 4
learning_rate = 1e-3

epochs = 500

# In[17]:
train_data = data[:ntrain, ...]
val_data = data[468-25:468, ...]
test_data = data[-ntest-1:, ...]
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

# In[18]:
x_train = train_data[:, :, :, :][:-1]
y_train = train_data[:, 0, :, :][1:]
print(x_train.shape)
print(y_train.shape)

# In[19]:
x_t2m_normalizer = MinMaxScaler(x_train[:,0,:,:])

x_train_t2m_normalized = x_t2m_normalizer.encode(x_train[:,0,:,:])
x_train_t2m_normalized = x_train_t2m_normalized.reshape(443,1,720,1440)
print(x_train_t2m_normalized.shape)

# In[20]:
x_lsm_normalizer = MinMaxScaler(x_train[:,1,:,:])

x_train_lsm_normalized = x_lsm_normalizer.encode(x_train[:,1,:,:])
x_train_lsm_normalized = x_train_lsm_normalized.reshape(443,1,720,1440)
x_train_lsm_normalized.shape

# In[21]:
x_slt_normalizer = MinMaxScaler(x_train[:,2,:,:])

x_train_slt_normalized = x_slt_normalizer.encode(x_train[:,2,:,:])
x_train_slt_normalized = x_train_slt_normalized.reshape(443,1,720,1440)
x_train_slt_normalized.shape

# In[22]:
x_orography_normalizer = MinMaxScaler(x_train[:,3,:,:])

x_train_orography_normalized = x_orography_normalizer.encode(x_train[:,3,:,:])
x_train_orography_normalized = x_train_orography_normalized.reshape(443,1,720,1440)
x_train_orography_normalized.shape

# In[23]:
x_tisr_normalizer = MinMaxScaler(x_train[:,4,:,:])

x_train_tisr_normalized = x_tisr_normalizer.encode(x_train[:,4,:,:])
x_train_tisr_normalized = x_train_tisr_normalized.reshape(443,1,720,1440)
x_train_tisr_normalized.shape

# In[24]:
x_cvh_normalizer = MinMaxScaler(x_train[:,5,:,:])

x_train_cvh_normalized = x_cvh_normalizer.encode(x_train[:,5,:,:])
x_train_cvh_normalized = x_train_cvh_normalized.reshape(443,1,720,1440)
x_train_cvh_normalized.shape

# In[25]:
x_cvl_normalizer = MinMaxScaler(x_train[:,6,:,:])

x_train_cvl_normalized = x_cvl_normalizer.encode(x_train[:,6,:,:])
x_train_cvl_normalized = x_train_cvl_normalized.reshape(443,1,720,1440)
x_train_cvl_normalized.shape

# In[26]:
x_train_data_normalized = torch.cat((x_train_t2m_normalized,
                                   x_train_lsm_normalized,
                                   x_train_slt_normalized,
                                   x_train_orography_normalized,
                                   x_train_tisr_normalized,
                                     x_train_cvh_normalized,
                                     x_train_cvl_normalized
                                   ), dim=1)


x_train_data_normalized.shape

# In[27]:
y_t2m_normalizer = MinMaxScaler(y_train[:,:,:])

y_train_t2m_normalized = y_t2m_normalizer.encode(y_train[:,:,:])
y_train_t2m_normalized.shape

# In[28]:
y_train_data_normalized = y_train_t2m_normalized
y_train_data_normalized.shape

# In[29]:
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_data_normalized.to(torch.float32), y_train_data_normalized.to(torch.float64)),batch_size=batch_size, shuffle=False)

# In[None]:


# In[30]:
x_test = test_data[:, :, :, :][:-1]
y_test = test_data[:, 0, :, :][1:]
print(x_test.shape)
print(y_test.shape)

# In[31]:
x_test_t2m_normalized = x_t2m_normalizer.encode(x_test[:,0,:,:])
x_test_t2m_normalized = x_test_t2m_normalized.reshape(60,1,720,1440)

print(x_test_t2m_normalized.shape)

# In[32]:
x_test_lsm_normalized = x_lsm_normalizer.encode(x_test[:,1,:,:])
x_test_lsm_normalized = x_test_lsm_normalized.reshape(60,1,720,1440)

x_test_lsm_normalized.shape

# In[33]:
x_test_slt_normalized = x_slt_normalizer.encode(x_test[:,2,:,:])
x_test_slt_normalized = x_test_slt_normalized.reshape(60,1,720,1440)

x_test_slt_normalized.shape

# In[34]:
x_test_orography_normalized = x_orography_normalizer.encode(x_test[:,3,:,:])
x_test_orography_normalized = x_test_orography_normalized.reshape(60,1,720,1440)

x_test_orography_normalized.shape

# In[35]:
x_test_tisr_normalized = x_tisr_normalizer.encode(x_test[:,4,:,:])
x_test_tisr_normalized = x_test_tisr_normalized.reshape(60,1,720,1440)

x_test_tisr_normalized.shape

# In[36]:
x_test_cvh_normalized = x_cvh_normalizer.encode(x_test[:,5,:,:])
x_test_cvh_normalized = x_test_cvh_normalized.reshape(60,1,720,1440)

x_test_cvh_normalized.shape

# In[37]:
x_test_cvl_normalized = x_cvl_normalizer.encode(x_test[:,6,:,:])
x_test_cvl_normalized = x_test_cvl_normalized.reshape(60,1,720,1440)

x_test_cvl_normalized.shape

# In[38]:
x_test_data_normalized = torch.cat((x_test_t2m_normalized,
                                   x_test_lsm_normalized,
                                   x_test_slt_normalized,
                                   x_test_orography_normalized,
                                   x_test_tisr_normalized,
                                    x_test_cvh_normalized,
                                    x_test_cvl_normalized
                                   ), dim=1)


x_test_data_normalized.shape

# In[39]:
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_data_normalized.to(torch.float32), y_test.to(torch.float32)),batch_size=batch_size, shuffle=False)


# In[40]:
import torch
import torch.nn as nn
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

# In[41]:
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, n=2, ks=3, stride=1, padding='same'):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        
        self.relu = nn.ReLU(inplace=True)
        self.identity = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0, stride=1)
        
        for i in range(1, n + 1):
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                 nn.ReLU(inplace=True),)
            setattr(self, 'conv%d' % i, conv)
            in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            
            # Apply convolution
            x = conv(x)
           
        # Apply residual connection
        i = self.identity(inputs)
        x = x + i
        
        # Apply ReLU after residual connection
        x = self.relu(x)

        return x

# In[42]:
class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2):
        super(unetUp_origin, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size)
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
       
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

# In[43]:
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

class UNet_2Plus(nn.Module):

    def __init__(self, in_channels=7, n_classes=1):
        
        super(UNet_2Plus, self).__init__()
        self.in_channels = in_channels

        filters = [32, 64, 128, 256, 512]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0])
        self.maxpool0 = nn.AvgPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1])
        self.maxpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1]+32, filters[2])
        self.maxpool2 = nn.AvgPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2]+64+32, filters[3])
        self.maxpool3 = nn.AvgPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3]+128+64+32, filters[4])
                
        self.maxpool0 = nn.AvgPool2d(2)
        
        self.maxpool1 = nn.AvgPool2d(2)
        self.maxpool2 = nn.AvgPool2d(4)
        
        self.maxpool3 = nn.AvgPool2d(2)
        self.maxpool4 = nn.AvgPool2d(4)
        self.maxpool5 = nn.AvgPool2d(8)
        
        self.maxpool6 = nn.AvgPool2d(2)
        self.maxpool7 = nn.AvgPool2d(4)
        self.maxpool8 = nn.AvgPool2d(8)
        self.maxpool9 = nn.AvgPool2d(16)
        
        #self.convnew =  unetConv2(96, filters[2])
        
        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0])
        self.up_concat11 = unetUp_origin(filters[2], filters[1])
        self.up_concat21 = unetUp_origin(filters[3], filters[2])
        self.up_concat31 = unetUp_origin(filters[4], filters[3])

        self.up_concat02 = unetUp_origin(filters[1], filters[0], 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], 5)

        #final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        #initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs) #32, 720, 1440
        
        maxpool0 = self.maxpool0(X_00) #32, 360, 720
        X_10 = self.conv10(maxpool0) #64, 360, 720
        
        maxpool1 = self.maxpool1(X_10) #64, 180, 320
        maxpool2 = self.maxpool2(X_00) #32, 180, 320
        I_X_20 = torch.cat([maxpool1,maxpool2],1) #96, 180, 320
        X_20 = self.conv20(I_X_20) #128, 180, 320
        #print(X_20.shape)
        
        maxpool3 = self.maxpool3(X_20) #128, 90, 180
        maxpool4 = self.maxpool4(X_10) #64, 90, 180
        maxpool5 = self.maxpool5(X_00) #32, 90, 180
        I_X_30 = torch.cat([maxpool3,maxpool4,maxpool5],1) #224, 90, 180
        X_30 = self.conv30(I_X_30) #256, 90, 180
        #print(X_30.shape)
    
        maxpool6 = self.maxpool6(X_30) #256, 45, 90
        maxpool7 = self.maxpool7(X_20) #128, 45, 90
        maxpool8 = self.maxpool8(X_10) #64, 45, 90
        maxpool9 = self.maxpool9(X_00) #32, 45, 90
        I_X_40 = torch.cat([maxpool6,maxpool7,maxpool8,maxpool9],1) #480, 90, 180
        X_40 = self.conv40(I_X_40) #512, 90, 180
        
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4
        return final

# In[44]:
model = UNet_2Plus().to('cuda') 
model = torch.nn.parallel.DataParallel(model, device_ids=[0,1])

# In[45]:
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=225)

# In[46]:
#Use sum:
def compute_weighted_RRMSE_Global(prediction, truth):
    prediction = prediction.view(1,-1)
    truth = truth.view(1,-1)
    
    weights_lat = np.cos(np.deg2rad(ds['latitude'][:720]))
    weights_lat = weights_lat.values.reshape(1,720)
    weights_lat = torch.from_numpy(weights_lat)
    weights_lat = weights_lat.repeat_interleave(1440).cuda()
    weights_lat /= weights_lat.mean()
    
    error = prediction - truth
    
    rmse = torch.sum(torch.sqrt(torch.sum((error**2)*weights_lat, 1)/1036800)).cuda()
    return rmse

# In[47]:
lsm.shape

# In[48]:
lsm = lsm[0,0,:720,:]
lsm.shape

# In[49]:
# Specify the path to the best checkpoint you want to load
best_checkpoint_path = '.../YOUR_CHECKPOINT_DIRECTORY'   # Replace with the actual path to your best checkpoint file

# Load the best checkpoint state
checkpoint = torch.load(best_checkpoint_path)

# Load the state_dict of your model from the loaded checkpoint
model.load_state_dict(checkpoint['state_dict'])

# Set the model in evaluation mode
model.eval()
RMS_error_Global = []
pred_global = torch.zeros(y_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_data_normalized.to(torch.float32), y_test.to(torch.float32)), batch_size=1, shuffle=False)

with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)
        out = model(x).reshape(720, 1440)
        out = y_t2m_normalizer.decode(out)
        
        pred_global[index] = out
        test_l2 += compute_weighted_RRMSE_Global(out, y).item()
        RMS_error_Global.append(test_l2)
        print(index, test_l2)
        index = index + 1

# In[50]:
np.mean(RMS_error_Global)


# In[61]:
def compute_weighted_ACC(predictions, truths):
    acc_values = []
    
    #Get the cosine weightings:
    weights_lat = np.cos(np.deg2rad(ds['latitude'][:720]))
    weights_lat = weights_lat.values.reshape(1,720)
    weights_lat = torch.from_numpy(weights_lat)
    weights_lat = weights_lat.repeat_interleave(1440).cuda()
    weights_lat /= weights_lat.mean()
    
    forecast_anomalies = torch.tensor(np.array(predictions.cpu())).reshape(1,-1).cuda()
    truth_anomalies = torch.tensor(np.array(truths.cpu())).reshape(1,-1).cuda()

    ACC = (
                    torch.sum(weights_lat * forecast_anomalies * truth_anomalies)/
                    torch.sqrt(torch.sum(weights_lat*forecast_anomalies**2) * torch.sum(weights_lat*truth_anomalies**2))
             )

    return ACC

# In[62]:
# Specify the path to the best checkpoint you want to load
best_checkpoint_path = '.../YOUR_CHECKPOINT_DIRECTORY'  # Replace with the actual path to your best checkpoint file

# Load the best checkpoint state
checkpoint = torch.load(best_checkpoint_path)

# Load the state_dict of your model from the loaded checkpoint
model.load_state_dict(checkpoint['state_dict'])

# Set the model in evaluation mode
model.eval()
ACC_error_Global = []
pred = torch.zeros(y_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_data_normalized.to(torch.float32), y_test.to(torch.float32)),batch_size=1, shuffle=False)

with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)
        out = model(x).reshape(720, 1440)
        out = y_t2m_normalizer.decode(out).cuda()

        out = out
        y = y
        
        pred[index] = out
        test_l2 += compute_weighted_ACC(out, y).item()
        ACC_error_Global.append(test_l2)
        print(index, test_l2)
        index = index + 1

# In[63]:
np.mean(ACC_error_Global)

