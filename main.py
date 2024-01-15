import torch
import torch.nn as nn
import mat73
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
import math
from scipy.io import savemat
import os

class B1Dataset():
    def __init__(self,data_path):
        data_dict      = mat73.loadmat(data_path)
        self.x         = data_dict['fieldmaps_prepared']
        self.n_samples = self.x.shape[0]                                               # Number of slices
        self.x_num     = self.x.shape[1]                                               # Number of x samples
        self.y_num     = self.x.shape[2]                                               # Number of y samples
        self.c_num     = self.x.shape[3]                                               # Number of coils
        self.x         = np.stack((self.x.real,self.x.imag),-1)                        # (n_samples, x_num, y_num, c_num, 2)
        self.x         = self.x.transpose((0, 4, 1, 3, 2))                             # (n_samples, 2, x_num, c_num, y_num)
        self.x         = self.x.reshape((self.n_samples, 2, -1, 44), order='F')        # (n_samples, 2, x_num*c_num, y_num) concatenate Tx channels
        self.x         = torch.from_numpy(self.x)                                      # convert to torch tensor
        self.x         = self.x.float()                                                # convert to float
        self.x         = self.x.to(device)                                             # send to GPU
        self.y         = torch.from_numpy(data_dict['indiv_masks']).float().to(device) # convert to torch tensor (individual masks)
        self.univ_lab  = torch.from_numpy(data_dict['univ_mask']).float().to(device)   # convert to torch tensor (universal mask)

    def __getitem__(self,index):
        return self.x[index], self.y[index], index
    def __len__(self):
        return self.n_samples


class ConvNet(nn.Module):
    def __init__(self, dataset):
        # Model definition
        super(ConvNet, self).__init__()
        num_values = math.floor((output_ch2*dataset.x_num*dataset.y_num*dataset.c_num*batch_size_c)/((max_p_x*max_p_x)**2))
        self.conv1 = nn.Conv2d(2             , output_ch1, ker_size)
        self.pool  = nn.MaxPool2d(max_p_x    , max_p_y)
        self.conv2 = nn.Conv2d(output_ch1    , output_ch2, ker_size)
        self.fc1   = nn.Linear(num_values    , fc1_num)
        self.fc2   = nn.Linear(fc1_num       , fc2_num)
        self.fc3   = nn.Linear(fc2_num       , fc3_num, bias=False)

    def forward(self, x):
        # Forward model
        pad_size = math.floor(ker_size/2)
        x = self.pool(F.relu(self.conv1(F.pad(x,(pad_size,pad_size,pad_size,pad_size)))))
        x = self.pool(F.relu(self.conv2(F.pad(x,(pad_size,pad_size,pad_size,pad_size)))))
        x = x.view(-1, x.numel())
        x = F.relu(self.fc1(x))
        x = F.relu6(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Set seeds for reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

# Set address of results
add_name   = ""

# Set seeds for reproducibility
torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
g = torch.Generator()
g.manual_seed(0)

# Hyper-parameters
num_epochs     = 100            # Number of epochs
learning_rate  = 0.001          # Learning rate
batch_size_c   = 1              # Batch size
angle          = 5.7            # Flip angle
output_ch1     = 6              # Number of output channels of the first convolutional layer
output_ch2     = 16             # Number of output channels of the second convolutional layer
ker_size       = 5              # Kernel size
fc1_num        = 120            # Number of neurons in the first fully connected layer
fc2_num        = 84             # Number of neurons in the second fully connected layer
fc3_num        = 16             # Number of neurons in the third fully connected layer
max_p_x        = 2              # Max pooling size in x direction
max_p_y        = 2              # Max pooling size in y direction
exp_num        = 1              # Experiment number
num_workers    = 0              # It should be 0 for Windows machines
save_flag      = False          # Save the results or not

# 0) Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) Load data
data_path  = ".mat"
dataset    = B1Dataset(data_path)

# Split the data into training, validation and testing
train_num  = int(dataset.x.shape[0] * 0.8)              # 80% of the data is for training
valid_num  = int((dataset.x.shape[0] - train_num)*0.5)  # 10% of the data is for validation, 10% for testing
test_num   = dataset.x.shape[0] - valid_num - train_num # 10% of the data is for validation, 10% for testing

train_dataset = torch.utils.data.Subset(dataset, range(train_num)) 
valid_dataset = torch.utils.data.Subset(dataset, range(train_num,train_num+valid_num))
test_dataset  = torch.utils.data.Subset(dataset, range(train_num+valid_num,train_num+valid_num+test_num))
full_dataset  = torch.utils.data.Subset(dataset, range(train_num+valid_num+test_num))

# Create data loaders
train_loader = DataLoader(dataset       = train_dataset,
                        batch_size      = batch_size_c,
                        shuffle         = False,
                        drop_last       = True,
                        worker_init_fn  = seed_worker,
                        num_workers     = num_workers,
                        generator       = g)

valid_loader = DataLoader(dataset       = valid_dataset,
                        batch_size      = batch_size_c,
                        shuffle         = False,
                        drop_last       = True,
                        worker_init_fn  = seed_worker,
                        num_workers     = num_workers,
                        generator       = g)

test_loader  = DataLoader(dataset       = test_dataset,
                        batch_size      = batch_size_c,
                        shuffle         = False,
                        drop_last       = True,
                        worker_init_fn  = seed_worker,
                        num_workers     = num_workers,
                        generator       = g)

full_loader= DataLoader(dataset       = dataset,
                        batch_size      = 1,
                        shuffle         = False,
                        drop_last       = False,
                        worker_init_fn  = seed_worker,
                        num_workers     = num_workers,
                        generator       = g)

# 2) Create Model structure
model     = ConvNet(dataset).to(device) 
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# 3) Start training
# pre-allocate some variables
n_total_steps = math.floor(len(train_dataset)/batch_size_c)

loss_arr       = np.zeros(num_epochs)
loss_arr_valid = np.zeros(num_epochs)

counter       = -1
Aw_abs_old    = np.zeros((dataset.x_num,dataset.y_num))
old_real      = np.zeros((dataset.c_num,1))
old_imag      = np.zeros((dataset.c_num,1))

for epoch in range(num_epochs):
    print ('-----------------------------')
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss training: {loss_arr[epoch-1]:.4f}, Loss validation: {loss_arr_valid[epoch-1]:.4f}')
    print ('-----------------------------')
    for i, (images, labels,i_data) in enumerate(train_loader):
        counter+=1

        # Forward pass
        outputs           = model(images)
        outputs           = torch.view_as_complex(outputs.view(1,dataset.c_num,2))[:,:,None,None]
        images            = torch.view_as_complex(reshape_fortran(images,(batch_size_c, 2, dataset.x_num, dataset.c_num, dataset.y_num)).permute((0,3,2,4,1)).contiguous())
        Aw_abs            = torch.abs(torch.sum(outputs*images,1)).squeeze()*labels.squeeze()*dataset.univ_lab
        b                 = (dataset.univ_lab*angle*math.pi/180)*labels.squeeze()
        N                 = ((dataset.univ_lab*labels.squeeze()).sum())
        loss              = ((((Aw_abs - b).abs())**2)).sum().sqrt()/N.sqrt()
        loss_arr[epoch]  += loss.item()/len(train_dataset)
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()

        # Update
        optimizer.step()
        
    # Validation loss
    for i, (images, labels ,i_data_valid) in enumerate(valid_loader):
        with torch.no_grad():
            outputs           = model(images)
            outputs           = torch.view_as_complex(outputs.view(1,dataset.c_num,2))[:,:,None,None]
            images            = torch.view_as_complex(reshape_fortran(images,(batch_size_c, 2, dataset.x_num, dataset.c_num, dataset.y_num)).permute((0,3,2,4,1)).contiguous())
            
            images = reshape_fortran(images, (batch_size_c, 2, dataset.x_num, dataset.c_num, dataset.y_num))
            images = images.permute((0, 3, 2, 4, 1))
            images = images.contiguous()
            images = torch.view_as_complex(images)
            
            Aw_abs            = torch.abs(torch.sum(outputs*images,1)).squeeze()*labels.squeeze()*dataset.univ_lab
            b                 = (dataset.univ_lab*angle*math.pi/180)*labels.squeeze()
            N                 = ((dataset.univ_lab*labels.squeeze()).sum())
            loss              = ((((Aw_abs - b).abs())**2)).sum().sqrt()/N.sqrt()
            loss_arr_valid[epoch]  += loss.item()/len(valid_dataset)

loss_arr_test  = np.zeros(len(test_dataset))
test_sample_id = np.zeros(len(test_dataset))

with torch.no_grad():
    for i, (images, labels ,i_data_valid) in enumerate(test_loader):
        
        outputs           = model(images)
        outputs           = torch.view_as_complex(outputs.view(1,dataset.c_num,2))[:,:,None,None]
        images            = torch.view_as_complex(reshape_fortran(images,(batch_size_c, 2, dataset.x_num, dataset.c_num, dataset.y_num)).permute((0,3,2,4,1)).contiguous())
        Aw_abs            = torch.abs(torch.sum(outputs*images,1)).squeeze()*labels.squeeze()*dataset.univ_lab
        b                 = (dataset.univ_lab*angle*math.pi/180)*labels.squeeze()
        N                 = ((dataset.univ_lab*labels.squeeze()).sum())
        loss              = ((((Aw_abs - b).abs())**2)).sum().sqrt()/N.sqrt()
        loss_arr_test[i]  = loss.item()
        test_sample_id[i] = i_data_valid
        

sim_whole_slices = np.sum(np.ascontiguousarray(dataset.x.cpu().detach().numpy().reshape((dataset.n_samples ,2,dataset.c_num,dataset.x_num,dataset.y_num)).transpose((0,2,3,4,1))).view(dtype=np.complex64)*outputs.cpu().detach().numpy()[:,:,:,:,None],axis=1)
mat_file = (np.concatenate((np.abs(sim_whole_slices.squeeze()),dataset.y.cpu().detach().numpy()*angle*math.pi/180),axis=2))

# Save some results to analyze them in MATLAB
subset_indices    = [0] # select your indices here as a list
subset            = torch.utils.data.Subset(train_dataset, subset_indices)
testloader_subset = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)
inputs            = next(iter(testloader_subset))

full_images = np.zeros((dataset.x_num,dataset.y_num,len(dataset)))
loss_full   = np.zeros(len(dataset))
output_full = np.zeros((dataset.c_num,len(dataset)),dtype=np.complex64)

# Test the model for all slices

with torch.no_grad():
    for i, (images, labels ,i_data_valid) in enumerate(full_loader):
        outputs               = model(images)
        outputs               = torch.view_as_complex(outputs.view(1,dataset.c_num,2))[:,:,None,None]
        images                = torch.view_as_complex(reshape_fortran(images,(batch_size_c, 2, dataset.x_num, dataset.c_num, dataset.y_num)).permute((0,3,2,4,1)).contiguous())
        Aw_abs                = torch.abs(torch.sum(outputs*images,1)).squeeze()*labels.squeeze()*dataset.univ_lab
        b                     = (dataset.univ_lab*angle*math.pi/180)*labels.squeeze()
        N                     = ((dataset.univ_lab*labels.squeeze()).sum())
        loss                  = ((((Aw_abs - b).abs())**2)).sum().sqrt()/N.sqrt()
        full_images[:,:,i]    = Aw_abs.cpu().numpy()
        loss_full[i]          = loss.item()
        output_full[:,i]      = outputs.cpu().numpy().squeeze()


# Save the results
mat_path   = os.path.join(add_name, 'results_' + f'exp_{exp_num:04}' +'.mat')
savemat(mat_path,{"loss_arr":loss_arr,
                "loss_arr_valid":loss_arr_valid,
                "loss_arr_test":loss_arr_test,
                "outputs":outputs.cpu().numpy(),
                "test_sample_id":test_sample_id,
                "full_images":full_images,
                "loss_full":loss_full,
                "output_full": output_full})

print(exp_num)