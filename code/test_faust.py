from model import PointNetBasis
from model import PointNetDesc as PointNetDesc
from dataload import Surr12kModelNetDataLoader as DataLoader
import dataload
import torch
import numpy as np
import hdf5storage 
import scipy 

device = torch.device("cpu")
DATA_PATH = './data/FAUST_noise_0.01.mat'

# Loading Models
basis_model = PointNetBasis(k=20, feature_transform=False)
desc_model = PointNetDesc(k=40, feature_transform=False)
checkpoint = torch.load('./models/trained2/basis_model_best.pth')
basis_model.load_state_dict(checkpoint)
checkpoint = torch.load('./models/trained2/desc_model_best.pth')
desc_model.load_state_dict(checkpoint)

basis_model = basis_model.eval()
desc_model = desc_model.eval()

# Loading Data
dd = hdf5storage.loadmat(DATA_PATH)
v = dd['vertices'].astype(np.float32)

# Computing Basis and Descriptors
pred_basis = basis_model(torch.transpose(torch.from_numpy(dd['vertices'].astype(np.float32)),1,2))
pred_desc = desc_model(torch.transpose(torch.from_numpy(dd['vertices'].astype(np.float32)),1,2))

# Save Output
dd['basis'] = np.squeeze(np.asarray(pred_basis[0].detach().numpy()))
dd['desc'] = np.squeeze(np.asarray(pred_desc[0].detach().numpy()))
scipy.io.savemat('./out_FAUST_noise_0.01.mat', dd)
