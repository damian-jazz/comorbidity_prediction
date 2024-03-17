import numpy as np
import glob
import nibabel as nib
import torch
from torch.utils.data import Dataset

def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = (in_sp[-3] - out_sp[-3]) / 2 
    y_crop = (in_sp[-2] - out_sp[-2]) / 2 
    z_crop = (in_sp[-1] - out_sp[-1]) / 2
    
    x_offset = 1 if x_crop != int(x_crop) else  0
    y_offset = 1 if y_crop != int(y_crop) else  0
    z_offset = 1 if z_crop != int(z_crop) else  0
        
    x_crop = int(x_crop)
    y_crop = int(y_crop)
    z_crop = int(z_crop)

    if nd == 3:
        data_crop = data[x_crop:-(x_crop + x_offset), y_crop:-(y_crop +  y_offset), z_crop:-(z_crop +  z_offset)]
    elif nd == 4:
        data_crop = data[:, x_crop:-(x_crop + x_offset), y_crop:-(y_crop +  y_offset), z_crop:-(z_crop +  z_offset)]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop

class datasetT1(Dataset):
    def __init__(self, ids, labels, modality, source_path, out_shape=(200, 240, 200)):
        self.ids = ids
        self.labels = labels
        self.modality = modality
        self.source_path = source_path
        self.out_shape = out_shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        id = self.ids.iloc[idx]
        label = self.labels.iloc[idx]
        img_path = glob.glob(f"{self.source_path}sub-{id}_*{self.modality}*.nii.gz")[0]
        img_data = nib.load(img_path).get_fdata().astype(np.single)

        # Preprocessing
        img_data = img_data / img_data.mean()
        img_data = img_data[22:222,28:268,1:201]
        #img_data = crop_center(img_data, self.out_shape)

        img_tensor = torch.from_numpy(img_data)
        label_tensor = torch.tensor(label.tolist(), dtype=torch.float32)
        
        return img_tensor.reshape(1, self.out_shape[0], self.out_shape[1], self.out_shape[2]), label_tensor
    