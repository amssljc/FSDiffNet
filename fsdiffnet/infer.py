from fsdiffnet.model_architecture import FSDiffNet, FSDiffNet_500
from torch.nn import DataParallel
import torch 
import os

from fsdiffnet.utils import vec2mat

def get_model_path(model_name):
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'models', model_name)
    return model_path


def infer_differential_graph(input:torch.Tensor, scale='normal', para_file = 'normal', device =  'cuda:0'):
    """
    Infer the differential graph from the input correlation matrices.

    Args:
        input (torch.Tensor): shape of (B,2,p,p), B is the batch size, p is the number of variables.
        scale (str, optional): model scale to use. Defaults to 'normal'.
        para_file (str, optional): _description_. Defaults to 'default'.
    """    
    if scale == 'normal':
        model = FSDiffNet()
    elif scale == 'large':
        model = FSDiffNet_500()
    if para_file == 'ABIDE':
        assert isinstance(model, FSDiffNet), "scale should be 'normal'."
    if para_file == 'BRCA':
        assert isinstance(model, FSDiffNet_500), "scale should be 'large'."
        
    model = DataParallel(model)
    model_name = para_file + '.pt'
    model_path = get_model_path(model_name)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
        )
    )
    
    device = torch.device(device)  # Specifies the first CUDA device
    model.to(device)  # Moves the model to the CUDA device  

    output = model(input)
    delta = vec2mat(output).squeeze()
    delta = (delta + delta.transpose(-2,-1))/2
    delta = delta.detach().cpu().numpy()

    return delta