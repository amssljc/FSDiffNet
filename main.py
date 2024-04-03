# =============================================================================
# import modules
# =============================================================================

import gc
import os.path
import pickle
import time
import warnings
from time import sleep

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import wandb
from fsdiffnet.fit import test_part, train_part
from fsdiffnet.generate_data import ExpressionProfiles
from fsdiffnet.model_architecture import (BCELoss_tanh_focal, DeepNet_baseline,
                                    FSDiffNet)
from fsdiffnet.utils import EarlyStopping, transfer_model

warnings.filterwarnings("ignore")
# torch.cuda.set_device(1)
# torch.backends.cudnn.benchmark = True
os.environ["WANDB_MODE"] = "offline"
# =============================================================================
# wandb initialize
# =============================================================================
data_params = {
    "state": "diff",
    "p": [20, 30, 40, 50, 60, 70, 80],
    "n": 70,
    "train_n": 200,  # small number for a fast test.
    "test_n": 100,  # small number for a fast test.
    "repeats": 10,
    "sparsity": [0.1, 0.3],
    "diff_ratio": [0.3, 0.7],
    "net_rand_mode": "BA",
    "diff_mode": "hub",
    "target_type": "float",
    "distirbution": "mixed",
    "flip": False,
    "withdiag": True,  # nonsense when state=="diff"
    "sigma_diag": True,
}
data_params.update(
    {
        "train_batch_size": 1,
        "test_batch_size": 1,
    }
)

wanbd_name = f"FSDiffNet"
notes = "Train a FSDiffNet."

wandb.init(
    project="FSDiffNet",
    tags=[data_params["state"]],
    name=wanbd_name,
    save_code=True,
    notes=notes,
    reinit=True,
)
artifact = wandb.Artifact(
    f"{wanbd_name}", type="everything", description=notes)
config = wandb.config
model_params = {"use_cuda": True, "pretrain": False}
optim_params = {"lr": 1e-3, "epochs": 500,
                "weight_decay": 0.0000, "focal": 0.05}

# config parameters
config.update(data_params)
config.update(model_params)
config.update(optim_params)

# DI for Diagonal target, DS for Diagonal sigma (input)
data_name = f'{config.net_rand_mode}_{config.state}_{config.diff_mode}_{config.distirbution}_p{config.p}n{config.n}_s{config.sparsity}_dr{config.diff_ratio}_r{config.repeats}_N{config.train_n}_{config.target_type}_{"DT" if config.withdiag else "nDT"}_{"DS" if config.sigma_diag else "nDS"}_{"f" if config.flip else "nf"}'

archi_name = "FSDiffNet"
experiment_name = data_name + archi_name
file_path = "./data/" + data_name + ".pkl"


# =============================================================================
# # generate datasets
# =============================================================================

if os.path.exists(file_path):
    print("Datasets already exist!")
    file = open(file_path, "rb")
    [train_data, test_data] = pickle.load(file)
    file.close()
else:
    print("Generating datasets......")
    sleep(0.5)
    train_data = ExpressionProfiles(
        config.p,
        config.n,
        config.train_n,
        config.repeats,
        config.sparsity,
        config.diff_ratio,
        config.net_rand_mode,
        config.diff_mode,
        target_type=config.target_type,
        flip=config.flip,
        withdiag=config.withdiag,
        sigma_diag=config.sigma_diag,
    )
    test_data = ExpressionProfiles(
        config.p,
        config.n,
        config.test_n,
        1,
        config.sparsity,
        config.diff_ratio,
        config.net_rand_mode,
        config.diff_mode,
        target_type=config.target_type,
        flip=config.flip,
        withdiag=config.withdiag,
        sigma_diag=config.sigma_diag,
    )
    print("Datasets have been generated!!!")
    file = open(file_path, "wb")
    pickle.dump([train_data, test_data], file)
    file.close()

use_cuda = config.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}

# =============================================================================
# # construct data loaders
# =============================================================================

train_loader = DataLoader(
    train_data, batch_size=config.train_batch_size, shuffle=True, **kwargs
)
test_loader = DataLoader(
    test_data, batch_size=config.test_batch_size, shuffle=True, **kwargs
)

# =============================================================================
# # construct the network object
# =============================================================================

print("Constructing the network......")
model = FSDiffNet()
model = DataParallel(model).to(device)

# 预训练
if config.pretrain:
    net_pretrain = DeepNet_baseline().to(device)
    net_pretrain = DataParallel(net_pretrain)
    net_pretrain.load_state_dict(torch.load(
        "/fsdiffnet/models/single-condition_model.pt"))
    model = transfer_model(model, net_pretrain)

    del net_pretrain

print("Net has been Constructed!!!")

# =============================================================================
# # training settings
# =============================================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.9, patience=3, min_lr=5e-6
)
loss_func = BCELoss_tanh_focal(config.focal)

wandb.watch(model, log="all")
early_stop = EarlyStopping(
    patience=10,
    verbose=True,
    experiment_name=experiment_name,
    input_shape=(config.train_batch_size, 2, config.p, config.p),
)

# =============================================================================
# # begin training
# =============================================================================

print(f"Training the network on device:{device} ......")
start = time.time()
for epoch in range(1, config.epochs + 1):

    print(
        f"==================== epoch: {epoch}, lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f} ===================="
    )
    train_part(
        config,
        model,
        device,
        train_loader,
        optimizer,
        scheduler,
        loss_func,
        epoch,
        experiment_name,
    )
    loss, aupr = test_part(
        config, model, device, test_loader, loss_func, epoch, experiment_name
    )
    scheduler.step(aupr)
    early_stop(loss, aupr, model, optimizer)
    wandb.log(
        {
            "best_AUPR": early_stop.val_metric_max,
            "lr": optimizer.state_dict()["param_groups"][0]["lr"],
        }
    )
    if early_stop.early_stop:
        print("Early stopping")
        break
    gc.collect()
end = time.time()
training_time = end - start
wandb.log({"training time cost": training_time})

print("Training finished!!!!!!")
wandb.finish()
