import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from multifaced_model import MultiFacedVAE
from single_model import SingleMaskVAE
from model import LLMFineTuner, RegressionHead
from util import *
import random
import time
from model_regression import *


def apply_mask(x, mask_ratio=0.1, mask_dim="seq_length"):
    batch_size, seq_length, embed_dim = x.size()
    if mask_dim == "seq_length":
        mask = torch.rand(batch_size, seq_length, 1, device=x.device) < mask_ratio
        mask = mask.expand(-1, -1, embed_dim)
    elif mask_dim == "embed_dim":
        mask = torch.rand(batch_size, 1, embed_dim, device=x.device) < mask_ratio
        mask = mask.expand(-1, seq_length, -1)
    else:
        raise ValueError(f"Unsupported mask_dim: {mask_dim}. Choose 'seq_length' or 'embed_dim'.")
    masked_x = x.clone()
    masked_x[mask] = 0.0
    return masked_x, mask

class TrafficInstructionDataset(Dataset):
    def __init__(self, embeddings, raw_data):
        self.embeddings = embeddings.cpu()
        self.raw_data = raw_data.cpu()
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        embeddings = self.embeddings[idx].to("cuda")
        raw_data = self.raw_data[idx].to("cuda")
        return {"embedding": embeddings, "raw_data": raw_data}

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    temporal_len = 2
    spatial_points = 50
    input_len = 4
    config = {
        "fine_tune_epochs": 50,
        "target_length": 3,
        "batch_size": 32,
        "learning_rate": 1e-5,
        "max_instruction_length": 40,
    }
    device_ids = [0,1,3,2]
    data_paths = {
        "speed": "../../speed_SZ.csv",
        "inflow": "../../inflow_SZ.csv",
        "demand": "../../demand_SZ.csv",
    }
    dataset, embeddings_multifaced = load_multifaced_embeddings(data_paths, temporal_len, spatial_points, device,device_ids)
    embeddings_single = load_single_embeddings(data_paths, temporal_len, spatial_points, device,device_ids)
    speed_embedding = torch.cat([embeddings_multifaced, embeddings_single[:, 0, :]], dim=-1)
    inflow_embedding = torch.cat([embeddings_multifaced, embeddings_single[:, 1, :]], dim=-1)
    demand_embedding = torch.cat([embeddings_multifaced, embeddings_single[:, 2, :]], dim=-1)
    speed_dataset = TrafficInstructionDataset(
        embeddings=speed_embedding[:, :input_len, :, :],
        raw_data=dataset[:, input_len : input_len + config["target_length"], :, 0, :, :]
    )
    inflow_dataset = TrafficInstructionDataset(
        embeddings=inflow_embedding[:, :input_len, :, :],
        raw_data=dataset[:, input_len : input_len + config["target_length"], :, 1, :, :]
    )
    demand_dataset = TrafficInstructionDataset(
        embeddings=demand_embedding[:, :input_len, :, :],
        raw_data=dataset[:, input_len : input_len + config["target_length"], :, 2, :, :]
    )
    model_name = "meta-llama/Llama-3.2-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    if "<start>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ["<start>", "<end>"]})
    fine_tuner = LLMFineTuner(model_name, model_name, device).to(device)
    regression_head = RegressionHead(hidden_dim=80, target_dim=(10, 10)).to(device)
    recovery_head = RecoveryHead(
                shared_conv1=regression_head.conv1,
                shared_conv2=regression_head.conv2,
                share_attention = regression_head.self_attention,
                llm_embed_dim= 128256
            ).to(device)
    if torch.cuda.device_count() > 1:
        fine_tuner = nn.DataParallel(fine_tuner, device_ids=device_ids)
        regression_head = nn.DataParallel(regression_head, device_ids=device_ids)
        recovery_head = nn.DataParallel(recovery_head, device_ids=device_ids)
    tasks = ["speed", "inflow", "demand"]
    datasets = [speed_dataset, inflow_dataset, demand_dataset]
    for task, task_dataset in zip(tasks, datasets):
        fine_tune_via_recovery(
            task_name=task,
            dataset=task_dataset,
            fine_tuner=fine_tuner,
            regression_head=regression_head,
            recovery_head=recovery_head,
            tokenizer=tokenizer,
            device=device,
            temporal_len=temporal_len,
            spatial_points=spatial_points,
            config=config,
            selected_regions = selected_regions,
            input_len = input_len,
        )



def load_multifaced_embeddings(data_paths, temporal_len, spatial_points, device, device_ids):
    batch_size = 32
    # Load data and move to the specified device
    speed = torch.tensor(loaddata(data_paths['speed'])).to(device)
    inflow = torch.tensor(loaddata(data_paths['inflow'])).to(device)
    demand = torch.tensor(loaddata(data_paths['demand'])).to(device)
    speed, demand, inflow = process_d(speed, demand, inflow)

    # Reshape data to match the embedding dimensions
    speed = speed.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    inflow = inflow.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    demand = demand.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    batch_data = torch.stack([speed, inflow, demand], dim=3)
    batch_data_out = torch.stack([speed, inflow, demand], dim=3)

    dataset = MyDataset(batch_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Obtain multifaced embeddings
    vae = MultiFacedVAE(in_channels=3, hidden_dim=32, latent_dim=64).to(device)
    if torch.cuda.device_count() > 1:
        vae = nn.DataParallel(vae, device_ids=device_ids)
    # Load pre-trained weights for MultiFacedVAE
    model_path = f'../multi_stage_mask/multi_faced_mask/model/multifaced_mask_t{temporal_len}_s{spatial_points}_epoch_1000.pth'
    vae.load_state_dict(torch.load(model_path))
    vae.eval()

    embeddings = []

    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            _, embedding = vae(batch_data)
            embeddings.append(embedding)

    embeddings = torch.cat(embeddings, dim=0)

    return batch_data_out, embeddings


def load_single_embeddings(data_paths, temporal_len, spatial_points, device, device_ids):
    batch_size = 32
    # Load data and move to the specified device
    speed = torch.tensor(loaddata(data_paths['speed'])).to(device)
    inflow = torch.tensor(loaddata(data_paths['inflow'])).to(device)
    demand = torch.tensor(loaddata(data_paths['demand'])).to(device)
    speed, demand, inflow = process_d(speed, demand, inflow)

    # Reshape data to match the embedding dimensions
    speed = speed.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    inflow = inflow.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    demand = demand.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    batch_data = torch.stack([speed, inflow, demand], dim=3)

    dataset = MyDataset(batch_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load MultiFacedVAE model and initialize SingleMaskVAE
    pre_trained_vae = MultiFacedVAE(in_channels=3, hidden_dim=32, latent_dim=64).to(device)

    # Load state_dict and remove "module." prefix if necessary
    pre_trained_model_path = f'../multi_stage_mask/multi_faced_mask/model/multifaced_mask_t{temporal_len}_s{spatial_points}_epoch_1000.pth'
    state_dict = torch.load(pre_trained_model_path)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    pre_trained_vae.load_state_dict(new_state_dict)

    vae = SingleMaskVAE(pre_trained_vae, final_embed_dim=16).to(device)

    if torch.cuda.device_count() > 1:
        vae = nn.DataParallel(vae, device_ids=device_ids)
    # Load pre-trained weights for SingleMaskVAE
    model_path = f'../multi_stage_mask/single_st_mask/model/single_mask_edited_t{temporal_len}_s{spatial_points}_epoch_1000.pth'
    vae.load_state_dict(torch.load(model_path))
    vae.eval()

    embeddings = []

    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            (_, speed_embedding), (_, inflow_embedding), (_, demand_embedding) = vae(batch_data)
            embeddings.append(torch.stack([speed_embedding, inflow_embedding, demand_embedding], dim=1))

    embeddings = torch.cat(embeddings, dim=0)

    return embeddings

if __name__ == "__main__":
    main()
