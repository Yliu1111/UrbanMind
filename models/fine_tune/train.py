# This project uses Llama 3.2, a foundational large language model developed by Meta.
# Llama 3.2 is licensed under the Llama 3.2 Community License,
# Copyright © Meta Platforms, Inc. All Rights Reserved.

import os
from model import TrafficInstructionDataset, LLMFineTuner, RegressionHead
from transformers import AutoTokenizer, AutoModelForCausalLM
from multifaced_model import MultiFacedVAE
from single_model import SingleMaskVAE
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from util import *
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import time


def embed_into_instruction_dynamic(tokenized_instruction, input_embeds, fine_tuner, tokenizer, device):

    input_ids = tokenized_instruction["input_ids"].squeeze(0)
    attention_mask = tokenized_instruction["attention_mask"].squeeze(0)
    batch_size, embed_length, embed_dim = input_embeds.shape


    start_token_id = tokenizer.convert_tokens_to_ids("<start>")
    end_token_id = tokenizer.convert_tokens_to_ids("<end>")
    if start_token_id is None or end_token_id is None:
        raise ValueError("Tokenizer does not contain <start> or <end> tokens. Please add them.")

    start_idx = (input_ids == start_token_id).nonzero(as_tuple=True)[0].item()
    end_idx = (input_ids == end_token_id).nonzero(as_tuple=True)[0].item()

    embedding_layer = fine_tuner.module.model.get_input_embeddings()
    embedding_to_hidden = fine_tuner.module.embedding_to_hidden


    prefix_ids = input_ids[:start_idx]
    suffix_ids = input_ids[end_idx + 1:]

    prefix_embeds = embedding_layer(prefix_ids).to(device)
    suffix_embeds = embedding_layer(suffix_ids).to(device)


    input_embeds_transformed = embedding_to_hidden(input_embeds)

    inputs_embeds_list = []
    attention_mask_list = []

    for i in range(batch_size):

        inputs_embeds = torch.cat([prefix_embeds, input_embeds_transformed[i], suffix_embeds], dim=0)


        attention_mask_dynamic = torch.cat([
            attention_mask[:start_idx],
            torch.ones(embed_length, device=device),
            attention_mask[end_idx + 1:]
        ])

        inputs_embeds_list.append(inputs_embeds)
        attention_mask_list.append(attention_mask_dynamic)


    inputs_embeds = torch.nn.utils.rnn.pad_sequence(inputs_embeds_list, batch_first=True, padding_value=0.0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    return inputs_embeds, attention_mask


def train_task(task_name, dataset, fine_tuner, regression_head, tokenizer, device, temporal_len, spatial_points, config,
               input_len):
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    optimizer = optim.Adam([
        {'params': fine_tuner.parameters(), 'lr': 1e-5},
        {'params': regression_head.parameters(), 'lr': 1e-4}
    ])
    criterion = nn.MSELoss()

    for epoch in range(config["epochs"]):
        fine_tuner.train()
        regression_head.train()
        total_loss = 0
        start_time = time.time()

        for batch in dataloader:
            embeddings = batch["embedding"].to(device)
            raw_data = batch["raw_data"].to(device)


            regions = list(range(embeddings.shape[2]))
            random.shuffle(regions)

            optimizer.zero_grad()

            for region in regions:

                input_embeds = embeddings[:, :input_len, region, :]
                target = raw_data[:, -config["target_length"]:, region, :, :]


                instruction = (
                    f"Given the historical {input_len} hours data: <start> <end> "
                    f"in traffic {task_name}, predict the next {config['target_length']} hours."
                )
                tokenized_instruction = tokenizer(
                    instruction,
                    return_tensors="pt",
                    max_length=config["max_instruction_length"],
                    truncation=True,
                    padding="max_length",
                ).to(device)


                inputs_embeds, updated_attention_mask = embed_into_instruction_dynamic(
                    tokenized_instruction, input_embeds, fine_tuner, tokenizer, device
                )


                llm_output = fine_tuner(inputs_embeds,
                                        updated_attention_mask)  # [batch_size, target_length, hidden_dim]


                prediction = regression_head(llm_output, config["target_length"])  # [batch_size, target_length, 10, 10]

                loss = criterion(prediction, target)  # [batch_size, target_length, 10, 10]

                loss.backward()

                total_loss += loss.item()


            optimizer.step()
        end_time = time.time()

        print(
            f"Task: {task_name}, Epoch {epoch + 1}/{config['epochs']}, time: {end_time - start_time}s, Loss: {total_loss / (50 * len(dataloader))}")


        if (epoch + 1) % 100 == 0:
            fine_tuner_path = f"./modelx/{task_name}_fine_tuner_target_{config['target_length']}_attn_t{temporal_len}_s{spatial_points}_epoch_{epoch + 1}.pt"
            regression_head_path = f"./modelx/{task_name}_regression_head_target_{config['target_length']}_attn_t{temporal_len}_s{spatial_points}_epoch_{epoch + 1}.pt"
            torch.save(fine_tuner.state_dict(), fine_tuner_path)
            torch.save(regression_head.state_dict(), regression_head_path)

    print(f"Saved fine-tuned model and regression head for task: {task_name}")



def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load data
    data_paths = {
        "speed": "../../speed_SZ.csv",
        "inflow": "../../inflow_SZ.csv",
        "demand": "../../demand_SZ.csv",
    }
    device_ids = [1, 0, 2, 3]
    input_len = 8
    temporal_len = 2
    spatial_points = 50

    dataset, embeddings_multifaced = load_multifaced_embeddings(data_paths,temporal_len, spatial_points, device,device_ids)
    embeddings_single = load_single_embeddings(data_paths, temporal_len, spatial_points, device,device_ids)

    speed_embedding = torch.cat([embeddings_multifaced, embeddings_single[:, 0, :]], dim=-1)  # Speed embedding
    inflow_embedding = torch.cat([embeddings_multifaced, embeddings_single[:, 1, :]], dim=-1)  # Inflow embedding
    demand_embedding = torch.cat([embeddings_multifaced, embeddings_single[:, 2, :]], dim=-1)  # Demand embedding

    model_name = "meta-llama/Llama-3.2-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    if "<start>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ["<start>", "<end>"]})

    config = {
        "epochs": 100,
        "target_length": 3,
        "batch_size": 32,
        "learning_rate": 1e-5,
        "max_instruction_length": 40,
    }

    speed_dataset = TrafficInstructionDataset(
        embeddings=speed_embedding[:40, :input_len, :, :],
        raw_data=dataset[:40, input_len : input_len + config["target_length"], :, 0, :, :],  # Speed raw_data
    )

    inflow_dataset = TrafficInstructionDataset(
        embeddings=inflow_embedding[:30, :input_len, :, :],
        raw_data=dataset[:30, input_len : input_len + config["target_length"], :, 1, :, :],  # Inflow raw_data
    )

    demand_dataset = TrafficInstructionDataset(
        embeddings=demand_embedding[:30, :input_len, :, :],
        raw_data=dataset[:30, input_len : input_len + config["target_length"], :, 2, :, :],  # Demand raw_data
    )


    for task, task_dataset in zip(["speed", "inflow", "demand"], [speed_dataset, inflow_dataset, demand_dataset]):
        fine_tuner = LLMFineTuner(model_name, model_name, device)
        regression_head = RegressionHead(hidden_dim=80, target_dim=(10, 10)).to(device)

        if torch.cuda.device_count() > 1:
            fine_tuner = nn.DataParallel(fine_tuner, device_ids=device_ids )
            regression_head = nn.DataParallel(regression_head, device_ids=device_ids)

        train_task(task, task_dataset, fine_tuner, regression_head, tokenizer, device, temporal_len, spatial_points,
                   config, input_len)



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





class TrafficDataset(Dataset):
    def __init__(self, embeddings, raw_data):
        """
        Args:
            embeddings: Tensor of shape [dataset_size, input_length, num_regions, embed_dim].
            raw_data: Tensor of shape [dataset_size, input_length, num_regions, grid_dim, grid_dim].
        """
        self.embeddings = embeddings
        self.raw_data = raw_data

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
            embeddings: Tensor of shape [input_length, num_regions, embed_dim].
            raw_data: Tensor of shape [input_length, num_regions, grid_dim, grid_dim].
        """
        return self.embeddings[idx], self.raw_data[idx]


if __name__ == '__main__':
    main()
