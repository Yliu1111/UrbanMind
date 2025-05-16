import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import nn

# Load data from CSV file
def loaddata(data_name):
    """
    Load data from CSV file and reshape to (days, regions, width, height).
    """
    path = data_name
    data = pd.read_csv(path, header=None)
    return data.values.reshape(-1, 63, 10, 10)

# Normalize data to range [-1, 1]
def normalize(data):
    """
    Normalize data to the range [-1, 1].
    """
    min_val = torch.min(data)
    max_val = torch.max(data)

    if max_val - min_val == 0:
        return torch.zeros_like(data)

    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data

# Process speed, inflow, and demand data with thresholds
def process_d(speed, demand, inflow):
    """
    Process speed, demand, and inflow data using thresholding and normalization.
    """
    demand_threshold = torch.quantile(demand, 0.9)
    inflow_threshold = torch.quantile(inflow, 0.9)

    speed = torch.clamp(speed, max=140)  # Speed thresholding
    demand = torch.clamp(demand, max=demand_threshold)  # Demand thresholding
    inflow = torch.clamp(inflow, max=inflow_threshold)  # Inflow thresholding

    # Normalize each data type
    normalized_speed = normalize(speed)
    normalized_demand = normalize(demand)
    normalized_inflow = normalize(inflow)

    # Reshape for concatenation later
    res_speed = normalized_speed.unsqueeze(1).reshape(-1, 63, 100).float()
    res_demand = normalized_demand.unsqueeze(1).reshape(-1, 63, 100).float()
    res_inflow = normalized_inflow.unsqueeze(1).reshape(-1, 63, 100).float()

    return res_speed, res_demand, res_inflow

def process_small(speed, demand, inflow):
    """
    Process speed, demand, and inflow data using thresholding and normalization.
    """
    demand_threshold = torch.quantile(demand, 0.9)
    inflow_threshold = torch.quantile(inflow, 0.9)

    speed = torch.clamp(speed, max=140)  # Speed thresholding
    demand = torch.clamp(demand, max=demand_threshold)  # Demand thresholding
    inflow = torch.clamp(inflow, max=inflow_threshold)  # Inflow thresholding

    # Normalize each data type
    normalized_speed = normalize(speed).float()
    normalized_demand = normalize(demand).float()
    normalized_inflow = normalize(inflow).float()

    return normalized_speed, normalized_demand, normalized_inflow
# Custom Dataset class
class MyDataset(Dataset):
    """
    Dataset class for speed, inflow, and demand data.
    """
    def __init__(self, speed):
        self.speed = speed

    def __len__(self):
        return len(self.speed)

    def __getitem__(self, index):
        return self.speed[index]


def generate_mask(batch_size, num_hours, mask_num):

    mask = torch.ones(batch_size, num_hours)
    zero_indices = torch.randperm(num_hours)[:mask_num]
    mask[:, zero_indices] = 0
    return mask




def apply_mask(data, mask_type="temporal", num_channels=3, spatial_points=50, temporal_hours=2, return_mask=False):

    D, H, R, C, W, Ht = data.shape
    masked_data = data.clone()
    mask = torch.ones_like(data)

    if mask_type == "temporal":

        masked_channel = np.random.randint(0, num_channels)

        mask_time_steps = np.random.choice(H, temporal_hours, replace=False)

        masked_data[:, mask_time_steps, :, masked_channel, :, :] = 0
        mask[:, mask_time_steps, :, masked_channel, :, :] = 0

    elif mask_type == "spatial":

        masked_channel = np.random.randint(0, num_channels)

        unique_points = set()
        while len(unique_points) < spatial_points:
            w, h = np.random.randint(0, W), np.random.randint(0, Ht)
            if (w, h) not in unique_points:
                unique_points.add((w, h))
                masked_data[:, :, :, masked_channel, w, h] = 0
                mask[:, :, :, masked_channel, w, h] = 0

    elif mask_type == "agnostic":

        if np.random.random() > 0.5:
            masked_data, mask = apply_mask(masked_data, mask_type="temporal", num_channels=num_channels,
                                           spatial_points=spatial_points, temporal_hours=temporal_hours, return_mask=True)
        else:
            masked_data, mask = apply_mask(masked_data, mask_type="spatial", num_channels=num_channels,
                                           spatial_points=spatial_points, temporal_hours=temporal_hours, return_mask=True)

    if return_mask:
        return masked_data, mask
    return masked_data



def apply_mask_single(data, mask_type="temporal", temporal_hours=2, spatial_points=50, return_mask=False):

    batch_size, hours, regions, channels, width, height = data.shape
    masked_data = data.clone()
    mask = torch.ones_like(data)

    for channel in range(channels):
        if mask_type == "temporal":

            mask_time_steps = np.random.choice(hours, temporal_hours, replace=False)
            masked_data[:, mask_time_steps, :, channel, :, :] = 0
            mask[:, mask_time_steps, :, channel, :, :] = 0

        elif mask_type == "spatial":

            unique_points = set()
            while len(unique_points) < spatial_points:
                w, h = np.random.randint(0, width), np.random.randint(0, height)
                if (w, h) not in unique_points:
                    unique_points.add((w, h))
                    masked_data[:, :, :, channel, w, h] = 0
                    mask[:, :, :, channel, w, h] = 0

        elif mask_type == "agnostic":

            if np.random.random() > 0.5:
                temp_masked, temp_mask = apply_mask_single(
                    data[:, :, :, channel, :, :].unsqueeze(3),
                    mask_type="temporal",
                    temporal_hours=temporal_hours,
                    return_mask=True
                )
                masked_data[:, :, :, channel, :, :] = temp_masked.squeeze(3)
                mask[:, :, :, channel, :, :] = temp_mask.squeeze(3)
            else:
                temp_masked, temp_mask = apply_mask_single(
                    data[:, :, :, channel, :, :].unsqueeze(3),
                    mask_type="spatial",
                    spatial_points=spatial_points,
                    return_mask=True
                )
                masked_data[:, :, :, channel, :, :] = temp_masked.squeeze(3)
                mask[:, :, :, channel, :, :] = temp_mask.squeeze(3)

    if return_mask:
        return masked_data, mask
    return masked_data

