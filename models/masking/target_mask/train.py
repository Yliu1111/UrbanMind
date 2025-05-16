import sys
sys.path.append('../')
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from util import *
from target_model import MultiFacedVAE, SingleMaskVAE

def vae_masked_loss_function(recon_data, original_data, mask):
    recon_loss = nn.MSELoss(reduction='none')(recon_data, original_data)
    masked_recon_loss = (recon_loss * mask).sum() / mask.sum()
    return masked_recon_loss

def single_mask_train(pre_trained_model_path, data_paths, epochs=1000, lr=0.001, batch_size=32, mask_ratios=(1, 1, 2), device=None):

    speed = torch.tensor(loaddata(data_paths['speed'])).to(device)
    inflow = torch.tensor(loaddata(data_paths['inflow'])).to(device)
    demand = torch.tensor(loaddata(data_paths['demand'])).to(device)
    speed, demand, inflow = process_d(speed, demand, inflow)

    speed = speed.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    inflow = inflow.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    demand = demand.reshape(-1, 12, 63, 10, 10)[:, :, :50, :, :]
    batch_data = torch.stack([speed, inflow, demand], dim=3)

    dataset = MyDataset(batch_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    pre_trained_vae = MultiFacedVAE(in_channels=3, hidden_dim=32, latent_dim=64).to(device)


    state_dict = torch.load(pre_trained_model_path)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    pre_trained_vae.load_state_dict(new_state_dict)

    vae = SingleMaskVAE(pre_trained_vae, final_embed_dim=16).to(device)

    if torch.cuda.device_count() > 1:
        vae = nn.DataParallel(vae)

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    temporal_epochs = 25
    spatial_epochs = 25
    agnostic_epochs = 50
    save_dir = "./model"
    os.makedirs(save_dir, exist_ok=True)

    spatial_points = 30
    temporal_hours = 2

    for epoch in range(epochs):
        total_loss = 0
        vae.train()

        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()


            epoch_within_100 = epoch % 100
            if epoch_within_100 < temporal_epochs:
                masked_data, mask = apply_mask_single(batch_data, mask_type="temporal", temporal_hours=temporal_hours, return_mask=True)
            elif epoch_within_100 < temporal_epochs + spatial_epochs:
                masked_data, mask = apply_mask_single(batch_data, mask_type="spatial", spatial_points=spatial_points, return_mask=True)
            else:
                masked_data, mask = apply_mask_single(batch_data, mask_type="agnostic",
                                                      temporal_hours=temporal_hours, spatial_points=spatial_points,return_mask=True)


            (speed_reconstructed, speed_embedding), (inflow_reconstructed, inflow_embedding), (demand_reconstructed, demand_embedding) = vae(masked_data)

            speed_loss = vae_masked_loss_function(speed_reconstructed, batch_data[..., 0:1, :, :], mask[..., 0:1, :, :])
            inflow_loss = vae_masked_loss_function(inflow_reconstructed, batch_data[..., 1:2, :, :],
                                                   mask[..., 1:2, :, :])
            demand_loss = vae_masked_loss_function(demand_reconstructed, batch_data[..., 2:3, :, :],
                                                   mask[..., 2:3, :, :])

            loss = speed_loss + inflow_loss + demand_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


        if (epoch + 1) % 200 == 0:
            torch.save(vae.state_dict(), os.path.join(save_dir, f'single_mask_edited_t{temporal_hours}_s{spatial_points}_epoch_{epoch + 1}.pth'))
        if epoch == epochs - 1:
            final_speed_embedding = speed_embedding
    if final_speed_embedding is not None:
        embedding_mean = final_speed_embedding.mean().item()
        embedding_max = final_speed_embedding.max().item()
        embedding_min = final_speed_embedding.min().item()
        print(f"Final Epoch Embedding - Mean: {embedding_mean:.4f}, Max: {embedding_max:.4f}, Min: {embedding_min:.4f} , shape {final_speed_embedding.shape}")
    return vae



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_paths = {
        'speed': '../../../speed_SZ.csv',
        'inflow': '../../../inflow_SZ.csv',
        'demand': '../../../demand_SZ.csv'
    }

    epochs = 1000
    lr = 0.001
    batch_size = 32
    mask_ratios = (1, 1, 2)
    pre_trained_model_path = "../multi_faced_mask/model/multifaced_mask_t2_s50_epoch_1000.pth"

    trained_model = single_mask_train(pre_trained_model_path, data_paths, epochs, lr, batch_size, mask_ratios, device)





if __name__ == '__main__':
    main()