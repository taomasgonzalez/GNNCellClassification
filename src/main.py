import importlib
import preprocess
import dataloader
import load_data
import torch


if __name__ == '__main__':
    data_dir, img_dir, graph_dir = "dataset/data", "dataset/images", "out/graphs"
    ann_data, histology_imgs = preprocess.main(data_dir, img_dir, graph_dir)

    train_patients, val_patients, test_patients = dataloader.train_val_test_split(ann_data=ann_data, seed=42)

    coo_connections = { patient: data.obsp['spatial_connectivities'].tocoo() for patient, data in ann_data.items() }

    edge_indices = {}

    for patient, coo in coo_connections.items():
        row = torch.from_numpy(coo.row).long()
        col = torch.from_numpy(coo.col).long()
        edge_indices[patient] = torch.stack([row, col], dim=0)

        print(f"{patient}: {edge_indices[patient].shape}")
