import argparse
import dataloader
from load_data import load_ann_data, load_histology
from graph import calculate_adj_matrix, create_graphs
from gene_filtering import prefilter_genes, prefilter_specialgenes
import os
from os.path import join as path_join
import torch


def get_parser():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="../dataset/data",
        help="Path to the directory containing .h5ad files"
    )

    parser.add_argument(
        "--img_dir", "-i",
        type=str,
        default="../dataset/images",
        help="Path to the directory containing images files"
    )

    parser.add_argument(
        "--graph_dir", "-g",
        type=str,
        default="../out/graphs",
        help="Path to the directory where generated graphs will be output to"
    )

    return parser


def prepare_and_save_tensors(ann_data, patients, graph_dir, tensors_dir):

    train_patients, _, _ = patients

    print("Getting coo connections...")
    coo_connections = dataloader.get_coo_connections(ann_data)
    for patient, coo in coo_connections.items():
        print(f"Patient {patient}: {coo.shape}")
    print("...done.\n")

    print("Getting edge indices...")
    edge_indices = dataloader.get_edge_indices(coo_connections)
    for patient, index in edge_indices.items():
        print(f"{patient}: {edge_indices[patient].shape}")
    print("...done.\n")

    print("Normalizing UMI count...")
    normalized_data = dataloader.get_normalized_umi_count(ann_data)
    print("...done.\n")

    print("Reducing dimensionality of data.x via PCA...")
    reduced_data = dataloader.get_pca_reduced(normalized_data, train_patients)
    for data in reduced_data.values():
        print(data.shape)
    print("...done.\n")

    print("Getting edge features...")
    edge_features = dataloader.get_edge_features(ann_data, edge_indices, graph_dir)
    for patient in ann_data.keys():
        print(f"{patient}: {edge_features[patient].shape}")
    print("...done.\n")


    print("Getting Normalized color averages from the histology images...")
    normalized_color_avgs = dataloader.get_normalized_color_avgs(ann_data)
    for patient_id in normalized_color_avgs.keys():
        print(normalized_color_avgs[patient_id].shape)
    print("...done.\n")


    print("Forming the data.x matrix...")
    data_x = dataloader.get_data_x(ann_data, reduced_data, normalized_color_avgs)
    for patient_id in data_x.keys():
        print(data_x[patient_id].shape)
    print("...done.\n")

    print("Creating data.y with brain layer guesses...")
    data_y = dataloader.get_data_y(ann_data)
    for patient_id in data_y.keys():
        print(data_y[patient_id])
    print("...done.\n")

    print("Creating data.pos with with pixel values")
    data_pos = dataloader.get_data_pos(ann_data)
    for patient_id in data_pos.keys():
        print(data_pos[patient_id].shape)
    print("...done.\n")

    os.makedirs(tensors_dir, exist_ok=True)
    torch.save(data_x, path_join(tensors_dir, "data_x.pt"))
    torch.save(edge_indices, path_join(tensors_dir, "edge_indices.pt"))
    torch.save(edge_features, path_join(tensors_dir, "edge_features.pt"))
    torch.save(data_pos, path_join(tensors_dir, "data_pos.pt"))
    torch.save(data_y, path_join(tensors_dir, "data_y.pt"))
    torch.save(patients, path_join(tensors_dir, "patients.pt"))
    
    return data_x, edge_indices, edge_features, data_pos, data_y, patients

def main(data_dir, img_dir, graph_dir):
    files = [file for file in os.listdir(data_dir) if file.endswith(".h5ad")]
    filepaths = [os.path.join(data_dir, file) for file in files]

    # ann_data[k] is the per patient AnnData
    ann_data = load_ann_data(filenames=files, filepaths=filepaths)
    # histology_images[k] per patient is the TIFF array
    histology_imgs = load_histology(ann_data=ann_data, img_dir=img_dir)

    # create_graphs(graph_dir=graph_dir, ann_data=ann_data, histology_imgs=histology_imgs)
    # # ann_data.X, ann_data.obsm["spatial"], histology_images[s]
    # prefilter_genes(ann_data, min_cells=3)
    # prefilter_specialgenes(ann_data)
    return ann_data, histology_imgs


if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()
    main(args.data_dir, args.img_dir, args.graph_dir)
