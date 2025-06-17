import dataloader
import mlflow_server
import model as model_module
from os.path import join as path_join
import preprocess
import subprocess
import train
import yaml


if __name__ == '__main__':

    dataset_dir = "dataset"

    script_path = path_join(dataset_dir, "getdata.sh")
    process_result = subprocess.run([script_path],  check=True)

    data_dir, img_dir = path_join(dataset_dir, "data"), path_join(dataset_dir, "images")
    graph_dir = path_join("out", "graphs")

    ann_data, histology_imgs = preprocess.main(data_dir, img_dir, graph_dir)

    train_patients, val_patients, test_patients = dataloader.train_val_test_split(ann_data=ann_data, seed=42)

    print(f"train_patients: {train_patients}")
    print(f"val_patients: {val_patients}")
    print(f"test_patients: {test_patients}")

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

    params = yaml.safe_load(open("params.yaml"))['train']

    patients = (train_patients, val_patients, test_patients)

    train_loader, val_loader, test_loader = dataloader.get_dataloaders(patients, data_x, \
                                                                       edge_indices, edge_features, \
                                                                       data_pos, data_y, params)

    device, model = model_module.get_model()

    optimizer = model_module.get_optimizer(model, params)
    criterion = model_module.get_criterion()
    scheduler = model_module.get_scheduler(optimizer, params)

    port="5000"
    artifacts_dir = "artifacts"
    pid_file_path = "mlflow.pid"
    log_dir="logs"
    experiment_name = "BrainLayerClassifier"

    mlflow_server.start_mlflow_server(port=port, artifacts_dir=artifacts_dir, pid_file_path=pid_file_path)

    writer = train.start_tracking_experiment(exp_name=experiment_name, port=port, log_dir=log_dir)

    loaders = (train_loader, val_loader)
    train.train_loop(model, optimizer, criterion, scheduler, loaders, device, params, writer)

    mlflow_server.stop_mlflow_server(pid_file_path=pid_file_path)
