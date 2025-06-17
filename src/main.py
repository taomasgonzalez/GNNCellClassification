import dataloader
import mlflow_server
import model as model_module
from os.path import join as path_join
import preprocess
import subprocess
import torch
import train
import yaml


if __name__ == '__main__':

    dataset_dir = "dataset"
    tensors_dir = path_join(dataset_dir, "tensors")

    try:
        data_x = torch.load(path_join(tensors_dir, "data_x.pt"))
        edge_indices = torch.load(path_join(tensors_dir, "edge_indices.pt"))
        edge_features = torch.load(path_join(tensors_dir, "edge_features.pt"))
        data_pos = torch.load(path_join(tensors_dir, "data_pos.pt"))
        data_y = torch.load(path_join(tensors_dir, "data_y.pt"))
        patients = torch.load(path_join(tensors_dir, "patients.pt"))
        train_patients, val_patients, test_patients = patients
    except FileNotFoundError:
        script_path = path_join(dataset_dir, "getdata.sh")
        process_result = subprocess.run([script_path],  check=True)
        data_dir, img_dir = path_join(dataset_dir, "data"), path_join(dataset_dir, "images")
        graph_dir = path_join("out", "graphs")
        
        ann_data, histology_imgs = preprocess.main(data_dir, img_dir, graph_dir)
        train_patients, val_patients, test_patients = dataloader.train_val_test_split(ann_data=ann_data, seed=42)
        patients = (train_patients, val_patients, test_patients)
        
        tensors = preprocess.prepare_and_save_tensors(ann_data, patients, graph_dir, tensors_dir)
        data_x, edge_indices, edge_features, data_pos, data_y, patients = tensors

    print(f"train_patients: {train_patients}")
    print(f"val_patients: {val_patients}")
    print(f"test_patients: {test_patients}")

    params = yaml.safe_load(open("params.yaml"))['train']

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
