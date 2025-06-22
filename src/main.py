import argparse
import dataloader
import mlflow_server
import model as model_module
import os
from os.path import join as path_join
import preprocess
import subprocess
import torch
import train
import yaml



def get_data(dataset_dir):
    script_path = path_join(dataset_dir, "getdata.sh")
    subprocess.run([script_path],  check=True)


def preprocess_data(dataset_dir, graph_dir):
    data_dir, img_dir = path_join(dataset_dir, "data"), path_join(dataset_dir, "images")
    os.makedirs(graph_dir, exist_ok=True)
    preprocess.main(data_dir, img_dir, graph_dir)


def featurize_data(dataset_dir, graph_dir, tensors_dir, params_file):
    data_dir = path_join(dataset_dir, "data")
    files = [file for file in os.listdir(data_dir) if file.endswith(".h5ad")]
    filepaths = [os.path.join(data_dir, file) for file in files]

    ann_data = preprocess.load_ann_data(filenames=files, filepaths=filepaths)

    with open(params_file) as parfile:
        params = yaml.safe_load(parfile)['featurize']

    patients = dataloader.train_val_test_split(ann_data, params['seed'])
    preprocess.prepare_and_save_tensors(ann_data, patients, graph_dir, tensors_dir)


def train_model(tensors_dir, params_file):

    data_x = torch.load(path_join(tensors_dir, "data_x.pt"))
    edge_indices = torch.load(path_join(tensors_dir, "edge_indices.pt"))
    edge_features = torch.load(path_join(tensors_dir, "edge_features.pt"))
    data_pos = torch.load(path_join(tensors_dir, "data_pos.pt"))
    data_y = torch.load(path_join(tensors_dir, "data_y.pt"))
    patients = torch.load(path_join(tensors_dir, "patients.pt"))

    with open(params_file) as parfile:
        all_params = yaml.safe_load(parfile)
        train_params = all_params['train']
        tracking_params = all_params['tracking']
    train_loader, val_loader, _ = dataloader.get_dataloaders(patients, data_x, \
                                                                       edge_indices, edge_features, \
                                                                       data_pos, data_y, train_params)

    device, model = model_module.get_model(train_params)

    optimizer = model_module.get_optimizer(model, train_params)

    # obtain class frequencies
    class_counts = torch.zeros(model.num_classes, dtype=torch.long)
    for batch in train_loader:
        class_counts += torch.bincount(batch.y, minlength=model.num_classes)
    total = class_counts.sum().item()
    class_freqs = class_counts.float() / total
    class_freqs = class_freqs.to(device)

    criterion = model_module.get_criterion(class_freqs, train_params)
    scheduler = model_module.get_scheduler(optimizer, train_params)

    port = str(tracking_params['port'])
    artifacts_dir = tracking_params['artifacts']
    pid_file_path = tracking_params['mlflow_pid']
    log_dir = tracking_params['log_dir']
    experiment_name = tracking_params['experiment_name']

    mlflow_server.start_mlflow_server(port=port, artifacts_dir=artifacts_dir, pid_file_path=pid_file_path)

    writer = train.start_tracking_experiment(exp_name=experiment_name, port=port, log_dir=log_dir)

    loaders = (train_loader, val_loader, None)
    train.train_loop(model, optimizer, criterion, scheduler, loaders, device, train_params, writer)

    mlflow_server.stop_mlflow_server(pid_file_path=pid_file_path)


def test_model(tensors_dir, params_file):
    data_x = torch.load(path_join(tensors_dir, "data_x.pt"))
    edge_indices = torch.load(path_join(tensors_dir, "edge_indices.pt"))
    edge_features = torch.load(path_join(tensors_dir, "edge_features.pt"))
    data_pos = torch.load(path_join(tensors_dir, "data_pos.pt"))
    data_y = torch.load(path_join(tensors_dir, "data_y.pt"))
    patients = torch.load(path_join(tensors_dir, "patients.pt"))

    with open(params_file) as parfile:
        all_params = yaml.safe_load(parfile)
        # test params should be the same as train params
        test_params = all_params['train']
        tracking_params = all_params['tracking']

    train_loader, _, test_loader = dataloader.get_dataloaders(patients, data_x, \
                                                                       edge_indices, edge_features, \
                                                                       data_pos, data_y, test_params)
    device, model = model_module.get_model(test_params)
    class_counts = torch.zeros(model.num_classes, dtype=torch.long)
    for batch in train_loader:
        class_counts += torch.bincount(batch.y, minlength=model.num_classes)
    total = class_counts.sum().item()
    class_freqs = class_counts.float() / total
    class_freqs = class_freqs.to(device)

    criterion = model_module.get_criterion(class_freqs, test_params)
    port = str(tracking_params['port'])
    artifacts_dir = tracking_params['artifacts']
    pid_file_path = tracking_params['mlflow_pid']
    log_dir = tracking_params['log_dir']
    experiment_name = tracking_params['experiment_name']

    mlflow_server.start_mlflow_server(port=port, artifacts_dir=artifacts_dir, pid_file_path=pid_file_path)

    writer = train.start_tracking_experiment(exp_name=experiment_name, port=port, log_dir=log_dir)
    train.test(model, criterion, test_loader, device, writer)

    mlflow_server.stop_mlflow_server(pid_file_path=pid_file_path)


def main():

    parser = argparse.ArgumentParser(description="GNNCellClassification pipeline")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    getdata = subparsers.add_parser("getdata", help="Download raw data")
    getdata.add_argument("--dataset_dir", default="dataset")


    dataset_dir_default = "dataset"
    data_dir_default = "dataset/data"
    graph_dir_default = "out/graphs"
    tensors_dir_default = "dataset/tensors"
    params_file_default = "params.yaml"

    preprocess = subparsers.add_parser("preprocess", help="Build adjacency graphs")
    preprocess.add_argument(
        "--dataset_dir", "-d",
        type=str,
        default=data_dir_default,
        help="Path to the directory containing the raw data"
    )
    preprocess.add_argument(
        "--graph_dir", "-g",
        type=str,
        default=graph_dir_default,
        help="Path to the directory where generated graphs will be output to"
    )

    featurize = subparsers.add_parser("featurize", help="Generate feature tensors")
    featurize.add_argument(
        "--dataset_dir", "-d",
        type=str,
        default=dataset_dir_default,
        help="Path to the directory where raw data is contained"
    )
    featurize.add_argument(
        "--graph_dir", "-g",
        type=str,
        default=graph_dir_default,
        help="Path to the directory containing generated graphs"
    )
    featurize.add_argument(
        "--tensors_dir", "-t",
        type=str,
        default=tensors_dir_default,
        help="Path to the directory where feature tensors will be output to"
    )
    featurize.add_argument(
        "--params_file", "-p",
        type=str,
        default=params_file_default,
        help="Path to yaml file containing the parameters used to obtain the features out of the data"
    )

    train = subparsers.add_parser("train", help="Train model")
    train.add_argument(
        "--params_file", "-p",
        type=str,
        default=params_file_default,
        help="Path to yaml file containing the model's hyperparameters used to train/define the model"
    )
    train.add_argument(
        "--tensors_dir", "-t",
        type=str,
        default=tensors_dir_default,
        help="Path to the directory where feature tensors will be retrieved from to train the model"
    )

    parser.add_argument(
        "--img_dir", "-i",
        type=str,
        default="../dataset/images",
        help="Path to the directory containing images files"
    )

    test = subparsers.add_parser("test", help="Test the model on the test set")
    test.add_argument(
        "--params_file", "-p",
        type=str,
        default=params_file_default,
        help="Path to yaml file containing the model's hyperparameters that were used to train/define the model"
    )
    test.add_argument(
        "--tensors_dir", "-t",
        type=str,
        default=tensors_dir_default,
        help="Path to the directory where feature tensors will be retrieved from to create the Test Dataloader"
    )

    args = parser.parse_args()

    if args.cmd == "getdata":
        get_data(args.dataset_dir)
    if args.cmd == "preprocess":
        preprocess_data(args.dataset_dir, args.graph_dir)
    if args.cmd == "featurize":
        featurize_data(args.dataset_dir, args.graph_dir, args.tensors_dir, args.params_file)
    if args.cmd == "train":
        train_model(args.tensors_dir, args.params_file)
    if args.cmd == "test":
        test_model(args.tensors_dir, args.params_file)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
