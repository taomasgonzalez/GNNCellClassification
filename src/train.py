from torch import no_grad
import os
import json
import mlflow
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score


class EarlyStopper:
    def __init__(self, patience=30):
        self.patience = patience
        self.counter = 0
        self.best_score = np.inf

    def should_early_stop(self, curr_score):
        if curr_score > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = curr_score
            self.counter = 0

        return False

    def reset(self):
        self.best_score = np.inf
        self.counter = 0


def start_tracking_experiment(exp_name="BrainLayerClassifier", port="5000", log_dir="../logs"):

    mlflow.set_tracking_uri("http://localhost:" + port)
    mlflow.set_experiment(exp_name)

    return SummaryWriter(log_dir)


def train_one_epoch(model, optimizer, criterion, dataloader, device):
    model.train()

    total_nodes, total_loss = (0, 0)
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index)

        loss = criterion(outputs, batch.y)
        loss.backward()
        optimizer.step()
        # By default, nn.CrossEntropyLoss() computes the mean loss over all samples in the batch
        # To accumulate the total loss, we undo this averaging and then use the total number of
        # training nodes instead.
        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes
    return total_loss / total_nodes


def validate_one_epoch(model, criterion, dataloader, device):
    model.eval()

    num_classes = model.num_classes

    val_metrics = {
        'acc_micro': MulticlassAccuracy(average="micro", num_classes=num_classes),
        'per_class_acc': MulticlassAccuracy(average=None, num_classes=num_classes),
        'f1_micro': MulticlassF1Score(average="micro", num_classes=num_classes),
        'per_class_f1': MulticlassF1Score(average=None, num_classes=num_classes)
    }

    total_nodes, total_loss = (0, 0)
    with no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index)
            loss = criterion(outputs, batch.y)

            for metric in val_metrics.values():
                metric.update(outputs, batch.y)

            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes
        avg_loss = total_loss / total_nodes

        val_metrics = {key: metric.compute() for key, metric in val_metrics.items()}
        return avg_loss, val_metrics


def train_loop(model, optimizer, criterion, scheduler, loaders, device, params, writer):
    train_loader, val_loader, test_loader = loaders
    num_epochs = params["num_epochs"]

    early_stopping_params = params['early_stopping']
    early_stopper = EarlyStopper(
        patience=early_stopping_params['patience'],
    )

    os.makedirs('models', exist_ok=True)

    best_val_acc = 0
    with mlflow.start_run():
        mlflow.log_params(params)

        for epoch in range(1, num_epochs + 1):
            print("epoch: ", epoch)

            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
            val_loss, val_metrics = validate_one_epoch(model, criterion, val_loader, device)

            val_acc = val_metrics['acc_micro'].item()
            per_class_acc = val_metrics['per_class_acc'].tolist()
            per_class_acc = {f"acc_class_{i}": acc for i, acc in enumerate(per_class_acc)}
            val_f1 = val_metrics['f1_micro'].item()
            per_class_f1 = val_metrics['per_class_f1'].tolist()
            per_class_f1 = {f"f1_class_{i}": acc for i, acc in enumerate(per_class_f1)}

            scheduler.step()
            print(f"Epoch {epoch:03d} : Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_acc, step=epoch)
            mlflow.log_metric('val_f1_score', val_f1, step=epoch)
            mlflow.log_metrics(per_class_acc, step=epoch)
            mlflow.log_metrics(per_class_f1, step=epoch)

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                model_path = os.path.join("models", "model.pth")
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path='model')

            metrics = {'val_accuracy': val_acc}

            with open('metrics.json', 'w') as f:
                json.dump(metrics, f)

            if early_stopper.should_early_stop(val_loss):
                mlflow.log_metric('early_stopping_epoch', epoch, step=epoch)
                break


def test(model, criterion, test_loader, device, writer):

    best_model_path = os.path.join("models", "model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_loss, test_metrics = validate_one_epoch(model, criterion, test_loader, device)

    test_acc = test_metrics['acc_micro'].item()
    per_class_acc = test_metrics['per_class_acc'].tolist()
    per_class_acc = {f"acc_class_{i}": acc for i, acc in enumerate(per_class_acc)}
    test_f1 = test_metrics['f1_micro'].item()
    per_class_f1 = test_metrics['per_class_f1'].tolist()
    per_class_f1 = {f"f1_class_{i}": acc for i, acc in enumerate(per_class_f1)}

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    writer.add_scalar('Loss/test', test_loss, 1)
    writer.add_scalar('Accuracy/test', test_acc, 1)

    mlflow.log_metric('test_loss', test_loss, step=1)
    mlflow.log_metric('test_accuracy', test_acc, step=1)
    mlflow.log_metric('test_f1_score', test_f1, step=1)
    mlflow.log_metrics(per_class_acc, step=1)
    mlflow.log_metrics(per_class_f1, step=1)
