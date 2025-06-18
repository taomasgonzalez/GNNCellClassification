from torch import no_grad
import os
import json
import mlflow
import os
import torch
from torch.utils.tensorboard import SummaryWriter


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

    correct, total_nodes, total_loss = (0, 0, 0)
    with no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index)
            loss = criterion(outputs, batch.y)
            pred = outputs.argmax(dim=1)
            total_loss += loss.item() * batch.num_nodes
            correct += (pred == batch.y).sum().item()
            total_nodes += batch.num_nodes
        avg_loss = total_loss / total_nodes
        avg_accuracy = correct / total_nodes
        return avg_loss, avg_accuracy


def train_loop(model, optimizer, criterion, scheduler, loaders, device, params, writer):
    train_loader, val_loader = loaders
    num_epochs = params["num_epochs"]

    print("loop")
    with mlflow.start_run():
        mlflow.log_params(params)

        for epoch in range(1, num_epochs + 1):
            print("epoch: ", epoch)

            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
            val_loss, val_acc = validate_one_epoch(model, criterion, val_loader, device)
            scheduler.step()
            print(f"Epoch {epoch:03d} : Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_acc, step=epoch)

            os.makedirs('models', exist_ok=True)
            model_path = os.path.join("models", "model.pth")
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path, artifact_path='model')

            metrics = {'val_accuracy': val_acc}
            with open('metrics.json', 'w') as f:
                json.dump(metrics, f)
