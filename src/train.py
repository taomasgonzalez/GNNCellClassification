from torch import no_grad


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
