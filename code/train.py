import torch


def train_model(model, optimizer, loss_fn, num_epochs, graph, device=None):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(graph.series)  # Forward pass
        loss = loss_fn(outputs, graph.y_transformed)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        epoch_loss += loss.item()  # Accumulate the loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")


def train_model_plot(model, optimizer, loss_fn, num_epochs, training_graph, test_graph):
    """
    Train the model and plot the training and test losses
    """
    model.train()
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(training_graph.series)  # Forward pass
        loss = loss_fn(outputs, training_graph.y_transformed)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        train_losses.append(loss.item())  # Store training loss

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_graph.series)
            test_loss = loss_fn(test_outputs, test_graph.y_transformed)
            test_losses.append(test_loss.item())
        model.train()

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Test Loss: {test_loss.item()}")

    return train_losses, test_losses
