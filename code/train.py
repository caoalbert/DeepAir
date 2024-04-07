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