import numpy as np
import torch
from DataLoader import DataLoader
from NeuralNetwork import NeuralNetwork

def main():
    # Load your data into a numpy array
    data = np.loadtxt('data.txt')

    # Initialize the DataLoader with your data and the train/test split ratio
    data_loader = DataLoader(data, 0.8)

    # Define the sequence length for the inputs
    sequence_length = 20

    # Load the data and convert it into tensors
    train_inputs, train_targets, test_inputs, test_targets = data_loader.load_data(sequence_length)

    # Define the number of hidden units in the network
    hidden_units = 128

    # Define the number of output units in the network
    output_units = 2

    # Initialize the NeuralNetwork with the input and output sizes, the number of hidden units and the sequence length
    model = NeuralNetwork(input_size=sequence_length, output_size=output_units, hidden_size=hidden_units, sequence_length=sequence_length)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the network for some number of epochs
    epochs = 100
    for epoch in range(epochs):
        # Zero the gradients
        optimizer.zero_grad()

        # Get the outputs from the network
        outputs = model(train_inputs)

        # Compute the loss
        loss = criterion(outputs, train_targets)

        # Backpropagate the error and update the parameters
        loss.backward()
        optimizer.step()

        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

    # Evaluate the network on the test data
    with torch.no_grad():
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
        print('Test Loss: {:.4f}'.format(test_loss.item()))

    # Get the estimated drift and diffusion coefficients
    drift_coeff, diffusion_coeff = test_outputs[0].tolist()
    print('Estimated Drift Coefficient: {:.4f}'.format(drift_coeff))
    print('Estimated Diffusion Coefficient: {:.4f}'.format(diffusion_coeff))

if __name__ == '__main__':
    main()
