def predict(network, x):
    output = x
    for layer in network:
        print(f"Layer: {layer.__class__.__name__}, Input: {output.shape}")  # Print input shape
        output = layer.forward(output)
        print(f"Layer: {layer.__class__.__name__}, Output: {output.shape}")  # Print output shape
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward pass
            output = predict(network, x)

            # calculate error
            error += loss(y, output)

            # backward pass
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad)  # No need to pass learning rate here

            # update weights and biases (this part is handled inside the layers' backward method)
            for layer in network:
                if isinstance(layer, Layer_Dense):  # Check if layer is of type Layer_Dense
                    layer.weights -= learning_rate * layer.dweights
                    layer.biases -= learning_rate * layer.dbiases

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
