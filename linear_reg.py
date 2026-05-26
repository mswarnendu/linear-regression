import numpy as np
import matplotlib.pyplot as plt


def train(epochs, lr, x, y):
    """
    Allows for the linear regression model to train and adjust the weight (slope) and bias (y-intercept)

    Args:
        - lr: Learning rate, safeguards the adjustment of the weights in the model
        - epochs: Number of epochs, basically the amount of times the model will go over all the x values
        - x: x values for the graph (features)
        - y: y values for the graph (actual values)

    Returns:
        - b: bias, y-intercept of the regression line
        - w: weight, slope of the regression line
        - losses: list of the loss values calculated from every epoch with MSE
    """

    # Random weight, np.random.rand() isn't extremely random so it works
    w = np.random.randn() * 0.01
    b = 0  # Initial bias amount, changes later
    N = len(x)  # Number of examples
    losses = []

    for epoch in range(1, epochs + 1):

        y_pred = w * x + b

        # Standard Mean Squared Error (MSE)
        loss = (1/N) * np.sum((y_pred - y) ** 2)

        gradient_w = (2/N) * np.sum(x * (y_pred - y))  # Gradient for weight
        gradient_b = (2/N) * np.sum((y_pred - y))  # Gradient for bias

        w = w - lr * gradient_w  # Weight adjustment
        b = b - lr * gradient_b  # Bias adjustment

        losses.append(loss)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}/1000 | Loss: {loss:.2f} | Bias: {b:.2f} | Weight: {w:.2f}")

    return w, b, losses


def main():

    EPOCHS = 1000  # 1000 epochs for the train() function
    LEARNING_RATE = 3e-4  # Learning rate for train()

    x = np.array([1, 4, 6, 8, 12, 13])  # x for the train() function
    y = np.array([2, 6, 9, 10, 14, 16])  # y for the train() function

    if x.shape != y.shape:
        raise ValueError(
            "The number of x and y values do not match. Please recheck your inputs")

    m, b, losses = train(EPOCHS, LEARNING_RATE, x, y)

    estimated_y = []

    epochs = np.linspace(0, EPOCHS, 1000)

    for input in x:
        estimated_y.append(m * input + b)

    estimated_y = np.array(estimated_y)

    print(f"\nEquation {m:.5f}x + {b:.5f}")

    plt.scatter(x, y)
    plt.plot(x, estimated_y)
    plt.title("Linear Regression of Dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    plt.plot(epochs, losses)
    plt.title("Loss over Epochs (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
