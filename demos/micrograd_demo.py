import argparse
from micrograd import Value, MLP


def main():
    # Set up command-line arguments for a professional CLI experience
    parser = argparse.ArgumentParser(
        description="Train a micrograd MLP on a toy dataset."
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs (default: 25)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate for gradient descent (default: 0.05)",
    )
    args = parser.parse_args()

    print("--- Initializing Micrograd Demo ---")
    print(f"Config -> Epochs: {args.epochs} | Learning Rate: {args.lr}")

    # 1. Initialize a small Multilayer Perceptron
    model = MLP(3, [4, 4, 1])
    print(f"Model created with {len(model.parameters())} trainable parameters.")

    # 2. Define a toy dataset
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    # 3. Training Loop
    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        # Forward pass: evaluate predictions for each sample
        ypred = [model(x) for x in xs]

        # Loss computation: Mean Squared Error (MSE)
        loss = sum(
            ((y_target - y_predicted) ** 2 for y_target, y_predicted in zip(ys, ypred)),
            0.0,
        )

        # Zero out accumulated gradients from the previous iteration
        model.zero_grad()

        # Backward pass: compute gradients via automatic differentiation
        loss.backward()

        # Gradient descent step: update weights & biases using dynamic learning rate
        for p in model.parameters():
            p.data -= args.lr * p.grad

        print(f"Epoch {epoch:2d} | Loss: {loss.data:.6f}")

    print("\nTraining finished successfully!")

    # 4. Evaluate final predictions
    print("\n--- Final Model Predictions ---")
    for x, y_target in zip(xs, ys):
        prediction = model(x)
        print(
            f"Input: {x} | Target: {y_target:4.1f} | Prediction: {prediction.data:4.4f}"
        )


if __name__ == "__main__":
    main()
