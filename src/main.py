from minds.data_loader import get_cifar10_dataloader
from minds.models import LightWeightCNN

def main():
    print("--- Testing the project skeleton ---")

    # 1. Load data
    train_loader, _ = get_cifar10_dataloader(batch_size=4)
    print("Data loaded successfully.")

    # 2. Initialize model
    model = LightWeightCNN()
    print("Model initialized successfully.")

    # 3. Get one batch of data
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

    # 4. Pass the batch through the model
    outputs = model(images)
    print(f"Model output shape: {outputs.shape}") # Should be [batch_size, 10]

    print("--- Skeleton test successful! ---")

if __name__ == '__main__':
    main()