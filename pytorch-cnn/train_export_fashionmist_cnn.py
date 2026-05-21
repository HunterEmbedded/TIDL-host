#https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# changed to CNN from MLP to give more complex model that TIDL will optimise to DSP rather than keep on CPU
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from nn_models import CNN

import onnx
from onnx import shape_inference



model_name = "model_cnn"
model_path = f"{model_name}.pth"
onnx_path = f"{model_name}.onnx"
onnx_path_shaped = f"{model_name}_shaped.onnx"


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# pass the dataset to dataloader in 64 sample blocks
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break



# create the model
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


model = CNN().to(device)
print(model)



# For the training set loss function and optimiser
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Do the training
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")



# now save the trained model to model_path
torch.save(model.state_dict(), model_path)


# and then use load it for test
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()

#Sanity test results with a sample 
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    x = x.unsqueeze(0).to(device)   # shape becomes [1, 1, 28, 28]
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


# Now export the trained model as ONNX 
# MNIST/FashionMNIST image shape: N x 1 x 28 x 28
dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)

# Export to ONNX
# Note that for now the legacy ONNX exporter warning can be ignored as we want legacy mode
torch.onnx.export(
    model,
    (dummy_input,),
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamo=False,
)

# now run shape inference and save 
model = onnx.load(onnx_path)
model = shape_inference.infer_shapes(model)
onnx.save(model, onnx_path_shaped)

# The output of this process is file onnx_path_shaped which is f"{model_name}_shaped.onnx

