from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torchvision
from torchsummary import summary
import numpy as np
from torch.optim.lr_scheduler import StepLR,OneCycleLR
from torch.optim import Adam
from torch.optim import SGD

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device,optimizer, epoch, train_loader):

    model.train() #set  the model  in training mode (which means the model knows to include thing like batchnorm and dropout)
    pbar = tqdm(train_loader)
    correct  = 0
    processed  = 0
    train_acc = []
    train_losses = []
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        # get the inputs
        data = data.to(device)
        target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        #printing training logs
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        train_losses.append(loss.item())

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    return train_losses,train_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    test_acc = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target,reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100*correct/len(test_loader.dataset))
    
    return test_losses,test_acc

def train_test_model(model, trainloader, testloader, EPOCHS=20, lr=0.01, device='cuda',optim='SGD'):
    wrong_prediction_list = []

    if optim == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optim == 'ADAM':
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)      #fallback if nothing given

    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for epoch in range(0, EPOCHS):

        print(f"Epoch {epoch}")

        x,y = train(model, device, train_loader = trainloader, optimizer = optimizer, epoch =  epoch)
        a,b = test(model, device, test_loader = testloader)

        train_losses.append(x)
        test_losses.append(a)
        train_accuracies.append(y)
        test_accuracies.append(b)
    
    model.eval()
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        match = pred.eq(labels.view_as(pred)).to('cpu').numpy()
        for j, i in enumerate(match):
            if(i == False):
                wrong_prediction_list.append((images[j], pred[j].item(), labels[j].item()))

    print(f'Total Number of incorrectly predicted images is {len(wrong_prediction_list)}')
    return model, wrong_prediction_list, train_losses, train_accuracies, test_losses, test_accuracies
