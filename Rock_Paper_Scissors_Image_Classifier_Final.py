import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os
from sklearn.manifold import TSNE

print('Setting up variables...')
train_transform = transforms.Compose(
    [transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose(
    [transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32
data_dir = 'D:\My_Stuff_U\Year_3\AI\Project_1\Project\database\output'

train_set = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
val_set = datasets.ImageFolder(data_dir + '/val', transform=transform)
test_set = datasets.ImageFolder(data_dir + '/test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True)

GPU_available = torch.cuda.is_available()

classes = ('Paper', 'Rock', 'Scissors')
classes_nums = (0, 1, 2)
colors_per_class = (1, 2, 3)

# function to show an image
def imshow(img, ax=None, normalise=True):
    if ax is None:
            fig, ax = plt.subplots()
    
    if GPU_available:
        npimg = img.cpu().numpy()
    else:
        npimg = img.numpy()

    image = np.transpose(npimg, (1, 2, 0))
    
    if normalise:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

# setup convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 3)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# print 4 random train images to test images have loaded
dataiter = iter(train_loader)
images, labels = dataiter.next()

fid, axes = plt.subplots(figsize=(10,4), ncols=4)
for i in range(4):
    ax = axes[i]
    imshow(images[i], ax=ax)
    ax.set_title(classes[labels[i].item()])
plt.show()

print("Beginning training @ %s" % datetime.now().strftime("%d-%m-%Y@%H-%M-%S"))

model = Net()
if GPU_available:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# store train & validation losses for later visualisation
train_losses, val_losses = [], []
val_loss_min = np.Inf
best_model_dir = ''
best_model_name = ''

num_epoch = 1

for epoch in range(num_epoch):  # loop over the dataset num_epoch times
    train_loss = 0.0
    val_loss = 0.0

    # train model
    iteration = 0
    model.train() # prep model for training
    for data, target in train_loader:
        # send to GPU device
        if GPU_available:
            data = data.cuda()
            target = target.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # record loss
        train_loss += loss.item()*data.size(0)

        iteration += 1
        #print('Training iteration: ' + str(iteration) + ' complete.')
    
    print("Beginning validation @ %s" % datetime.now().strftime("%d-%m-%Y@%H-%M-%S"))
    
    # validate model
    iteration = 0
    model.eval() # prep model for evaluation
    for data, target in val_loader:
        if GPU_available:
            data = data.cuda()
            target = target.cuda()

        outputs = model(data)
        loss = criterion(outputs, target)
        val_loss += loss.item()*data.size(0)

        iteration += 1
        #print('Validation iteration: ' + str(iteration) + ' complete.')

    train_loss = train_loss/len(train_loader.sampler)
    val_loss = val_loss/len(val_loader.sampler)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print('Epoch %d :\nTraining Loss: %.6f\tValidation Loss: %.6f' %
        (epoch + 1, train_loss, val_loss))

    # save model if it is better
    #min_loss_file_r = open('D:\My_Stuff_U\Year_3\AI\Project_1\Project\minloss.txt', 'r')
    #min_loss = float(min_loss_file_r.read())

    #if val_loss < min_loss and val_loss <= val_loss_min:
    if val_loss <= val_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'.format(
        val_loss_min,
        val_loss))

        now = datetime.now()
        datetime_string = now.strftime("%d-%m-%Y@%H-%M-%S")
        datehour_string = now.strftime("%d-%m-%Y@%Hh")
        folder = '\model_save[' + datehour_string + ']'
        if not os.path.exists('D:\My_Stuff_U\Year_3\AI\Project_1\Project\model_saves' + folder):
            os.makedirs('D:\My_Stuff_U\Year_3\AI\Project_1\Project\model_saves' + folder)
        path = 'D:\My_Stuff_U\Year_3\AI\Project_1\Project\model_saves' + folder + '\model[' + datetime_string + '].pt'
        torch.save(model.state_dict(), path)
        print('model[' + datetime_string + '] saved!')
        best_model_dir = path
        best_model_name = 'model[' + datetime_string + ']'

        #min_loss_file_w = open('D:\My_Stuff_U\Year_3\AI\Project_1\Project\minloss.txt', 'w')
        #if min_loss_file_w.write(str(val_loss)) != 0:
            #print('min_loss file updated successfully!')
        val_loss_min = train_loss

print("Finished training @ %s" % datetime.now().strftime("%d-%m-%Y@%H-%M-%S"))

# visualise training and validation losses over time
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

# show 4 random images and see if their predictions match their ground truths
dataiter = iter(test_loader)
images, labels = dataiter.next()

model = Net()
model.load_state_dict(torch.load(best_model_dir))

if GPU_available:
    model = model.cuda()
    images = images.cuda()

outputs = model(images)
_, predicted_tensor = torch.max(outputs, 1)
if GPU_available:
    predicted = np.squeeze(predicted_tensor.cpu().numpy())
else:
    predicted = np.squeeze(predicted_tensor.numpy())

fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for i in range(4):
    ax = axes[i]
    imshow(images[i], ax=ax)
    ax.set_title("{} ({})".format(classes[predicted[i].item()], classes[labels[i].item()]),
                 color=("green" if predicted[i]==labels[i] else "red"))
plt.show()

print("Beginning accuracy test @ %s with %s" % (datetime.now().strftime("%d-%m-%Y@%H-%M-%S"), best_model_name))

# get accuracy of network
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for evaluation

# run test data through model and calculate test loss & test accuracy
for data, target in test_loader:
    if GPU_available:
        data = data.cuda()
        target = target.cuda()

    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)

    _, pred = torch.max(output, 1)
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(3):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 3)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x

model = Net()
model.load_state_dict(torch.load(best_model_dir))

# visualisation of model via T-SNE
print("Beginning T-SNE visualisation process @ %s" % datetime.now().strftime("%d-%m-%Y@%H-%M-%S"))

# remove the final layer to get 500 feature vector as last layer;s output
#model = nn.Sequential(*list(model.children())[:-1])

if GPU_available:
    model.cuda()

dataiter = iter(test_loader)
images, labels = dataiter.next()
if GPU_available:
    images = images.cuda()

features = model.forward(images)
features = np.squeeze(features.cpu().detach().numpy())
labels = labels.numpy()

tsne = TSNE(n_components=2, verbose=1, n_iter=250).fit_transform(features)

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
# for every class, we'll add a scatter plot separately
for label in classes_nums:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == label]

    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    # setup custom colours per label
    if label == 0:
        color = 'r'
    if label == 1:
        color = 'b'
    if label == 2:
        color = 'y'

    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=classes[label])

ax.legend(loc='best')
plt.show()