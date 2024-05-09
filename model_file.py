# ### Mount Google Drive to Save Model for Future use
#
# from google.colab import drive
#
# drive.mount('/content/drive')
#
# ### Dataset Download and Preprocessing
#
#
# import splitfolders  # or import split_folders
#
# # Split with a ratio.
# # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#
# splitfolders.ratio("images/", output="/content/drive/MyDrive/Food Recognition/Dataset", seed=1337, ratio=(.8, .2),
#                    group_prefix=None)  # default values
#
# # Split val/test with a fixed number of items e.g. 100 for each set.
# # To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
# # splitfolders.fixed("DATASET1/", output="OUTPUT1", seed=1337, fixed=(100, 100), oversample=False, group_prefix=None) # default values
#
# import os
#
# data_dir = '/content/drive/MyDrive/Food Recognition/Dataset'
# print(os.listdir(data_dir))
# classes = os.listdir(data_dir + "/train")
# print(classes)
#
#
#
# classes.sort()
# print(classes)
#
# ### Model Implementation
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from PIL import Image
# import torch
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.models.resnet import resnet50
#
# valid_transforms = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
#
# train_dataset = torchvision.datasets.ImageFolder(data_dir +
# train_transforms = torchvision.transforms.Compose([
#     torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
#     torchvision.transforms.RandomAffine(15),
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.RandomRotation(15),
#     torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])'/train', transform=train_transforms)
# valid_dataset = torchvision.datasets.ImageFolder(data_dir + '/val', transform=valid_transforms)
#
# batch_size = 128
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=2, pin_memory=True)
#
#
# def visualize_images(dataloader):
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     figure, ax = plt.subplots(nrows=10, ncols=10, figsize=(20, 30))
#     classes = list(dataloader.dataset.class_to_idx.keys())
#     img_no = 0
#     for images, labels in dataloader:
#         for i in range(10):
#             for j in range(10):
#                 img = np.array(images[img_no]).transpose(1, 2, 0)
#                 lbl = labels[img_no]
#
#                 ax[i, j].imshow((img * std) + mean)
#                 ax[i, j].set_title(classes[lbl])
#                 ax[i, j].set_axis_off()
#                 img_no += 1
#         break
#
#
# visualize_images(train_loader)
#
# visualize_images(valid_loader)
#
# model = resnet50(pretrained=True)
#
# model
#
# # Freeze first few layers. You can try different values instead of 100
# for i, param in enumerate(model.parameters()):
#     if i < 100:
#         param.requires_grad = False
#
# model.fc = torch.nn.Sequential(
#     torch.nn.Dropout(0.5),
#     torch.nn.Linear(2048, 101)
# )
#
# ### Calculate Learning Rate
#
# !pip3
# install
# torch_lr_finder
# from torch_lr_finder import LRFinder
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
# lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
# lr_finder.range_test(train_loader, end_lr=0.001, num_iter=25)
# lr_finder.plot()
# lr_finder.reset()
#
# ### Model Training
#
# cuda = True
# epochs = 10
# model_name = '/content/drive/MyDrive/Food Recognition/resnet50.pt'
# # model_name = '/content/resnet50_new.pt'
# optimizer = torch.optim.Adam(model.parameters(), lr=4e-5, weight_decay=0.001)
# criterion = torch.nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True)
#
# writer = SummaryWriter()  # For Tensorboard
# early_stop_count = 0
# ES_patience = 5
# best = 0.0
# if cuda:
#     model.cuda()
#
# for epoch in range(epochs):
#
#     # Training
#     model.train()
#     correct = 0
#     train_loss = 0.0
#     tbar = tqdm(train_loader, desc='Training', position=0, leave=True)
#     for i, (inp, lbl) in enumerate(tbar):
#         optimizer.zero_grad()
#         if cuda:
#             inp, lbl = inp.cuda(), lbl.cuda()
#         out = model(inp)
#         loss = criterion(out, lbl)
#         train_loss += loss
#         out = out.argmax(dim=1)
#         correct += (out == lbl).sum().item()
#         loss.backward()
#         optimizer.step()
#         tbar.set_description(
#             f"Epoch: {epoch + 1}, loss: {loss.item():.5f}, acc: {100.0 * correct / ((i + 1) * train_loader.batch_size):.4f}%")
#     train_acc = 100.0 * correct / len(train_loader.dataset)
#     train_loss /= (len(train_loader.dataset) / batch_size)
#
#     # Validation
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         val_loss = 0.0
#         vbar = tqdm(valid_loader, desc='Validation', position=0, leave=True)
#         for i, (inp, lbl) in enumerate(vbar):
#             if cuda:
#                 inp, lbl = inp.cuda(), lbl.cuda()
#             out = model(inp)
#             val_loss += criterion(out, lbl)
#             out = out.argmax(dim=1)
#             correct += (out == lbl).sum().item()
#         val_acc = 100.0 * correct / len(valid_loader.dataset)
#         val_loss /= (len(valid_loader.dataset) / batch_size)
#     print(f'\nEpoch: {epoch + 1}/{epochs}')
#     print(f'Train loss: {train_loss}, Train Accuracy: {train_acc}')
#     print(f'Validation loss: {val_loss}, Validation Accuracy: {val_acc}\n')
#
#     scheduler.step(val_loss)
#
#     # write to tensorboard
#     writer.add_scalar("Loss/train", train_loss, epoch)
#     writer.add_scalar("Loss/val", val_loss, epoch)
#     writer.add_scalar("Accuracy/train", train_acc, epoch)
#     writer.add_scalar("Accuracy/val", val_acc, epoch)
#
#     if val_acc > best:
#         best = val_acc
#         torch.save(model, model_name)
#         early_stop_count = 0
#         print('Accuracy Improved, model saved.\n')
#     else:
#         early_stop_count += 1
#
#     if early_stop_count == ES_patience:
#         print('Early Stopping Initiated...')
#         print(f'Best Accuracy achieved: {best:.2f}% at epoch:{epoch - ES_patience}')
#         print(f'Model saved as {model_name}')
#         break
#     writer.flush()
# # writer.close()
#
# ### Results Analysis
#
# !kill
# 330
# %reload_ext
# tensorboard
# %tensorboard - -logdir
# runs
#
# ### Save and Load Model
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# torch.save(model.state_dict(), "/content/Dataset/model.pth")
#
# model.load_state_dict(torch.load("/content/model (1).pth", map_location='cpu'))
# model.eval()
#
# ### Testing and Evaluation
#
# !pip
# install
# Pillow
#
# from PIL import Image
#
#
# # Process our image
# def process_image(image_path):
#     # Load Image
#     img = Image.open(image_path)
#
#     # Get the dimensions of the image
#     width, height = img.size
#
#     # Resize by keeping the aspect ratio, but changing the dimension
#     # so the shortest size is 255px
#     img = img.resize((255, int(255 * (height / width))) if width < height else (int(255 * (width / height)), 255))
#
#     # Get the dimensions of the new image size
#     width, height = img.size
#
#     # Set the coordinates to do a center crop of 224 x 224
#     left = (width - 224) / 2
#     top = (height - 224) / 2
#     right = (width + 224) / 2
#     bottom = (height + 224) / 2
#     img = img.crop((left, top, right, bottom))
#
#     # Turn image into numpy array
#     img = np.array(img)
#
#     # Make the color channel dimension first instead of last
#     img = img.transpose((2, 0, 1))
#
#     # Make all values between 0 and 1
#     img = img / 255
#
#     # Normalize based on the preset mean and standard deviation
#     img[0] = (img[0] - 0.485) / 0.229
#     img[1] = (img[1] - 0.456) / 0.224
#     img[2] = (img[2] - 0.406) / 0.225
#
#     # Add a fourth dimension to the beginning to indicate batch size
#     img = img[np.newaxis, :]
#
#     # Turn into a torch tensor
#     image = torch.from_numpy(img)
#     image = image.float()
#     return image
#
#
# # Using our model to predict the label
# def predict(image, model):
#     # Pass the image through our model
#     image = image.to('cpu')
#     output = model(image)
#
#     # Reverse the log function in our output
#     output = torch.exp(output)
#
#     # Get the top predicted class, and the output percentage for
#     # that class
#     probs, classes = output.topk(1, dim=1)
#     return 100 if probs.item() > 100 else probs.item(), classes.item()
#
#
# # Show Image
# def show_image(image):
#     # Convert image to numpy
#     image = image.numpy()
#
#     # Un-normalize the image
#     image[0] = image[0] * 0.226 + 0.445
#
#     # Print the image
#     fig = plt.figure(figsize=(25, 4))
#     plt.imshow(np.transpose(image[0], (1, 2, 0)))
#
#
# # Process Image
# # image = process_image("mac.jpeg")
# image = process_image("/content/burger.jpg")
# # Give image to model to predict output
# top_prob, top_class = predict(image, model)
# # Show the image
# show_image(image)
# if top_class in range(0, 102):
#     c = classes[top_class]
# else:
#     c = "non_food"
# # Print the results
# print("The model is ", top_prob, "% certain that the image is ", c)
#
# # Process Image
# # image = process_image("mac.jpeg")
# image = process_image("/content/bird.jpg")
# # Give image to model to predict output
# top_prob, top_class = predict(image, model)
# # Show the image
# show_image(image)
# if top_class in range(0, 102) and top_prob > 50:
#     c = classes[top_class]
# else:
#     c = "non_food"
# # Print the results
# print("The model is ", top_prob, "% certain that the image is ", c)