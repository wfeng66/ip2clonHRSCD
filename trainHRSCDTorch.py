"""
This script train ip2cl on HRSCD dataset
"""
import os, cv2
import numpy as np
# import pandas as pd
# import random, tqdm
# import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics as smp_metrics
import segmentation_models_pytorch.utils as utils
# import albumentations as album

Image.MAX_IMAGE_PIXELS = 100000000


# DATA_DIR = '/content/gdrive/MyDrive/PhD/SAA/MapFormer'
DATA_DIR = '/home/lab/feng/MapFormer/Data'
# DATA_DIR = 'G://Research/SAA/2D/MapFormer/Data'

x_train_post_dir = os.path.join(DATA_DIR, 'cropped_data/train_post/images')
x_train_pre_dir = os.path.join(DATA_DIR, 'cropped_data/train_pre/images')
y_train_post_dir = os.path.join(DATA_DIR, 'cropped_data/train_post/targets')
y_train_pre_dir = os.path.join(DATA_DIR, 'cropped_data/train_pre/targets')

x_test_post_dir = os.path.join(DATA_DIR, 'cropped_data/test_post/images')
x_test_pre_dir = os.path.join(DATA_DIR, 'cropped_data/test_pre/images')
y_test_post_dir = os.path.join(DATA_DIR, 'cropped_data/test_post/targets')
y_test_pre_dir = os.path.join(DATA_DIR, 'cropped_data/test_pre/targets')


# Get class names
class_names = ['no change', 'change']
# Get class RGB values
class_rgb_values = [[0, 0, 0], [255, 255, 255]]

# Useful to shortlist specific classes in datasets with large number of classes
select_classes = ['no change', 'change']

# Get RGB values of required classes
select_class_rgb_values = np.array(class_rgb_values)

# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

# Perform one hot encoding on label
def one_hot(mask):
    msk = np.zeros((2, mask.shape[0], mask.shape[1]))
    msk[0, :, :] = (mask==0)
    msk[1, :, :] = (mask==1)
    return msk

# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def diff(img1, img2, msk_pre):
    msk_pre3d = np.zeros_like(img1)
    for i in range(3):
        msk_pre3d[:, :, i] = msk_pre[:, :]
    # msk_pre3d = np.stack((msk_pre, msk_pre, msk_pre), -3)
    # return img1 - img2*msk_pre3d          # non-normalization version
    # Normalization version
    Z0 = img1 - img1*msk_pre3d    # background
    Z1 = (img1-img2)*msk_pre3d    # foreground
    return Z1.astype(np.uint8)      # only keep the TOI
    """
    nonzero_values = Z1[Z1 != 0]
    max_value = np.max(nonzero_values)
    min_value = np.min(nonzero_values)
    # Z1_normalized = (Z1 - min_value) / (max_value - min_value)
    Z1_normalized = (Z1 - min_value)/ (max_value - min_value)
    Z1_normalized[Z1 == 0] = 0    # set background to zero
    return ((Z0 + Z1_normalized)*255).astype(np.uint8)
    """


def diff(img1, img2, msk_pre):
    msk_pre3d = np.zeros_like(img1)
    for i in range(3):
        msk_pre3d[:, :, i] = msk_pre[:, :]
    return img1 - img2*msk_pre3d          # non-normalization version






class BuildingsDataset(torch.utils.data.Dataset):

    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_pre_dir,
            images_post_dir,
            masks_pre_dir,
            masks_post_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
            val = False,
            test = False
    ):

        self.image_pre_paths = [os.path.join(images_pre_dir, image_id) for image_id in sorted(os.listdir(images_pre_dir))]
        self.image_post_paths = [os.path.join(images_post_dir, image_id) for image_id in sorted(os.listdir(images_post_dir))]
        self.mask_pre_paths = [os.path.join(masks_pre_dir, image_id) for image_id in sorted(os.listdir(masks_pre_dir))]
        self.mask_post_paths = [os.path.join(masks_post_dir, image_id) for image_id in sorted(os.listdir(masks_post_dir))]

        self.n = len(self.image_pre_paths)

        if val:
          self.image_pre_paths = self.image_pre_paths[:self.n//2]
          self.image_post_paths = self.image_post_paths[:self.n//2]
          self.mask_pre_paths = self.mask_pre_paths[:self.n//2]
          self.mask_post_paths = self.mask_post_paths[:self.n//2]

        if test:
          self.image_pre_paths = self.image_pre_paths[self.n//2:]
          self.image_post_paths = self.image_post_paths[self.n//2:]
          self.mask_pre_paths = self.mask_pre_paths[self.n//2:]
          self.mask_post_paths = self.mask_post_paths[self.n//2:]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read images and masks
        image_pre = cv2.cvtColor(cv2.imread(self.image_pre_paths[i]), cv2.COLOR_BGR2RGB)
        image_post = cv2.cvtColor(cv2.imread(self.image_post_paths[i]), cv2.COLOR_BGR2RGB)
        mask_pre = np.array(Image.open(self.mask_pre_paths[i]), dtype=np.uint8)
        mask_post = np.array(Image.open(self.mask_post_paths[i]), dtype=np.uint8)

        image = diff(image_post, image_pre, mask_pre)           # ip2cl
        # image = np.stack((image_pre, image), axis=-1).reshape((256,256, -1))          # combine image_pre
        # concatenate mask_pre with ip2cl
        mask_pre = np.expand_dims(mask_pre, axis=-1)
        mask_pre = np.repeat(mask_pre, 3, axis=-1)
        image = np.stack((mask_pre, image), axis=-1).reshape((256,256, -1))

        mask_post = one_hot(mask_post)

        trans = T.Compose([T.ToTensor()])

        image = trans(image)
        mask_post = trans(mask_post.transpose((1, 2, 0)))

        return image, mask_post

    def __len__(self):
        # return length of
        return len(self.image_pre_paths)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channnels=3, out_classes=2, up_sample_mode='conv_transpose'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(in_channnels, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x


# Get UNet model
model = UNet(in_channnels=6)        # if only train a standard ip2cl, get rid of the parameter


# Get train and val dataset instances
# For ip2cl
train_dataset = BuildingsDataset(
    x_train_pre_dir, x_train_post_dir, y_train_pre_dir, y_train_post_dir,
    class_rgb_values=select_class_rgb_values,
)

valid_dataset = BuildingsDataset(
    x_test_pre_dir, x_test_post_dir, y_test_pre_dir, y_test_post_dir,
    class_rgb_values=select_class_rgb_values,
    val = True
)

# Get train and val data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)


# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
TRAINING = True

# Set num of epochs
EPOCHS = 200

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.models.segmentation import deeplabv3_resnet101

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss function
# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, input, target):
#         smooth = 1e-6
#         input_flat = input.view(-1)
#         target_flat = target.view(-1)
#         intersection = (input_flat * target_flat).sum()
#         dice = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
#         return 1 - dice

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Calculate Binary Cross-Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # Calculate Focal Loss
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# loss = DiceLoss()
# loss.__name__ = 'DiceLoss'
loss = FocalLoss(alpha=1, gamma=2, reduction='mean')
loss.__name__ = 'FocalLoss'

# Define metrics
from torchmetrics import F1Score, Accuracy, Precision, Recall

# Define metrics
# iou_score = IoU(num_classes=2)  # Assuming binary segmentation
fscore = F1Score(num_classes=2, task="binary")  # Assuming binary classification
accuracy = Accuracy(num_classes=2, task="binary")  # Assuming binary classification
precision = Precision(num_classes=2, task="binary")  # Assuming binary classification
recall = Recall(num_classes=2, task="binary")  # Assuming binary classification

fscore.__name__ = 'F1Score'
accuracy.__name__ = 'Accuracy'
precision.__name__ = 'Precision'
recall.__name__ = 'Recall'


metrics = [
    # iou_score,
    fscore,
    accuracy,
    precision,
    recall
]


# define optimizer
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.00008),
])

# define learning rate scheduler (not used in this NB)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)

# load best saved model checkpoint from previous commit (if present)
if os.path.exists('../input/unet-for-building-segmentation-pytorch/best_model.pth'):
    model = torch.load('../input/unet-for-building-segmentation-pytorch/best_model.pth', map_location=DEVICE)


train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)


if TRAINING:

    best_f1_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)


        # Save model if a better val IoU score is obtained
        if best_f1_score < valid_logs['F1Score']:
            best_f1_score = valid_logs['F1Score']
            torch.save(model, './best_model.pth')
            print('Model saved!')


# load best saved model checkpoint from the current run
if os.path.exists('./best_model.pth'):
    best_model = torch.load('./best_model.pth', map_location=DEVICE)
    print('Loaded UNet model from this run.')

# create test dataloader to be used with UNet model (with preprocessing operation: to_tensor(...))
test_dataset = BuildingsDataset(
    x_test_pre_dir, x_test_post_dir, y_test_pre_dir, y_test_post_dir,
    class_rgb_values=select_class_rgb_values,
    test=True
)

test_dataloader = DataLoader(test_dataset)

test_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

valid_logs = test_epoch.run(test_dataloader)
print("Evaluation on Test Data: ")
for key in valid_logs.keys():
    print(f"{key}: {valid_logs[key]}")