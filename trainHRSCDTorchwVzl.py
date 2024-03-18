from typing import Any, Iterator

from sklearn.model_selection import train_test_split
import segmentation_models as sm
import os, sys, time, cv2, glob, easygui, random, time, argparse, datetime, pickle, tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
# from mine import wload, wdump, show_time
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# constant setting
# root_path = '/home/wei/feng/'
ROOT_PATH = 'G:\\Research\\SAA\\2D\\MapFormer'
DATA_DIR = 'G:\\Research\\SAA\\2D\\MapFormer\\Data\\cropped_data'
# AUTOTUNE = tf.data.experimental.AUTOTUNE
Image.MAX_IMAGE_PIXELS = 100000000
IMG_SIZE = 2048
N_CHANNELS = 3
N_CLASSES = 2
SEED = 42
EPOCHS = 100
BATCH_SIZE = 1
BUFFER_SIZE = 1000
OUTPUT_CLASSES = 2
dropout_rate = 0.5
valRate = 0.2
LR = 0.0001
ACTIVATION = None
loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.001, 0.2, 0.3, 0.3, 0.199]))
loss.__name__ = 'CrossEntropy'
# dice_loss = smp.losses.DiceLoss(class_weights=np.ones(N_CLASSES) / N_CLASSES)
# focal_loss = smp.losses.FocalLoss(mode='multilabel')
# total_loss = dice_loss + (2 * focal_loss)
# Get class names
class_names = ['no change', 'change']
# Get class RGB values
class_rgb_values = [[0, 0, 0], [255, 255, 255]]

# Useful to shortlist specific classes in datasets with large number of classes
select_classes = ['no change', 'change']

# Get RGB values of required classes
select_class_rgb_values = np.array(class_rgb_values)

# GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 2,
          'pin_memory': True}

x_train_post_dir = os.path.join(DATA_DIR, 'train_post\\images')
x_train_pre_dir = os.path.join(DATA_DIR, 'train_pre\\images')
y_train_post_dir = os.path.join(DATA_DIR, 'train_post\\targets')
y_train_pre_dir = os.path.join(DATA_DIR, 'train_pre\\targets')

x_test_post_dir = os.path.join(DATA_DIR, 'test_post\\images')
x_test_pre_dir = os.path.join(DATA_DIR, 'test_pre\\images')
y_test_post_dir = os.path.join(DATA_DIR, 'test_post\\targets')
y_test_pre_dir = os.path.join(DATA_DIR, 'test_pre\\targets')

def diff(img1, img2, msk_pre):
    msk_pre3d = np.zeros_like(img1)
    msk_pre[msk_pre>1] = 1                  # make sure there isn't any mask pixels > 1 when involving all object in maks_pre
    for i in range(3):
        msk_pre3d[:, :, i] = msk_pre[:, :]
    # msk_pre3d = np.stack((msk_pre, msk_pre, msk_pre), -3)
    return img1 - img2*msk_pre3d          # non-normalization version


def onehot2msk(onehot, dim):
  """
  Function use to convert one-hot mask to original
  onehot: the mask in one-hot format
  dim:    the axis need to squeeze
  """
  msk = np.argmax(onehot, axis=dim)
  return msk


def one_hot(mask):
    msk = np.zeros((2, mask.shape[0], mask.shape[1]))
    msk[0, :, :] = (mask==0)
    msk[1, :, :] = (mask==1)
    return msk



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
        # image = np.stack((image_pre, image), axis=-1).reshape((IMAGE_SIZE,IMAGE_SIZE, -1))          # combine image_pre
        # concatenate mask_pre with ip2cl
        # mask_pre = np.expand_dims(mask_pre, axis=-1)
        # mask_pre = np.repeat(mask_pre, 3, axis=-1)
        # image = np.stack((mask_pre, image), axis=-1).reshape((IMAGE_SIZE,IMAGE_SIZE, -1))

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


def vizTraining(ep, loss_tr, loss_val, iou_tr, iou_val, path):
    # summarize history for dice
    fig1 = plt.figure(figsize=(12, 8))
    plt.plot(ep, loss_tr)
    plt.plot(ep, loss_val)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    fig1.savefig(os.path.join(path, str(len(ep)) + 'ep_DiceLoss' + time.strftime("%Y%m%d") + '.jpg'))
    plt.close(fig1)
    # summarize history for iou
    fig2 = plt.figure(figsize=(12, 8))
    plt.plot(ep, iou_tr)
    plt.plot(ep, iou_val)
    plt.title('IoU Score')
    plt.ylabel('IoU')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    fig2.savefig(os.path.join(path, str(len(ep)) + 'ep_IoU' + time.strftime("%Y%m%d") + '.jpg'))
    plt.close(fig2)


class KeepAsIs(object):
    def __call__(self, pic):
        return pic


def filter_samples(images, masks):
    images_filtered = []
    masks_filtered = []
    for image, mask in zip(images, masks):
        # Convert PyTorch tensor to NumPy array
        mask_np = mask.cpu().numpy()
        # Check if maximum value of mask is greater than 0
        if np.max(mask_np) > 0:
            images_filtered.append(image)
            masks_filtered.append(mask)
    return images_filtered, masks_filtered


def visualize(images, masks, preds):
    print("viz")
    # Convert tensors to PIL Images
    to_pil = T.ToPILImage()
    images = [to_pil(img) for img in images]
    masks = [torch.argmax(mask, dim=0) for mask in masks]
    preds = [torch.argmax(mask, dim=0) for mask in preds]

    # Visualize as a 4x2 matrix
    fig, axs = plt.subplots(len(images), 3, figsize=(6, 10))

    print(len(images), len(masks), len(preds))


    for i in range(len(images)):
        axs[i, 0].imshow(images[i])
        axs[i, 0].axis('off')

        axs[i, 1].imshow(masks[i])
        axs[i, 1].axis('off')

        axs[i, 2].imshow(preds[i])
        axs[i, 2].axis('off')

    plt.show()


def main():
    # Initialize lists to store images and masks
    images_to_viz = []
    masks_to_viz = []
    preds_to_viz = []

    # expt_path = os.path.join(root_path, 'Experiments', net+bb+method)
    expt_path = ROOT_PATH + '/Experiments/'
    if os.path.exists(expt_path):
        model = torch.load(expt_path + 'best_model.pth', map_location=device)
        print('Loaded UNet model from this run.')

    # Get train and val dataset instances
    # For ip2cl
    train_dataset = BuildingsDataset(
        x_train_pre_dir, x_train_post_dir, y_train_pre_dir, y_train_post_dir,
        # augmentation=get_training_augmentation(),
        # preprocessing=get_preprocessing(preprocessing_fn=None),
        class_rgb_values=select_class_rgb_values,
    )

    valid_dataset = BuildingsDataset(
        x_test_pre_dir, x_test_post_dir, y_test_pre_dir, y_test_post_dir,
        # augmentation=get_validation_augmentation(),
        # preprocessing=get_preprocessing(preprocessing_fn=None),
        class_rgb_values=select_class_rgb_values,
        val=True
    )

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0)

    test_dataset = BuildingsDataset(
        x_test_pre_dir, x_test_post_dir, y_test_pre_dir, y_test_post_dir,
        class_rgb_values=select_class_rgb_values,
        test=True
    )

    test_dataloader = DataLoader(test_dataset, batch_size=4)

    train_iterator = iter(train_loader)
    train_samples = next(train_iterator)

    # Validation Loader
    val_iterator = iter(valid_loader)
    val_samples = next(val_iterator)

    # Iterate through the dataloader
    for images, masks in test_dataloader:
        # Filter samples with change
        images_filtered, masks_filtered = filter_samples(images, masks)

        # Concatenate filtered images and masks
        images_to_viz.extend(images_filtered)
        masks_to_viz.extend(masks_filtered)

        # If the number of images to visualize is divisible by the batch size (4), predict on the batch
        if len(images_to_viz) > 10:
            images_batch = torch.stack(images_to_viz[:4]).to(device)
            preds_batch = model(images_batch)
            preds_to_viz.extend(preds_batch.cpu())
            # Visualize the predictions
            visualize(images_to_viz[:4], masks_to_viz[:4], preds_to_viz)
            images_to_viz, masks_to_viz = images_to_viz[4:], masks_to_viz[4:]






"""
    # predict
    model.eval()
    [train_imgs, train_msks] = train_samples
    [val_imgs, val_msks] = val_samples
    train_imgs, train_msks, val_imgs, val_msks = train_imgs.to(device), train_msks.to(device), val_imgs.to(
        device), val_msks.to(device)
    print(train_imgs.shape, val_imgs.shape)
    with torch.no_grad():
        train_preds = model(train_imgs)
        val_preds = model(val_imgs)
        train_msks, train_preds, val_msks, val_preds = train_msks.cpu(), train_preds.cpu(), val_msks.cpu(), val_preds.cpu()

    # print(type(train_samples), len(train_samples), train_samples[1].shape)
    # <class 'list'> 2 torch.Size([4, 3, 512, 512]) and torch.Size([4, 5, 512, 512])
    print(train_preds.shape, val_preds.shape)

    # Ensure tensor values are in the range [0, 1] for proper visualization
    train_samples = tuple(torch.clamp(sample, 0, 1) for sample in train_samples)
    val_samples = tuple(torch.clamp(sample, 0, 1) for sample in val_samples)

    # Convert one-hot encoded masks to non-one-hot format before visualization
    train_masks = [torch.argmax(mask, dim=0) for mask in train_samples[1]]
    val_masks = [torch.argmax(mask, dim=0) for mask in val_samples[1]]
    train_preds = [torch.argmax(mask, dim=0) for mask in train_preds]
    val_preds = [torch.argmax(mask, dim=0) for mask in val_preds]
    # print(len(non_one_hot_masks), non_one_hot_masks[0].shape)

    # Convert tensors to PIL Images
    to_pil = T.ToPILImage()
    train_images = [to_pil(img) for img in train_samples[0]]
    val_images = [to_pil(img) for img in val_samples[0]]
    # print(len(pil_images), pil_images[0].shape)

    # Visualize as a 4x2 matrix
    fig, axs = plt.subplots(8, 3, figsize=(6, 10))

    for i in range(4):
        axs[i, 0].imshow(train_images[i])
        axs[i, 0].axis('off')

        axs[i, 1].imshow(train_masks[i])
        axs[i, 1].axis('off')

        axs[i, 2].imshow(train_preds[i])
        axs[i, 2].axis('off')

    for i in range(4, 8):
        axs[i, 0].imshow(val_images[i-4])
        axs[i, 0].axis('off')

        axs[i, 1].imshow(val_masks[i-4])
        axs[i, 1].axis('off')

        axs[i, 2].imshow(val_preds[i-4])
        axs[i, 2].axis('off')

    plt.show()
"""

if __name__ == "__main__":
    main()



