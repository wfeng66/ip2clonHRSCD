import os, argparse, glob, tqdm
from PIL import Image, ImageOps
import numpy as np

Image.MAX_IMAGE_PIXELS = 100000000
# root_path = 'G://Research/SAA/2D/MapFormer/'
root_path = '/home/lab/feng/MapFormer/'
postTrain_path = root_path + "Data/original_size/train_post/"
postTestImg_path = root_path + "Data/original_size/test_post/"

imgPostTrList = glob.glob(postTrain_path + 'images/*.tif')
imgPostTrList = [file.replace('\\', '/') for file in imgPostTrList]
imgPreTrList = [path.replace('_post', '_pre').replace('-0M50-E080', '') for path in imgPostTrList]
imgPreTrList = [path.replace('2012', '2005') if '14-2012-' in path else path.replace('2012', '2006') for path in
                imgPreTrList]
mskPostTrList = [path.replace('images', 'targets') for path in imgPostTrList]
mskPreTrList = [path.replace('train_post', 'train_pre') for path in mskPostTrList]
imgPostTstList = glob.glob(postTestImg_path + 'images/*.tif')
imgPreTstList = [path.replace('_post', '_pre').replace('-0M50-E080', '') for path in imgPostTstList]
imgPreTstList = [path.replace('2012', '2005') if '14-2012-' in path else path.replace('2012', '2006') for path in
                 imgPreTstList]
mskPostTstList = [path.replace('images', 'targets') for path in imgPostTstList]
mskPreTstList = [path.replace('test_post', 'test_pre') for path in mskPostTstList]




def crop_large_image():
    """
    This function is used to crop the HRSCD data set images and masks to small size
    If only keep the building, need to enable the following code:
        msk_pre_arr[msk_pre_arr != 1] = 0
        cropped_msk_post_arr = cropped_msk_pre_arr * cropped_msk_post_arr
    """
    crop_size = (256, 256)
    original_size = (10000, 10000)
    width, height = original_size
    crop_width, crop_height = crop_size

    num_cols = width // crop_width + 1
    num_rows = height // crop_height + 1

    n_trImg = len(imgPreTrList)
    for k in tqdm.tqdm(range(n_trImg)):
        msk_post = Image.open(mskPostTrList[k])
        msk_post_arr = np.array(msk_post)
        msk_pre = Image.open(mskPreTrList[k])
        # msk_pre_arr = np.array(msk_pre)
        img_pre = Image.open(imgPreTrList[k])
        img_post = Image.open(imgPostTrList[k])
        # msk_pre_arr[msk_pre_arr != 1] = 0                                       # set all categories as backgroud but building

        i = 0
        for row in range(num_rows):
            for col in range(num_cols):
                left = col * crop_width
                upper = row * crop_height
                right = left + crop_width
                lower = upper + crop_height

                # cropped_msk_post_arr = msk_post_arr[upper:lower, left:right]
                # if np.max(cropped_msk_post_arr) == 0:          # if no change in this cropping
                #     continue

                # if no building change in this cropping
                # cropped_msk_pre_arr = msk_pre_arr[upper:lower, left:right]
                # cropped_msk_post_arr = cropped_msk_pre_arr * cropped_msk_post_arr
                # if np.max(cropped_msk_post_arr) == 0:
                #     continue
                # else:
                #     cropped_msk_pre = Image.fromarray(cropped_msk_pre_arr)
                #     cropped_msk_post = Image.fromarray(cropped_msk_post_arr)

                        # Check if this is the last column or row
                
                # Check if this is the last column or row
                last_col = (col == num_cols -1)
                last_row = (row == num_rows - 1)

                # Adjust right and lower boundaries if it's the last column or row
                if last_col:
                    right = width
                if last_row:
                    lower = height

                cropped_img_post = img_post.crop((left, upper, right, lower))
                cropped_img_pre = img_pre.crop((left, upper, right, lower))
                cropped_msk_post = msk_post.crop((left, upper, right, lower))
                cropped_msk_pre = msk_pre.crop((left, upper, right, lower))


                if last_col or last_row:
                    cropped_img_post = ImageOps.expand(cropped_img_post, border=(0, 0, crop_height-cropped_img_post.size[0], crop_width-cropped_img_post.size[1]))
                    cropped_img_pre = ImageOps.expand(cropped_img_pre, border=(0, 0, crop_height-cropped_img_pre.size[0], crop_width-cropped_img_pre.size[1]))
                    cropped_msk_post = ImageOps.expand(cropped_msk_post, border=(0, 0, crop_height-cropped_msk_post.size[0], crop_width-cropped_msk_post.size[1]))
                    cropped_msk_pre = ImageOps.expand(cropped_msk_pre, border=(0, 0, crop_height-cropped_msk_pre.size[0], crop_width-cropped_msk_pre.size[1]))


                cropped_img_pre_filename = imgPreTrList[k].replace('original_size', 'cropped_data').replace('.tif', '_'+str(i)+'.tif')
                cropped_img_pre.save(cropped_img_pre_filename)
                cropped_img_post_filename = imgPostTrList[k].replace('original_size', 'cropped_data').replace('.tif', '_'+str(i)+'.tif')
                cropped_img_post.save(cropped_img_post_filename)
                cropped_msk_pre_filename = mskPreTrList[k].replace('original_size', 'cropped_data').replace('.tif', '_'+str(i)+'.tif')
                cropped_msk_pre.save(cropped_msk_pre_filename)
                cropped_msk_post_filename = mskPostTrList[k].replace('original_size', 'cropped_data').replace('.tif', '_'+str(i)+'.tif')
                cropped_msk_post.save(cropped_msk_post_filename)

                i += 1

    n_tstImg = len(imgPreTstList)
    for k in tqdm.tqdm(range(n_tstImg)):
        msk_post = Image.open(mskPostTstList[k])
        msk_post_arr = np.array(msk_post)
        msk_pre = Image.open(mskPreTstList[k])
        # msk_pre_arr = np.array(msk_pre)
        img_pre = Image.open(imgPreTstList[k])
        img_post = Image.open(imgPostTstList[k])
        # msk_pre_arr[msk_pre_arr != 1] = 0  # set all categories as backgroud but building

        i = 0
        for row in range(num_rows):
            for col in range(num_cols):
                left = col * crop_width
                upper = row * crop_height
                right = left + crop_width
                lower = upper + crop_height

                # if np.max(msk_post_arr[upper:lower, left:right]) == 0:          # if no change in this cropping
                #     continue
                # else:
                #     cropped_msk_post_arr = msk_post_arr[upper:lower, left:right]

                # if no building change in this cropping
                # cropped_msk_pre_arr = msk_pre_arr[upper:lower, left:right]
                # cropped_msk_post_arr = cropped_msk_pre_arr * cropped_msk_post_arr
                # if np.max(cropped_msk_post_arr) == 0:
                #     continue
                # else:
                #     cropped_msk_pre = Image.fromarray(cropped_msk_pre_arr)
                #     cropped_msk_post = Image.fromarray(cropped_msk_post_arr)

                
                # Check if this is the last column or row
                last_col = (col == num_cols)
                last_row = (row == num_rows)

                # Adjust right and lower boundaries if it's the last column or row
                if last_col:
                    right = width
                if last_row:
                    lower = height

                cropped_img_post = img_post.crop((left, upper, right, lower))
                cropped_img_pre = img_pre.crop((left, upper, right, lower))
                cropped_msk_post = msk_post.crop((left, upper, right, lower))
                cropped_msk_pre = msk_pre.crop((left, upper, right, lower))

                if last_col or last_row:
                    cropped_img_post = ImageOps.expand(cropped_img_post, border=(0, 0, crop_height-cropped_img_post.size[0], crop_width-cropped_img_post.size[1]))
                    cropped_img_pre = ImageOps.expand(cropped_img_pre, border=(0, 0, crop_height-cropped_img_pre.size[0], crop_width-cropped_img_pre.size[1]))
                    cropped_msk_post = ImageOps.expand(cropped_msk_post, border=(0, 0, crop_height-cropped_msk_post.size[0], crop_width-cropped_msk_post.size[1]))
                    cropped_msk_pre = ImageOps.expand(cropped_msk_pre, border=(0, 0, crop_height-cropped_msk_pre.size[0], crop_width-cropped_msk_pre.size[1]))

                cropped_img_pre_filename = imgPreTstList[k].replace('original_size', 'cropped_data').replace('.tif','_' + str(i) + '.tif')
                cropped_img_pre.save(cropped_img_pre_filename)
                cropped_img_post_filename = imgPostTstList[k].replace('original_size', 'cropped_data').replace('.tif','_' + str(i) + '.tif')
                cropped_img_post.save(cropped_img_post_filename)
                cropped_msk_pre_filename = mskPreTstList[k].replace('original_size', 'cropped_data').replace('.tif','_' + str(i) + '.tif')
                cropped_msk_pre.save(cropped_msk_pre_filename)
                cropped_msk_post_filename = mskPostTstList[k].replace('original_size', 'cropped_data').replace('.tif','_' + str(i) + '.tif')
                cropped_msk_post.save(cropped_msk_post_filename)

                i += 1

if __name__ == "__main__":
    crop_large_image()