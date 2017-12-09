import os
import numpy as np
import cv2
import copy
import random
from tensorflow import expand_dims

class ImageDataGenerator:
    def __init__(self, class_list, gtFile=None, data_augment=False, shuffle=False, 
                 max_image_height = 1024, max_image_width = 2048,
                 # 5 + 1 classes
                 classes = 6,
                 input_prefix = '/n/data/cityscapes/leftImg8bit',
                 input_suffix = '_leftImg8bit.png',
                 gt_prefix='/scratch/xz/cityscapes/data/gtFine', 
                 gt_suffix='_gtFine_labelTrainIds.png'): 
                 # no use here
                 #mean = np.array([114.532, 118.607, 119.061]), scale_size=(224, 224),
                 #nb_classes = 1):
        
        # Init params
        self.data_augment = data_augment
        self.max_image_height = max_image_height
        self.max_image_width = max_image_width
        self.classes = classes
        self.in_prefix = input_prefix
        self.in_suffix = input_suffix
        self.gt_prefix = gt_prefix
        self.gt_suffix = gt_suffix
        self.shuffle = shuffle
        self.pointer = 0
        # used for iterations count
        self.iter = 0
        self.read_class_list(class_list)
        self.prefetch(gtFile)
        
        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self,class_list):
        """
        Scan the image file and get the image paths and ground truth path
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.in_paths = []
            self.gt_paths = []
            for l in lines:
                input_path = l.strip()
                gt_path = input_path.replace(self.in_prefix, self.gt_prefix).replace(self.in_suffix, self.gt_suffix)
                self.in_paths.append(input_path)
                self.gt_paths.append(gt_path)
            
            #store total number of data
            self.data_size = len(self.in_paths)
        

    def prefetch(self, gtFile=None):
        """
        prefetch images and gt to speed up training
        """ 
        print "prefetching from {}".format(gtFile)
        # do not prefetch raw images since it's too big to fit into main memory
        # self.all_in_images = np.ndarray([self.data_size, self.max_image_height, self.max_image_width, 3], dtype=np.uint8)
        # NOTE: using int8 to save space
        self.all_gt_images = np.ndarray([self.data_size, self.max_image_height, self.max_image_width, self.classes], dtype=np.int8)
        '''
        for i in range(self.data_size):
            self.all_in_images[i] = cv2.imread(self.in_paths[i], cv2.IMREAD_COLOR)
        '''
        if gtFile is None:
            print "from raw images"
            for i in range(self.data_size):
                gt_img = cv2.imread(self.gt_paths[i], cv2.IMREAD_GRAYSCALE)
                gt_img = np.reshape(gt_img, gt_img.shape + (1,))
                # convert gt_img into one hot labels
                # NOTE: using int8 to save space
                tem = np.zeros((self.max_image_height, self.max_image_width, self.classes), dtype=np.int8)
                for index in np.ndindex(gt_img.shape):
                    tem[index[0: 2] + (int(min(self.classes-1, gt_img[index])),)] = 1
                self.all_gt_images[i] = tem
        else:
            print "from pre-processed file"
            gt_dict = np.load(gtFile).item()
            for i in range(self.data_size):
                self.all_gt_images[i] = gt_dict[self.gt_paths[i]]
        
        print "prefetch done"
        # print "all_in_images shape:{}".format(self.all_in_images.shape)
        print "all_gt_images shape:{}".format(self.all_gt_images.shape)

    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        in_paths = copy.deepcopy(self.in_paths)
        gt_paths = copy.deepcopy(self.gt_paths)
        #all_in_imgs = copy.deepcopy(self.all_in_images)
        all_gt_imgs = copy.deepcopy(self.all_gt_images)

        self.in_paths = []
        self.gt_paths = []
        #self.all_in_images = np.ndarray([self.data_size, self.max_image_height, self.max_image_width, 3], dtype=np.uint8)
        self.all_gt_images = np.ndarray([self.data_size, self.max_image_height, self.max_image_width, self.classes], dtype=np.int8)
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(in_paths))
        for i in idx:
            self.in_paths.append(in_paths[i])
            self.gt_paths.append(gt_paths[i])
        self.all_gt_images = all_gt_imgs[idx, :, :, :]
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
    
    def next_batch(self, batch_size, mode=0, ratio=1):
        """
        This function gets the next n ( = batch_size) images from the path list
        and loads the images into them into memory
        NOTE: 3 modes if data augmented
        NOTE: ratio: downsample ratio, should be 2^k
        """
        # Get next batch of image (path) and labels
        in_paths = self.in_paths[self.pointer:self.pointer + batch_size]
        gt_paths = self.gt_paths[self.pointer:self.pointer + batch_size]
        
        # Read images
        scale_height = self.max_image_height
        scale_width = self.max_image_width
        min_len = min(scale_height, scale_width)
        max_len = max(scale_height, scale_width)

        if self.data_augment:
            if mode == 0:
                pass
            elif mode == 1:
                scale_height /= ratio
                scale_width /= ratio
            elif mode == 2:
                scale_height = min_len
                scale_width = min_len
            elif mode == 3:
                scale_height = min_len / ratio
                scale_width = min_len / ratio

        # note the order and channels
        self.in_images = np.ndarray([batch_size, scale_height, scale_width, 3], dtype=np.float32)
        self.gt_images = np.ndarray([batch_size, scale_height, scale_width, self.classes], dtype=np.float32) # NOTE: for output, using float is ok
        #in_imgs = self.all_in_images[self.pointer:self.pointer+batch_size, :, :, :]
        in_imgs = np.ndarray([batch_size, self.max_image_height, self.max_image_width, 3], dtype=np.uint8)
        for i in range(len(in_paths)):
            in_imgs[i] = cv2.imread(in_paths[i], cv2.IMREAD_COLOR)

        gt_imgs = self.all_gt_images[self.pointer:self.pointer+batch_size, :, :, :]

        if self.data_augment:
            if mode == 0:
                # mode 0: raw image
                pass
            elif mode == 1:
                # mode 1: downsample of raw image
                choice = random.randint(0, 1)
                in_imgs = self.downsample_batch(in_imgs, ratio, choice)
                gt_imgs = self.downsample_batch(gt_imgs, ratio, choice)
            elif mode == 2:
                # mode 2: cropping image to 1:1
                # random crop using numpy slicing
                upper = random.randint(min_len, max_len)
                height_upper = min(upper, in_imgs.shape[1])
                width_upper = min(upper, in_imgs.shape[2])
                in_imgs = in_imgs[:,  height_upper - min_len: height_upper, width_upper - min_len: width_upper, :]
                gt_imgs = gt_imgs[:, height_upper - min_len: height_upper, width_upper - min_len: width_upper, :]
            elif mode == 3:
                # mode 3: downsample of cropped image
                # random crop using numpy slicing
                upper = random.randint(min_len, max_len)
                height_upper = min(upper, in_imgs.shape[1])
                width_upper = min(upper, in_imgs.shape[2])
                in_imgs = in_imgs[:, height_upper - min_len: height_upper, width_upper - min_len: width_upper, :]
                gt_imgs = gt_imgs[:, height_upper - min_len: height_upper, width_upper - min_len: width_upper, :]
                # downsample
                choice = random.randint(0, 1)
                in_imgs = self.downsample_batch(in_imgs, ratio, choice)
                gt_imgs = self.downsample_batch(gt_imgs, ratio, choice)
            
        assert in_imgs.shape == (batch_size, scale_height, scale_width, 3)
        assert gt_imgs.shape == (batch_size, scale_height, scale_width, self.classes)
        # type conversion
        in_imgs = in_imgs.astype(np.float32)
        gt_imgs = gt_imgs.astype(np.float32)
        self.in_images = in_imgs
        self.gt_images = gt_imgs
         
        #update pointer
        self.pointer += batch_size
        # reset before exhaustion
        if self.pointer + batch_size > self.data_size:
            self.reset_pointer()
            self.iter += 1

        return self.in_images, self.gt_images

    def test_mk(self, path):
        if os.path.isdir(path):
            return
        else:
            os.mkdir(path)

    def downsample(self, img, ratio, choice):
        """
        downsample a image of ratio 2^k
        just discarding all odd/even rows/columns to size 1/2
        choice: [0/1] choose to first keep odd/even
        """
        keep = range(ratio)
        while len(keep) > 1:
            if choice == 0:
                keep = keep[0: len(keep): 2]
                choice = 1
            else:
                keep = keep[1: len(keep): 2]
                choice = 0
        keep = keep[0]
        rows_keep = range(keep, img.shape[0], ratio)
        cols_keep = range(keep, img.shape[1], ratio)
        img = img[rows_keep, :, :]
        img = img[:, cols_keep, :]
        return img

    def downsample_batch(self, imgs, ratio, choice):
        """
        downsample a batch of images of ratio 2^k
        just discarding all odd/even rows/columns to size 1/2
        choice: [0/1] choose to first keep odd/even
        """
        keep = range(ratio)
        while len(keep) > 1:
            if choice == 0:
                keep = keep[0: len(keep): 2]
                choice = 1
            else:
                keep = keep[1: len(keep): 2]
                choice = 0
        keep = keep[0]
        rows_keep = range(keep, imgs.shape[1], ratio)
        cols_keep = range(keep, imgs.shape[2], ratio)
        imgs = imgs[:, rows_keep, :, :]
        imgs = imgs[:, :, cols_keep, :]
        return imgs



