import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from model import NET
from datagenerator import ImageDataGenerator

"""
Configuration settings
"""
#######################
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

######################
file_prefix = '/scratch/xz/cityscapes'
# restore_epoch
restore_epoch = 15

######################
# Path to the textfiles for the trainings, validation and test set
train_file = os.path.join(file_prefix, 'data/train.in')
val_file = os.path.join(file_prefix, 'data/val.in')
test_file = os.path.join(file_prefix, 'data/test.in')
######################
# path to prefetching ground truth files
train_gt_file = os.path.join(file_prefix, 'data/gtFile_train_onehot.npy')
val_gt_file = os.path.join(file_prefix, 'data/gtFile_val_onehot.npy')
test_gt_file = os.path.join(file_prefix, 'data/gtFile_test_onehot.npy')
#####################
save_prefix = os.path.join(file_prefix, 'data/out4_no_norm')
save_suffix = '_out4_no_norm.png'

EVAL_TRAIN = False
EVAL_TEST = False
EVAL_VAL = True
TRAIN_SAVE = True
TEST_SAVE = True
VAL_SAVE = True

#####################
pixels = np.array([[128, 64, 128], # class 0
                   [244, 35, 232], # class 1
                   [70, 70, 70],   # class 2
                   [102, 102, 156],# class 3
                   [0, 0, 142],    # class 4
                   [0, 0, 0]],     # class 5
                   dtype=np.uint8)

######################
# Learning params
weight = 1.
batch_size = 4

######################
# Network params
num_classes = 6
train_layers = []
height = 512
width = 1024
mode = 1
ratio = 2

######################
# Path for tf.summary.FileWriter and to store model checkpoints
old_checkpoint_path = os.path.join(file_prefix, "record/old_tfrecord4_no_norm")

# TF placeholder for graph input and output
# TODO: to change this according to different settings
x = tf.placeholder(tf.float32, shape=[batch_size, height, width, 3])
y = tf.placeholder(tf.float32, shape=[batch_size, height, width, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = NET(x, height, width, keep_prob, train_layers, out_channels=num_classes, do_vbp=False, batch_size=batch_size)

# Link variable to model output
# NOTE: no softmax used, should use an extra softmax layer
pred_maps = model.conv10_1
tf.Print(pred_maps, [tf.constant("pred_maps"), pred_maps])
softmax_maps = tf.nn.softmax(pred_maps, dim=-1)
tf.Print(pred_maps, [tf.constant("softmax_maps"), softmax_maps])

assert pred_maps.shape == y.shape
assert softmax_maps.shape == y.shape

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
print "train_var num:{} list: ".format(len(var_list))
for v in var_list:
    print v.name

# Op for calculating the loss
with tf.name_scope("cross_ent_with_smooth"):
    ry = tf.reshape(y, [-1, num_classes])
    rpred_maps = tf.reshape(pred_maps, [-1, num_classes])
    crent_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ry, logits=rpred_maps))
    smooth = tf.reduce_mean(tf.squared_difference(pred_maps[:, 0:height-1, :, :], pred_maps[:, 1:, :, :]))
    smooth = tf.constant(weight, dtype=tf.float32)*(smooth + tf.reduce_mean(tf.squared_difference(pred_maps[:, :, 0:width-1, :], pred_maps[:, :, 1:, :])))
    loss = crent_loss + smooth

# Evaluation op: IoU
with tf.name_scope("IoU"):
    meanIoU = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    classIoU = []
    pred_max = tf.reduce_max(softmax_maps, axis=3) #[, , ,]
    tf.Print(pred_max, [tf.constant("pred_max"), pred_max])

    '''
    iss = []
    uss = []
    '''
    for c in range(num_classes):
        class_map = softmax_maps[:, :, :, c]
        class_max = tf.equal(class_map, pred_max) # bool
        gt_class = tf.cast(y[:, :, :, c], tf.bool)
        Inter = tf.reduce_sum(tf.cast(tf.logical_and(gt_class, class_max), tf.float32), axis=[1, 2])
        #iss.append(tf.Print(Inter, [tf.constant("class{}_Inter".format(c)), Inter]))
        Union = tf.reduce_sum(tf.cast(tf.logical_or(gt_class, class_max), tf.float32), axis=[1, 2])
        #uss.append(tf.Print(Union, [tf.constant("class{}_Union".format(c)), Union]))
        # check for validity
        tf.assert_greater_equal(Union, Inter)
        print "Inter shape: {}".format(Inter.shape)
        print "Union shape: {}".format(Union.shape)
        # add extra ones to union to prevent 0/0
        cIoU = Inter / (Union + tf.ones([batch_size], dtype=tf.float32))
        class_validNum = tf.greater_equal(tf.reduce_sum(y[:, :, :, c], axis=[1, 2]), tf.ones([batch_size], dtype=np.float32))
        class_validNum = tf.reduce_sum(tf.cast(class_validNum, tf.float32))
        cIoU = tf.reduce_sum(cIoU) / (tf.constant(0.01, dtype=tf.float32) + class_validNum)
        classIoU.append(cIoU)
        meanIoU = meanIoU + cIoU
    valid_classes = tf.greater_equal(tf.reduce_sum(y, axis=[0, 1, 2]), tf.ones([num_classes], dtype=np.float32))
    valid_classes = tf.reduce_sum(tf.cast(valid_classes, tf.float32))
    meanIoU = meanIoU / (tf.constant(0.01, dtype=tf.float32) + valid_classes)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

if EVAL_TRAIN:
  # Initalize the data generator seperately for the training and validation set
  train_generator = ImageDataGenerator(train_file, train_gt_file,
                                       data_augment = True, shuffle = False)
  train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int32)

if EVAL_VAL:
  val_generator = ImageDataGenerator(val_file, val_gt_file,
                                     data_augment= True, shuffle = False) 
  val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int32)

if EVAL_TEST:
  test_generator = ImageDataGenerator(test_file, test_gt_file,
                                     data_augment= True, shuffle = False) 
  test_batches_per_epoch = np.floor(test_generator.data_size / batch_size).astype(np.int32)


# Start Tensorflow session
with tf.Session(config=tf.ConfigProto(log_device_placement=False, \
        allow_soft_placement=True)) as sess:
 
    restore_path = os.path.join(old_checkpoint_path, 'model_epoch%d.ckpt'%(restore_epoch))
    print "restoring from ckpt: {}...".format(restore_path)
    saver.restore(sess, restore_path)
    print "Start Evaluation"

    if EVAL_TRAIN:
        print("{} on training set...".format(datetime.now()))
        crent_ls = 0.
        smooth_ls = 0.
        ls = 0.
        mIoU = 0.
        cIoU = np.zeros((num_classes,), dtype=np.float32)
        count = 0
        out = np.ndarray([train_generator.data_size, height, width, num_classes], dtype=np.float32)
        for step in range(train_batches_per_epoch):
            count += 1
            print('step number: {}'.format(step))
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size, mode, ratio)
            # And run the data
            result = sess.run([softmax_maps, meanIoU, crent_loss, smooth, loss] + classIoU, feed_dict={x: batch_xs, 
                                                                                                       y: batch_ys, 
                                                                                                       keep_prob: 1.})
            out[step*batch_size: (step+1)*batch_size] = result[0]
            mIoU += result[1]
            crent_ls += result[2]
            smooth_ls += result[3]
            ls += result[4]
            cIoU += np.array(result[5:])
        crent_ls /= count
        smooth_ls /= count
        ls /= count
        mIoU /= count
        cIoU /= count
        print 'cIoU: {}'.format(cIoU)
        print 'mIoU: {}'.format(mIoU)
        print 'crent_ls: {}'.format(crent_ls)
        print 'smooth_ls: {}'.format(smooth_ls)
        print 'ls: {}'.format(ls)
        if TRAIN_SAVE:
            for _ in range(count*batch_size):
                path = train_generator.in_paths[_].replace('/n/data/cityscapes/leftImg8bit', save_prefix)
                path = path.replace('_leftImg8bit.png', save_suffix)
                tem = out[_] #[, , ,]
                pred_max = np.max(tem, axis=-1) #[, ,]
                img = np.zeros((height, width, 3), dtype=np.uint8)
                for c in range(num_classes):
                    class_map = tem[:, :, c]
                    class_assign = pred_max == class_map
                    img[class_assign] = pixels[c]
                # change RGB to BGR
                img = img[:, :, [2, 1,0]]
                cv2.imwrite(path, img)
            print "training set output saved"
            
    if EVAL_TEST:
        print("{} on testing set...".format(datetime.now()))
        crent_ls = 0.
        smooth_ls = 0.
        ls = 0.
        mIoU = 0.
        cIoU = np.zeros((num_classes,), dtype=np.float32)
        count = 0
        out = np.ndarray([test_generator.data_size, height, width, num_classes], dtype=np.float32)
        for step in range(test_batches_per_epoch):
            count += 1
            print('step number: {}'.format(step))
            # Get a batch of images and labels
            batch_xs, batch_ys = test_generator.next_batch(batch_size, mode, ratio)
            # And run the data
            result = sess.run([softmax_maps, meanIoU, crent_loss, smooth, loss] + classIoU, feed_dict={x: batch_xs, 
                                                                                                       y: batch_ys, 
                                                                                                       keep_prob: 1.})
            out[step*batch_size: (step+1)*batch_size] = result[0]
            mIoU += result[1]
            crent_ls += result[2]
            smooth_ls += result[3]
            ls += result[4]
            cIoU += np.array(result[5:])
        crent_ls /= count
        smooth_ls /= count
        ls /= count
        mIoU /= count
        cIoU /= count
        print 'cIoU: {}'.format(cIoU)
        print 'mIoU: {}'.format(mIoU)
        print 'crent_ls: {}'.format(crent_ls)
        print 'smooth_ls: {}'.format(smooth_ls)
        print 'ls: {}'.format(ls)
        if TEST_SAVE:
            for _ in range(count*batch_size):
                path = test_generator.in_paths[_].replace('/n/data/cityscapes/leftImg8bit', save_prefix)
                path = path.replace('_leftImg8bit.png', save_suffix)
                tem = out[_] #[, , ,]
                pred_max = np.max(tem, axis=-1) #[, ,]
                img = np.zeros((height, width, 3), dtype=np.uint8)
                for c in range(num_classes):
                    class_map = tem[:, :, c]
                    class_assign = pred_max == class_map
                    img[class_assign] = pixels[c]
                # change RGB to BGR
                img = img[:, :, [2, 1,0]]
                cv2.imwrite(path, img)
            print "testing set output saved"

    if EVAL_VAL:
        print("{} on val set...".format(datetime.now()))
        crent_ls = 0.
        smooth_ls = 0.
        ls = 0.
        mIoU = 0.
        cIoU = np.zeros((num_classes,), dtype=np.float32)
        count = 0
        out = np.ndarray([val_generator.data_size, height, width, num_classes], dtype=np.float32)
        for step in range(val_batches_per_epoch):
            count += 1
            print('step number: {}'.format(step))
            # Get a batch of images and labels
            batch_xs, batch_ys = val_generator.next_batch(batch_size, mode, ratio)
            # And run the data
            result = sess.run([softmax_maps, meanIoU, crent_loss, smooth, loss] + classIoU, feed_dict={x: batch_xs, 
                                                                                                       y: batch_ys, 
                                                                                                       keep_prob: 1.})
            out[step*batch_size: (step+1)*batch_size] = result[0]
            mIoU += result[1]
            crent_ls += result[2]
            smooth_ls += result[3]
            ls += result[4]
            cIoU += np.array(result[5:])
        crent_ls /= count
        smooth_ls /= count
        ls /= count
        mIoU /= count
        cIoU /= count
        print 'cIoU: {}'.format(cIoU)
        print 'mIoU: {}'.format(mIoU)
        print 'crent_ls: {}'.format(crent_ls)
        print 'smooth_ls: {}'.format(smooth_ls)
        print 'ls: {}'.format(ls)
        if VAL_SAVE:
            for _ in range(count*batch_size):
                path = val_generator.in_paths[_].replace('/n/data/cityscapes/leftImg8bit', save_prefix)
                path = path.replace('_leftImg8bit.png', save_suffix)
                tem = out[_] #[, , ,]
                pred_max = np.max(tem, axis=-1) #[, ,]
                img = np.zeros((height, width, 3), dtype=np.uint8)
                for c in range(num_classes):
                    class_map = tem[:, :, c]
                    class_assign = pred_max == class_map
                    img[class_assign] = pixels[c]
                # change RGB to BGR
                img = img[:, :, [2, 1,0]]
                cv2.imwrite(path, img)
            print "valing set output saved"
