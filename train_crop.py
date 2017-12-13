import os
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

######################
# Learning params
learning_rate = 0.0001
weight = 1. # control regularzation term weight
beta1 = 0.9 #momentum for adam

######################
# Network params
dropout_rate = .5 # no use here
num_classes = 6
train_layers = ['conv1_1', 'conv1_2', \
                'conv2_1', 'conv2_2', \
                'conv3_1', 'conv3_2', 'conv3_3', \
                'conv4_1', 'conv4_2', 'conv4_3', \
                'conv5_1', 'conv5_2', 'conv5_3', \
                'deconv1', \
                'conv6_1', 'conv6_2', \
                'deconv2', \
                'conv7_1', 'conv7_2', \
                'deconv3', \
                'conv8_1', 'conv8_2', \
                'deconv4', \
                'conv9_1', \
                'deconv5', \
                'conv10_1']

restore_layers = None
'''
                 ['conv1_1', 'conv1_2', \
                  'conv2_1', 'conv2_2', \
                  'conv3_1', 'conv3_2', 'conv3_3', \
                  'conv4_1', 'conv4_2', 'conv4_3', \
                  'conv5_1', 'conv5_2', 'conv5_3']
                '''
#####################
big_epoch_num = 20 # passing through all settings counts as one epoch
# configuration for input 
size = [[512, 1024], [256, 512], \
        [512, 512], [256, 256]]
mode_ratio = [[1, 2], [1, 4],  \
              [3, 2], [3, 4]]
batch_sizes = [4, 16, 8, 32]
small_epoches = [1, 4, 2, 8]
small_epoches_per_big = sum(small_epoches)
small_epoches_presum = [sum(small_epoches[0:_+1])-small_epoches[_] for _ in range(len(small_epoches))]

####################
init_big_epoch = 10
init_setting = 0
init_small_epoch = 0
init_step = 0

######################
# How often we want to write the tf.summary data to disk
display_steps = [10, 2, 5, 1]
save_epoches = [1, 4, 2, 8]

######################
# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = os.path.join(file_prefix, "record/tfrecord4_no_norm_crop")
checkpoint_path = os.path.join(file_prefix, "record/tfrecord4_no_norm_crop")
old_checkpoint_path = os.path.join(file_prefix, "record/old_tfrecord4_no_norm_crop")

# Create parent path if it doesn't exist
if not os.path.isdir(filewriter_path): 
    os.mkdir(filewriter_path)
if not os.path.isdir(checkpoint_path): 
    os.mkdir(checkpoint_path)
if not os.path.isdir(old_checkpoint_path): 
    os.mkdir(old_checkpoint_path)

######################
# restore_path
FROM_SCRATCH = False
restore_path = os.path.join(old_checkpoint_path, 'model_big_9_setting_3_small_7.ckpt')

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, train_gt_file,
                                     mask=(3,),
                                     data_augment = True, shuffle = True)

val_generator = ImageDataGenerator(val_file, val_gt_file,
                                   data_augment= True, shuffle = False) 
"""
saving space!
test_generator = ImageDataGenerator(test_file, test_gt_file,
                                   data_augment= False, shuffle = False) 
"""
# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# main loop
# loop over bit epoches
for big_epoch in range(init_big_epoch, big_epoch_num):
    # median loop
    # loop over all settings
    if not big_epoch == init_big_epoch:
        init_setting = 0
    for setting in range(init_setting, len(size)):
        height, width = size[setting]
        mode, ratio = mode_ratio[setting]
        batch_size = batch_sizes[setting]
        small_epoch_num = small_epoches[setting]
        display_step = display_steps[setting]
        save_epoch = save_epoches[setting]

        #####################
        # reset pointer
        train_generator.reset_pointer()
        val_generator.reset_pointer()
        #test_generator.reset_pointer()

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int32)
        val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int32)
        #test_batches_per_epoch = np.floor(test_generator.data_size / batch_size).astype(np.int32)

        # set a new graph
        # TF placeholder for graph input and output
        # change this according to different settings
        x = tf.placeholder(tf.float32, shape=[batch_size, height, width, 3])
        y = tf.placeholder(tf.float32, shape=[batch_size, height, width, num_classes])
        keep_prob = tf.placeholder(tf.float32)

        # Initialize model
        model = NET(x, height, width, keep_prob, train_layers, out_channels=num_classes, do_vbp=False, batch_size=batch_size)

        # Link variable to model output
        # NOTE: no softmax used, should use an extra softmax layer
        pred_maps = model.conv10_1
        softmax_maps = tf.nn.softmax(pred_maps, dim=-1)
        assert pred_maps.shape == y.shape
        assert softmax_maps.shape == y.shape

        # TODO: to make sure what will happen if we redefine the same variables
        # loss
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

        # training
        # List of trainable variables of the layers we want to train
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        print "train_var num:{} list: ".format(len(var_list))
        for v in var_list:
            print v.name

        # Train op
        with tf.name_scope('train'):
            with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
                # Get gradients of all trainable variables
                gradients = tf.gradients(loss, var_list)
                gradients = list(zip(gradients, var_list))
                
                # Create optimizer and apply gradient descent to the trainable variables
                optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
                train_op = optimizer.apply_gradients(grads_and_vars=gradients)

        # recording
        # Add gradients to summary  
        with tf.name_scope('training_record'):
            summary = []
            for gradient, var in gradients:
                summary.append(tf.summary.histogram(var.name + '/gradient', gradient))

            # Add the variables we train to the summary  
            for var in var_list:
                summary.append(tf.summary.histogram(var.name, var))

            # Add loss to summary
            summary.append(tf.summary.scalar('cross_ent_loss', crent_loss))
            summary.append(tf.summary.scalar('smooth_loss', smooth))
            summary.append(tf.summary.scalar('cross_ent_loss_with_smooth', loss))

            # Add the IoU to the summary
            for c in range(num_classes):
                summary.append(tf.summary.scalar('class{}_IoU'.format(c), classIoU[c]))
            summary.append(tf.summary.scalar('meanIoU', meanIoU))
            # Merge all summaries together
            merged_summary = tf.summary.merge(summary)

        # test record
        # test loss and error_bias
        with tf.name_scope("test_metric"):
            test_dict = {}
            test_summary = []
            test_crent_loss = tf.placeholder(tf.float32, [])
            test_smooth_loss = tf.placeholder(tf.float32, [])
            test_loss = tf.placeholder(tf.float32, [])
            test_classIoU = [tf.placeholder(tf.float32, []) for _ in range(num_classes)]
            test_meanIoU = tf.placeholder(tf.float32, [])

            # Add test loss and error_bias to summary
            test_summary.append(tf.summary.scalar('test_cross_ent_loss', test_crent_loss))
            test_dict[test_crent_loss] = 0.

            test_summary.append(tf.summary.scalar('test_smooth_loss', test_smooth_loss))
            test_dict[test_smooth_loss] = 0.

            test_summary.append(tf.summary.scalar('test_cross_ent_loss_with_smooth', test_loss))
            test_dict[test_loss] = 0.
            for c in range(num_classes):
                test_summary.append(tf.summary.scalar('test_class{}_IoU'.format(c), test_classIoU[c]))
                test_dict[test_classIoU[c]] = 0.
            test_summary.append(tf.summary.scalar('test_meanIoU', test_meanIoU))
            test_dict[test_meanIoU] = 0.
            test_merged_summary = tf.summary.merge(test_summary)

        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver()

        # Start Tensorflow session
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, \
                allow_soft_placement=True)) as sess:
            if FROM_SCRATCH:
                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                # Add the model graph to TensorBoard
                writer.add_graph(sess.graph)
            else:
                print "restoring from ckpt: {}...".format(restore_path)
                saver.restore(sess, restore_path)

            print "big_epoch: {}".format(big_epoch)
            print "setting: {}".format(setting)
            print "training_batches_per_epoch: {}, val_batches_per_epoch: {}.".format(\
                   train_batches_per_epoch, val_batches_per_epoch)
            print("{} Start training...".format(datetime.now()))
            # small loop
            # loop over small epoches under one setting
            if not (big_epoch == init_big_epoch and setting == init_setting):
                init_samll_epoch = 0
            for epoch in range(init_small_epoch, small_epoch_num):
                print("{} small epoch: {}".format(datetime.now(), epoch))
                if not(big_epoch == init_big_epoch and setting == init_setting and epoch == init_small_epoch):
                    init_step = 0
                for step in range(init_step, train_batches_per_epoch):
                    print('\tsmall epoch: {} step: {}'.format(epoch, step))
                    # Get a batch of images and labels
                    batch_xs, batch_ys = train_generator.next_batch(batch_size, mode, ratio)
                    # And run the training op
                    sess.run(train_op, feed_dict={x: batch_xs, 
                                                  y: batch_ys, 
                                                  keep_prob: dropout_rate})
                    # Generate summary with the current batch of data and write to file
                    if (step+1)%display_step == 0:
                        print('{} displaying...'.format(datetime.now()))
                        s = sess.run(merged_summary, feed_dict={x: batch_xs, 
                                                                y: batch_ys, 
                                                                keep_prob: 1.})
                        writer.add_summary(s, (big_epoch*small_epoches_per_big + small_epoches_presum[setting] + epoch)*train_generator.data_size + step*batch_size)
                        
                if (epoch+1)%save_epoch == 0:
                    print("{} Saving checkpoint of model...".format(datetime.now()))  
                    #save checkpoint of the model
                    checkpoint_name = os.path.join(checkpoint_path, 'model_big_{}_setting_{}_small_{}.ckpt'.format(big_epoch, setting, epoch))
                    restore_path = checkpoint_name
                    FROM_SCRATCH = False
                    save_path = saver.save(sess, checkpoint_name)  
                    print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
                    os.system('cp {}* {}/'.format(checkpoint_name, old_checkpoint_path))
                    print "copied to {}".format(old_checkpoint_path)

                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                test_crent_ls = 0.
                test_smooth_ls = 0
                test_ls = 0.
                test_mIoU = 0.
                test_cIoU = np.zeros((num_classes,), dtype=np.float32)
                test_count = 0
                for _ in range(val_batches_per_epoch):
                    batch_tx, batch_ty = val_generator.next_batch(batch_size, mode, ratio)
                    result = sess.run([meanIoU, crent_loss, smooth, loss] + classIoU, feed_dict={x: batch_tx, 
                                                                                                 y: batch_ty, 
                                                                                                 keep_prob: 1.})
                    test_mIoU += result[0]
                    test_crent_ls += result[1]
                    test_smooth_ls += result[2]
                    test_ls += result[3]
                    test_cIoU += np.array(result[4:])
                    test_count += 1
                test_mIoU /= test_count
                test_crent_ls /= test_count
                test_smooth_ls /= test_count
                test_ls /= test_count
                test_cIoU /= test_count
                print 'test_cIoU: {}\n test_mIoU: {}\n test_crent_ls: {}\n test_smooth_ls: {}\n test_ls: {}'.format(\
                       test_cIoU, test_mIoU, test_crent_ls, test_smooth_ls,  test_ls)
                test_dict[test_crent_loss] = np.float32(test_crent_ls)
                test_dict[test_smooth_loss] = np.float32(test_smooth_ls)
                test_dict[test_loss] = np.float32(test_ls)
                test_dict[test_meanIoU] = np.float32(test_mIoU)
                for c in range(num_classes):
                    test_dict[test_classIoU[c]] = np.float32(test_cIoU[c])
                print len(test_dict.keys())
                assert len(test_dict.keys()) == 4 + num_classes
                s1 = sess.run(test_merged_summary, feed_dict=test_dict)
                writer.add_summary(s1, (big_epoch*small_epoches_per_big + small_epoches_presum[setting] + epoch + 1)*train_generator.data_size)
        # reset current default graph
        tf.reset_default_graph()
