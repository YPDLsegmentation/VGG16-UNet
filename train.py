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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################
file_prefix = '/scratch/xz/cityscapes'
# restore_epoch
# init_epoch + 1
FROM_SCRATCH = True
restore_epoch = 0
init_epoch = 0
init_step = 0

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
learning_rate = 0.001
weight = 5. # control regularzation term weight
num_epochs = 50
batch_size = 4
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
                'deconv1', 'norm1', \
                'conv6_1', 'conv6_2', \
                'deconv2', 'norm2', \
                'conv7_1', 'conv7_2', \
                'deconv3', 'norm3', \
                'conv8_1', 'conv8_2', \
                'deconv4', 'norm4', \
                'conv9_1', \
                'deconv5', \
                'conv10_1']
height = 1024
width = 2048

######################
# How often we want to write the tf.summary data to disk
display_step = 10
save_epoch = 10

######################
# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = os.path.join(file_prefix, "record/tfrecord4_norm")
checkpoint_path = os.path.join(file_prefix, "record/tfrecord4_norm")
old_checkpoint_path = os.path.join(file_prefix, "record/old_tfrecord4_norm")

# Create parent path if it doesn't exist
if not os.path.isdir(filewriter_path): 
    os.mkdir(filewriter_path)
if not os.path.isdir(checkpoint_path): 
    os.mkdir(checkpoint_path)
if not os.path.isdir(old_checkpoint_path): 
    os.mkdir(old_checkpoint_path)


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

# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))
  
  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in var_list:
  tf.summary.histogram(var.name, var)

# Add loss to summary
tf.summary.scalar('cross_ent_loss', crent_loss)
tf.summary.scalar('smooth_loss', smooth)
tf.summary.scalar('cross_ent_loss_with_smooth', loss)

# Add the IoU to the summary
for c in range(num_classes):
    tf.summary.scalar('class{}_IoU'.format(c), classIoU[c])
tf.summary.scalar('meanIoU', meanIoU)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# test loss and error_bias
with tf.name_scope("test_metric"):
    test_loss = tf.placeholder(tf.float32, [])
    test_classIoU = [tf.placeholder(tf.float32, [])] * num_classes
    test_meanIoU = tf.placeholder(tf.float32, [])

# Add test loss and error_bias to summary
ts1 = tf.summary.scalar('test_cross_ent_loss_with_smooth', test_loss)
ts2 = []
for c in range(num_classes):
    ts2.append(tf.summary.scalar('test class{}_IoU'.format(c), test_classIoU[c]))
ts3 = tf.summary.scalar('test meanIoU', test_meanIoU)

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, train_gt_file,
                                     data_augment = False, shuffle = True)
val_generator = ImageDataGenerator(val_file, val_gt_file,
                                   data_augment= False, shuffle = False) 
"""
saving space!
test_generator = ImageDataGenerator(test_file, test_gt_file,
                                   data_augment= False, shuffle = False) 
"""

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int32)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int32)
#test_batches_per_epoch = np.floor(test_generator.data_size / batch_size).astype(np.int32)

# Start Tensorflow session
with tf.Session(config=tf.ConfigProto(log_device_placement=False, \
        allow_soft_placement=True)) as sess:
 
  if FROM_SCRATCH:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
  else:
    restore_path = os.path.join(old_checkpoint_path, 'model_epoch%d.ckpt'%(restore_epoch))
    print "restoring from ckpt: {}...".format(restore_path)
    saver.restore(sess, restore_path)

  # Load the pretrained weights into the non-trainable layer
  #model.load_initial_weights(sess)
  
  print "training_batches_per_epoch: {}, val_batches_per_epoch: {}.".format(\
        train_batches_per_epoch, val_batches_per_epoch)
  print("{} Start training...".format(datetime.now()))
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
  
  # Loop over number of epochs
  for epoch in range(init_epoch, num_epochs):
    
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        if epoch == init_epoch:
            step += init_step
        
        while step < train_batches_per_epoch:
            
            print('epoch number: {}. step number: {}'.format(epoch+1, step))

            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs, 
                                          y: batch_ys, 
                                          keep_prob: dropout_rate})
            
            # Generate summary with the current batch of data and write to file
            if step%display_step == 0:
                print('{} displaying...'.format(datetime.now()))
                s = sess.run(merged_summary, feed_dict={x: batch_xs, 
                                                       y: batch_ys, 
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)
                

            step += 1
            
        if (epoch+1)%save_epoch:
            print("{} Saving checkpoint of model...".format(datetime.now()))  
            #save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+ str(epoch+1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)  
            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_ls = 0.
        test_mIoU = 0.
        test_cIoU = np.zeros((num_classes,), dtype=np.float32)
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            result = sess.run([meanIoU, loss] + classIoU, feed_dict={x: batch_tx, 
                                                                     y: batch_ty, 
                                                                     keep_prob: 1.})
            test_mIoU += result[0]
            test_ls += result[1]
            test_cIoU += np.array(result[2:])
            test_count += 1
        test_mIoU /= test_count
        test_ls /= test_count
        test_cIoU /= test_count
        print 'test_cIoU: {}, test_mIoU: {}, test_ls: {}'.format(test_cIoU, test_mIoU, test_ls)
        s1 = sess.run(ts1, feed_dict={test_loss: np.float32(test_ls)})
        writer.add_summary(s1, train_batches_per_epoch*(epoch + 1))

        for c in range(num_classes):
            s2 = sess.run(ts2[c], feed_dict={test_classIoU[c]: np.float32(test_cIoU[c])})
            writer.add_summary(s2, train_batches_per_epoch*(epoch + 1))

        s3 = sess.run(ts3, feed_dict={test_meanIoU: np.float32(test_mIoU)})
        writer.add_summary(s3, train_batches_per_epoch*(epoch + 1))

        
        
