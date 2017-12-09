import tensorflow as tf
import numpy as np

class VGG16(object):
  
  def __init__(self, x, keep_prob, num_classes, skip_layer, 
               weights_path = 'DEFAULT', do_vbp = False, batch_size = 64):
    
    # Parse input arguments into class variables
    self.X = x
    self.batch_size = batch_size
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.SKIP_LAYER = skip_layer
    
    if weights_path == 'DEFAULT':      
      self.WEIGHTS_PATH = 'vgg16.npy'
    else:
      self.WEIGHTS_PATH = weights_path
    
    # Call the create function to build the computational graph of AlexNet
    self.create()
    if do_vbp:
      self.vbp()
    
  def create(self):
    
    #input size [batch_size, 224, 224, 3]
    self.conv1 = conv(self.X, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv1') # [~, 224, 224, 64]
    self.conv2 = conv(self.conv1, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv2') # [~, 224, 224, 64]
    self.pool1 = max_pool(self.conv2, 2, 2, 2, 2, padding = 'VALID', name = 'pool1') # [~, 112, 112, 64]
    
    self.conv3 = conv(self.pool1, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv3') # [~, 112, 112, 128]
    self.conv4 = conv(self.conv3, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv4') # [~, 112, 112, 128]
    self.pool2 = max_pool(self.conv4, 2, 2, 2, 2, padding = 'VALID', name ='pool2') # [~, 56, 56, 128]
    
    self.conv5 = conv(self.pool2, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv5') # [~, 56, 56, 256]
    self.conv6 = conv(self.conv5, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv6') # [~, 56, 56, 256]
    self.conv7 = conv(self.conv6, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv7') # [~, 56, 56, 256]
    self.pool3 = max_pool(self.conv7, 2, 2, 2, 2, padding = 'VALID', name ='pool3') # [~, 28, 28, 256]

    self.conv8 = conv(self.pool3, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv8') # [~, 28, 28, 512]
    self.conv9 = conv(self.conv8, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv9') # [~, 28, 28, 512]
    self.conv10 = conv(self.conv9, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv10') # [~, 28, 28, 512]
    self.pool4 = max_pool(self.conv10, 2, 2, 2, 2, padding = 'VALID', name ='pool4') # [~, 14, 14, 512]
    
    self.conv11 = conv(self.pool4, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv11') # [~, 14, 14, 512]
    self.conv12 = conv(self.conv11, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv12') # [~, 14, 14, 512]
    self.conv13 = conv(self.conv12, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv13') # [~, 14, 14, 512]
    self.pool5 = max_pool(self.conv13, 2, 2, 2, 2, padding = 'VALID', name = 'pool5') # [~, 7, 7, 512]
    
    self.flattened = tf.reshape(self.pool5, [-1, 7*7*512])
    self.fc1 = fc(self.flattened, 7*7*512, 4096, name='fc1')
    self.dropout1 = dropout(self.fc1, self.KEEP_PROB)
    self.fc2 = fc(self.dropout1, 4096, 4096, name='fc2')
    self.dropout2 = dropout(self.fc2, self.KEEP_PROB)
    self.fc3 = fc(self.dropout2, 4096, self.NUM_CLASSES, relu = False, name='fc3')


  def vbp(self):
    # self.pool5
    self.ave_pool5 = tf.reduce_mean(self.pool5, axis=3, keep_dims=True)
    print "ave_pool5 shape: ", self.ave_pool5.shape
    self.deconv_pool5 = tf.nn.conv2d_transpose(self.ave_pool5, tf.ones([2, 2, 1, 1]), output_shape=[self.batch_size, 14, 14, 1], strides=[1, 2, 2, 1], padding="VALID")
    print "deconv_pool5 shape: ", self.deconv_pool5.shape
    
    self.ave_conv13 = tf.reduce_mean(self.conv13, axis=3, keep_dims=True)
    self.mask_conv13 = tf.multiply(self.ave_conv13, self.deconv_pool5);
    self.deconv13 = tf.nn.conv2d_transpose(self.mask_conv13, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 14, 14, 1], strides=[1, 1, 1, 1], padding="SAME")

    self.ave_conv12 = tf.reduce_mean(self.conv12, axis=3, keep_dims=True)
    self.mask_conv12 = tf.multiply(self.ave_conv12, self.deconv13);
    self.deconv12 = tf.nn.conv2d_transpose(self.mask_conv12, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 14, 14, 1], strides=[1, 1, 1, 1], padding="SAME")
    
    self.ave_conv11 = tf.reduce_mean(self.conv11, axis=3, keep_dims=True)
    self.mask_conv11 = tf.multiply(self.ave_conv11, self.deconv12);
    self.deconv11 = tf.nn.conv2d_transpose(self.mask_conv12, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 14, 14, 1], strides=[1, 1, 1, 1], padding="SAME")
    # *************************
    self.ave_pool4 = tf.reduce_mean(self.pool4, axis=3, keep_dims=True)
    print "ave_pool4 shape: ", self.ave_pool4.shape
    self.mask_pool4 = tf.multiply(self.ave_pool4, self.deconv11)
    self.deconv_pool4 = tf.nn.conv2d_transpose(self.mask_pool4, tf.ones([2, 2, 1, 1]), output_shape=[self.batch_size, 28, 28, 1], strides=[1, 2, 2, 1], padding="VALID")
    print "deconv_pool4 shape: ", self.deconv_pool4.shape
    
    self.ave_conv10 = tf.reduce_mean(self.conv10, axis=3, keep_dims=True)
    self.mask_conv10 = tf.multiply(self.ave_conv10, self.deconv_pool4);
    self.deconv10 = tf.nn.conv2d_transpose(self.mask_conv10, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 28, 28, 1], strides=[1, 1, 1, 1], padding="SAME")

    self.ave_conv9 = tf.reduce_mean(self.conv9, axis=3, keep_dims=True)
    self.mask_conv9 = tf.multiply(self.ave_conv9, self.deconv10);
    self.deconv9 = tf.nn.conv2d_transpose(self.mask_conv9, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 28, 28, 1], strides=[1, 1, 1, 1], padding="SAME")
    
    self.ave_conv8 = tf.reduce_mean(self.conv8, axis=3, keep_dims=True)
    self.mask_conv8 = tf.multiply(self.ave_conv8, self.deconv9);
    self.deconv8 = tf.nn.conv2d_transpose(self.mask_conv8, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 28, 28, 1], strides=[1, 1, 1, 1], padding="SAME")
    # ************************
    self.ave_pool3 = tf.reduce_mean(self.pool3, axis=3, keep_dims=True)
    print "ave_pool3 shape: ", self.ave_pool3.shape
    self.mask_pool3 = tf.multiply(self.ave_pool3, self.deconv8)
    self.deconv_pool3 = tf.nn.conv2d_transpose(self.mask_pool3, tf.ones([2, 2, 1, 1]), output_shape=[self.batch_size, 56, 56, 1], strides=[1, 2, 2, 1], padding="VALID")
    print "deconv_pool3 shape: ", self.deconv_pool3.shape
    
    self.ave_conv7 = tf.reduce_mean(self.conv7, axis=3, keep_dims=True)
    self.mask_conv7 = tf.multiply(self.ave_conv7, self.deconv_pool3);
    self.deconv7 = tf.nn.conv2d_transpose(self.mask_conv7, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 56, 56, 1], strides=[1, 1, 1, 1], padding="SAME")

    self.ave_conv6 = tf.reduce_mean(self.conv6, axis=3, keep_dims=True)
    self.mask_conv6 = tf.multiply(self.ave_conv6, self.deconv7);
    self.deconv6 = tf.nn.conv2d_transpose(self.mask_conv6, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 56, 56, 1], strides=[1, 1, 1, 1], padding="SAME")
    
    self.ave_conv5 = tf.reduce_mean(self.conv5, axis=3, keep_dims=True)
    self.mask_conv5 = tf.multiply(self.ave_conv5, self.deconv6);
    self.deconv5 = tf.nn.conv2d_transpose(self.mask_conv5, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 56, 56, 1], strides=[1, 1, 1, 1], padding="SAME")
    # ************************
    self.ave_pool2 = tf.reduce_mean(self.pool2, axis=3, keep_dims=True)
    print "ave_pool2 shape: ", self.ave_pool2.shape
    self.mask_pool2 = tf.multiply(self.ave_pool2, self.deconv5)
    self.deconv_pool2 = tf.nn.conv2d_transpose(self.mask_pool2, tf.ones([2, 2, 1, 1]), output_shape=[self.batch_size, 112, 112, 1], strides=[1, 2, 2, 1], padding="VALID")
    print "deconv_pool2 shape: ", self.deconv_pool2.shape
    
    self.ave_conv4 = tf.reduce_mean(self.conv4, axis=3, keep_dims=True)
    self.mask_conv4 = tf.multiply(self.ave_conv4, self.deconv_pool2);
    self.deconv4 = tf.nn.conv2d_transpose(self.mask_conv4, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 112, 112, 1], strides=[1, 1, 1, 1], padding="SAME")

    self.ave_conv3 = tf.reduce_mean(self.conv3, axis=3, keep_dims=True)
    self.mask_conv3 = tf.multiply(self.ave_conv3, self.deconv4);
    self.deconv3 = tf.nn.conv2d_transpose(self.mask_conv3, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 112, 112, 1], strides=[1, 1, 1, 1], padding="SAME")
    # ************************
    self.ave_pool1 = tf.reduce_mean(self.pool1, axis=3, keep_dims=True)
    print "ave_pool1 shape: ", self.ave_pool1.shape
    self.mask_pool1 = tf.multiply(self.ave_pool1, self.deconv3)
    self.deconv_pool1 = tf.nn.conv2d_transpose(self.mask_pool1, tf.ones([2, 2, 1, 1]), output_shape=[self.batch_size, 224, 224, 1], strides=[1, 2, 2, 1], padding="VALID")
    print "deconv_pool1 shape: ", self.deconv_pool1.shape
    
    self.ave_conv2 = tf.reduce_mean(self.conv2, axis=3, keep_dims=True)
    self.mask_conv2 = tf.multiply(self.ave_conv2, self.deconv_pool1);
    self.deconv2 = tf.nn.conv2d_transpose(self.mask_conv2, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 224, 224, 1], strides=[1, 1, 1, 1], padding="SAME")

    self.ave_conv1 = tf.reduce_mean(self.conv1, axis=3, keep_dims=True)
    self.mask_conv1 = tf.multiply(self.ave_conv1, self.deconv2);
    self.deconv1 = tf.nn.conv2d_transpose(self.mask_conv1, tf.ones([3, 3, 1, 1]), output_shape=[self.batch_size, 224, 224, 1], strides=[1, 1, 1, 1], padding="SAME")

    # ***********************
    self.mask = (self.deconv1) / (tf.reduce_max(self.deconv1) + 1e-5)
    print "mask shape: ", self.mask.shape

  def load_initial_weights(self, session):
    all_vars = tf.trainable_variables()
    # Load the weights into memory
    weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()
    for name in weights_dict:
      print "restoring var {}...".format(name)
      var = [var for var in all_vars if var.name == name][0]
      session.run(var.assign(weights_dict[name]))

            
     
  def save_weights(self, session, file_name='pretrain.npy'):
    save_vars = tf.trainable_variables()
    weights_dict = {}
    for var in save_vars:
      weights_dict[var.name] = session.run(var)
    np.save('pretrain.npy', weights_dict) 
    print "weights saved in file {}".format(file_name)
  
"""
Predefine all necessary layer for the AlexNet
""" 
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
  """
  Adapted from: https://github.com/ethereon/caffe-tensorflow
  """
  # Get number of input channels
  input_channels = int(x.get_shape()[-1])
  
  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)
  
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])  
    
    
    if groups == 1:
      conv = convolve(x, weights)
      
    # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
      
      # Concat the convolved output together again
      conv = tf.concat(axis = 3, values = output_groups)
      
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)
        
    return relu
  
def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)      
      return relu
    else:
      return act
    

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)
  
def lrn(x, radius, alpha, beta, name, bias=1.0):
  return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)
  
def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)
  
    
