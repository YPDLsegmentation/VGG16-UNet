import tensorflow as tf
import numpy as np

class NET(object):
  
  def __init__(self, x, height, width, 
               keep_prob, skip_layer, out_channels = 1,  
               weights_path = 'DEFAULT', do_vbp = False, batch_size = 64):
    
    # Parse input arguments into class variables
    self.X = x
    self.height = height
    self.width = width
    self.batch_size = batch_size
    self.KEEP_PROB = keep_prob
    self.SKIP_LAYER = skip_layer
    self.out_channels = out_channels
    
    if weights_path == 'DEFAULT':      
      self.WEIGHTS_PATH = 'vgg16.npy'
    else:
      self.WEIGHTS_PATH = weights_path
    
    # Call the create function to build the computational graph of Net
    self.create()
    if do_vbp:
      self.vbp()
    
  def create(self):
    
    #encoder
    #input size [batch_size, w, h, 3]
    self.conv1_1 = conv(self.X, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv1_1') # [~, w, h, 64]
    self.conv1_2 = conv(self.conv1_1, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv1_2') # [~, w, h, 64]
    self.pool1 = max_pool(self.conv1_2, 2, 2, 2, 2, padding = 'SAME', name = 'pool1') # [~, w/2, h/2, 64]
    print "conv1_1 shape: {}".format(self.conv1_1.shape)
    print "conv1_2 shape: {}".format(self.conv1_2.shape)
    print "pool1 shape: {}".format(self.pool1.shape)
    
    self.conv2_1 = conv(self.pool1, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv2_1') # [~, w/2, h/2, 128]
    self.conv2_2 = conv(self.conv2_1, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv2_2') # [~, w/2, h/2, 128]
    self.pool2 = max_pool(self.conv2_2, 2, 2, 2, 2, padding = 'SAME', name ='pool2') # [~, w/4, h/4, 128]
    print "conv2_2 shape: {}".format(self.conv2_1.shape)
    print "conv2_2 shape: {}".format(self.conv2_2.shape)
    print "pool2 shape: {}".format(self.pool2.shape)
    
    self.conv3_1 = conv(self.pool2, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv3_1') # [~, w/8, h/8, 256]
    self.conv3_2 = conv(self.conv3_1, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv3_2') # [~, w/4, h/4, 256]
    self.conv3_3 = conv(self.conv3_2, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv3_3') # [~, w/4, h/4, 256]
    self.pool3 = max_pool(self.conv3_3, 2, 2, 2, 2, padding = 'SAME', name ='pool3') # [~, w/8, h/8, 256]
    print "conv3_1 shape: {}".format(self.conv3_1.shape)
    print "conv3_2 shape: {}".format(self.conv3_2.shape)
    print "conv3_3 shape: {}".format(self.conv3_3.shape)
    print "pool3 shape: {}".format(self.pool3.shape)

    self.conv4_1 = conv(self.pool3, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv4_1') # [~, w/8, h/8, 512]
    self.conv4_2 = conv(self.conv4_1, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv4_2') # [~, w/8, h/8, 512]
    self.conv4_3 = conv(self.conv4_2, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv4_3') # [~, w/8, h/8, 512]
    self.pool4 = max_pool(self.conv4_3, 2, 2, 2, 2, padding = 'SAME', name ='pool4') # [~, w/16, h/16, 512]
    print "conv4_1 shape: {}".format(self.conv4_1.shape)
    print "conv4_2 shape: {}".format(self.conv4_2.shape)
    print "conv4_3 shape: {}".format(self.conv4_3.shape)
    print "pool4 shape: {}".format(self.pool4.shape)
    
    self.conv5_1 = conv(self.pool4, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv5_1') # [~, w/16, h/16, 512]
    self.conv5_2 = conv(self.conv5_1, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv5_2') # [~, w/16, h/16, 512]
    self.conv5_3 = conv(self.conv5_2, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv5_3') # [~, w/16, h/16, 512]
    self.pool5 = max_pool(self.conv5_3, 2, 2, 2, 2, padding = 'SAME', name = 'pool5') # [~, w/32, h/32, 512]
    print "conv5_1 shape: {}".format(self.conv5_1.shape)
    print "conv5_2 shape: {}".format(self.conv5_2.shape)
    print "conv5_3 shape: {}".format(self.conv5_3.shape)
    print "pool5 shape: {}".format(self.pool5.shape)

    #decoder
    self.deconv1 = deconv(self.pool5, 3, 3, 512, 2, 2, output_shape=[self.batch_size, self.height/16, self.width/16, 512], padding='SAME', name = 'deconv1') #[~, w/16, h/16, 512]
    self.norm1 = norm_rescale(self.pool4, 512, 'norm1')                                                                                                      #[~, w/16, h/16, 512]
    self.concat1 = tf.concat([self.deconv1, self.norm1], axis=3)                                                                                             #[~, w/16, h/16, 1024]
    print "deconv1 shape: {}".format(self.deconv1.shape)
    print "norm1 shape: {}".format(self.norm1.shape)
    print "concat1 shape: {}".format(self.concat1.shape)

    self.conv6_1 = conv(self.concat1, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv6_1') # [~, w/16, h/16, 512]
    self.conv6_2 = conv(self.conv6_1, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv6_2') # [~, w/16, h/16, 256]
    print "conv6_1 shape: {}".format(self.conv6_1.shape)
    print "conv6_2 shape: {}".format(self.conv6_2.shape)
    
    self.deconv2 = deconv(self.conv6_2, 3, 3, 256, 2, 2, output_shape=[self.batch_size, self.height/8, self.width/8, 256], padding='SAME', name = 'deconv2') #[~, w/8, h/8, 256]
    self.norm2 = norm_rescale(self.pool3, 256, 'norm2')                                                                                                      #[~, w/8, h/8, 256]
    self.concat2 = tf.concat([self.deconv2, self.norm2], axis=3)                                                                                             #[~, w/8, h/8, 512]
    print "deconv2 shape: {}".format(self.deconv2.shape)
    print "norm2 shape: {}".format(self.norm2.shape)
    print "concat2 shape: {}".format(self.concat2.shape)
    
    self.conv7_1 = conv(self.concat2, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv7_1') # [~, w/8, h/8, 256]
    self.conv7_2 = conv(self.conv7_1, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv7_2') # [~, w/8, h/8, 128]
    print "conv7_1 shape: {}".format(self.conv7_1.shape)
    print "conv7_2 shape: {}".format(self.conv7_2.shape)

    self.deconv3 = deconv(self.conv7_2, 3, 3, 128, 2, 2, output_shape=[self.batch_size, self.height/4, self.width/4, 128], padding='SAME', name = 'deconv3') #[~, w/4, h/4, 128]
    self.norm3 = norm_rescale(self.pool2, 128, 'norm3')                                                                                                      #[~, w/4, h/4, 128]
    self.concat3 = tf.concat([self.deconv3, self.norm3], axis=3)                                                                                             #[~, w/4, h/4, 256]
    print "deconv3 shape: {}".format(self.deconv3.shape)
    print "norm3 shape: {}".format(self.norm3.shape)
    print "concat3 shape: {}".format(self.concat3.shape)

    self.conv8_1 = conv(self.concat3, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv8_1') # [~, w/4, h/4, 128]
    self.conv8_2 = conv(self.conv8_1, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv8_2') # [~, w/4, h/4, 64]
    print "conv8_1 shape: {}".format(self.conv8_1.shape)
    print "conv8_2 shape: {}".format(self.conv8_2.shape)
    
    self.deconv4 = deconv(self.conv8_2, 3, 3, 64, 2, 2, output_shape=[self.batch_size, self.height/2, self.width/2, 64], padding='SAME', name = 'deconv4') #[~, w/2, h/2, 64]
    self.norm4 = norm_rescale(self.pool1, 64, 'norm4')                                                                                                     #[~, w/2, h/2, 64]
    self.concat4 = tf.concat([self.deconv4, self.norm4], axis=3)                                                                                           #[~, w/2, h/2, 128]
    print "deconv4 shape: {}".format(self.deconv4.shape)
    print "norm4 shape: {}".format(self.norm4.shape)
    print "concat4 shape: {}".format(self.concat4.shape)
    
    self.conv9_1 = conv(self.concat4, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv9_1') # [~, w/2, h/2, 64]
    print "conv9_1 shape: {}".format(self.conv9_1.shape)

    self.deconv5 = deconv(self.conv9_1, 3, 3, self.out_channels, 2, 2, output_shape=[self.batch_size, self.height, self.width, 64], padding='SAME', name = 'deconv5') #[~, w, h, 64]
    self.conv10_1 = conv(self.deconv5, 1, 1, self.out_channels, 1, 1, padding = 'SAME', non_linear="NONE", name='conv10_1')#[~, w, h, out_channels(<64)]
    print "deconv5 shape: {}".format(self.deconv5.shape)
    print "conv10_1 shape: {}".format(self.conv10_1.shape)

  def vbp(self):
    pass

  def load_initial_weights(self, session, restore_vars=None):
    if restore_vars is None:
      return
    # Load the weights into memory
    weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()
    for var in restore_vars:
      name = var.name
      if weights_dict.has_key(name):
        print "restoring var {}...".format(name)
        session.run(var.assign(weights_dict[name]))
      else:
        print "no value for var {}".format(name)
     
  def save_weights(self, session, file_name='pretrain.npy', save_vars=None):
    if save_vars is None:
      return
    weights_dict = {}
    for var in save_vars:
      weights_dict[var.name] = session.run(var)
    np.save(file_name, weights_dict) 
    print "weights saved in file {}".format(file_name)
  
"""
Predefine all necessary layer for the Model
""" 
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1, non_linear="RELU"):
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
    
    # Apply non-linear function
    if non_linear == "RELU":
        nonlin = tf.nn.relu(bias, name = scope.name)
    elif non_linear == "SIGMOID":
        nonlin = tf.sigmoid(bias, name = scope.name)
    elif non_linear == 'NONE':
        nonlin = tf.identity(bias, name = scope.name)  
        
    return nonlin
  
def deconv(x, filter_height, filter_width, num_filters, stride_y, stride_x, output_shape, name,
           padding='SAME', non_linear="RELU"):

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the deconv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, num_filters, input_channels])
    biases = tf.get_variable('biases', shape = [num_filters])  
    
    dconv = tf.nn.conv2d_transpose(x, weights, output_shape, [1, stride_y, stride_x, 1], padding)
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(dconv, biases), dconv.get_shape().as_list())
    # Apply non-linear function
    if non_linear == "RELU":
        nonlin = tf.nn.relu(bias, name = scope.name)
    elif non_linear == "SIGMOID":
        nonlin = tf.sigmoid(bias, name = scope.name)
    elif non_linear == "SOFTMAX":
        nonlin = tf.nn.softmax(bias, dim=-1, name = scope.name)
    elif non_linear == 'NONE':
        nonlin = tf.identity(bias, name = scope.name)  
        
    return nonlin

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
  
def norm_rescale(x, channels, name):
    with tf.variable_scope(name) as scope:
        scale = tf.get_variable('scale', shape=[channels], trainable=True, dtype=tf.float32)
        return scale * tf.nn.l2_normalize(x, dim=[1, 2]) # NOTE: per feature map normalizatin
    
