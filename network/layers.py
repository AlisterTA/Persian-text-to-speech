import tensorflow as tf

#https://www.w3cschool.cn/doc_tensorflow_python/tensorflow_python-tf-layers-conv1d.html
'''He initialization is implemented in variance_scaling_initializer(), but it didnt work for me (loss keeps blowing up) so i used Xavier initialization.i found the following questions useful:

https://stackoverflow.com/questions/43284047/what-is-the-default-kernel-initializer-in-tf-layers-conv2d-and-tf-layers-dense
https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-whatare'''
def conv1d(input_tensor,filters,kernel_size,strides,padding,dilation_rate,activation,dropout_rate):
    
    if padding.lower() == "causal":
        # pre-padding for causality
        #for more information :https://github.com/keras-team/keras/issues/8751
        pad_len = (kernel_size - 1) * dilation_rate  # padding size
        input_tensor = tf.pad(input_tensor, [[0, 0], [pad_len, 0], [0, 0]])
        padding = "valid"
        
    if filters==None:
        filters=input_tensor.get_shape().as_list()[-1]
    
    
    output=tf.layers.conv1d(
        input_tensor,
        filters,
        kernel_size,
        strides,
        padding,
        data_format='channels_last',
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None)
    if(dropout_rate>0):
        output =tf.contrib.layers.layer_norm(output,begin_norm_axis=-1)
        output = tf.layers.dropout(output, rate=dropout_rate)
    return output

# it's  actually simillar to gated convolution
#Highway(X; L) = σ(H1) * H2 + (1 − σ(H1))  X, where H1, H2 are properly-sized two matrices,
#output by a layer L as [H1, H2] = L(X). 
#The operator  is the element-wise multiplication, and σ is the element-wise sigmoid function.
def highwaynet(input_tensor,filters,kernel_size,strides,padding,dilation_rate,activation,dropout_rate,scope_name):
    with tf.variable_scope(scope_name,reuse=None):
        
            _input_tensor=input_tensor
            if filters==None:
                filters=input_tensor.get_shape().as_list()[-1]
            output=conv1d(input_tensor=input_tensor,filters=2*filters,
                            kernel_size=kernel_size,strides=strides,padding=padding, 
                            dilation_rate=dilation_rate,activation=activation,dropout_rate=dropout_rate)

            
            H1,H2=tf.split(output,2,axis=-1)
            if(dropout_rate>0):
                H1=tf.contrib.layers.layer_norm(H1,begin_norm_axis=-1)
                H2=tf.contrib.layers.layer_norm(H2,begin_norm_axis=-1)
            output=tf.nn.sigmoid(H1)*H2+(1.-tf.nn.sigmoid(H1))*_input_tensor
            if(dropout_rate>0):
                output = tf.layers.dropout(output, rate=dropout_rate)
    return output

#fancy deconvolution
def deconv(input_tensor,filters,kernel_size,strides,padding,activation,dropout_rate):
    
    if filters==None:
        filters=input_tensor.get_shape().as_list()[-1]
        
    #expand inputs dimension to make it suitable for 2d conv     
    input_tensor=tf.expand_dims(input_tensor, 1)
    output=tf.layers.conv2d_transpose(
        input_tensor,
        filters,
        (1,kernel_size),
        (1,strides),
        padding,
        data_format='channels_last',
        activation=activation,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None)
    
    output = tf.squeeze(output, 1)
    if(dropout_rate>0):
        output =tf.contrib.layers.layer_norm(output,begin_norm_axis=-1)
        output = tf.layers.dropout(output, rate=dropout_rate)
        
    return output