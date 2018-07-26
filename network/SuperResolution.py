import tensorflow as tf
from layers import conv1d,highwaynet,deconv

#gets mel as input (time/reduction factor) and spits out the upsampled meg 
def super_resolution(input_tensor,dropout_rate,num_hidden_layers,n_fft):
    L1=conv1d(input_tensor=input_tensor,filters=num_hidden_layers,kernel_size=1,strides=1,padding="SAME",dilation_rate=1,
       activation=None,dropout_rate=dropout_rate)

    L2=highwaynet(input_tensor=L1,filters=None,kernel_size=3,
                         strides=1,padding="SAME",dilation_rate=1,
                         activation=None,dropout_rate=dropout_rate,scope_name="SRNET_highwaynet_Block"+str(1))
    L3=highwaynet(input_tensor=L2,filters=None,kernel_size=3,
                         strides=1,padding="SAME",dilation_rate=3,
                         activation=None,dropout_rate=dropout_rate,scope_name="SRNET_highwaynet_Block"+str(2))
    
    L4=deconv(input_tensor=L3,filters=None,kernel_size=3,strides=2,
              padding='SAME',activation=None,dropout_rate=dropout_rate)    
    L5=highwaynet(input_tensor=L4,filters=None,kernel_size=3,
        strides=1,padding="SAME",dilation_rate=1,
        activation=None,dropout_rate=dropout_rate,scope_name="SRNET_highwaynet_Layer2"+str(3))
    L6=highwaynet(input_tensor=L5,filters=None,kernel_size=3,
        strides=1,padding="SAME",dilation_rate=3,
        activation=None,dropout_rate=dropout_rate,scope_name="SRNET_highwaynet_Layer2_1"+str(4))
        
    L7=deconv(input_tensor=L6,filters=None,kernel_size=3,strides=2,
              padding='SAME',activation=None,dropout_rate=dropout_rate)    
    L8=highwaynet(input_tensor=L7,filters=None,kernel_size=3,
        strides=1,padding="SAME",dilation_rate=1,
        activation=None,dropout_rate=dropout_rate,scope_name="SRNET_highwaynet_Layer2"+str(5))
    L9=highwaynet(input_tensor=L8,filters=None,kernel_size=3,
        strides=1,padding="SAME",dilation_rate=3,
        activation=None,dropout_rate=dropout_rate,scope_name="SRNET_highwaynet_Layer2_1"+str(6))
        
    L10=conv1d(input_tensor=L9,filters=2*num_hidden_layers,kernel_size=1,strides=1,padding="SAME",dilation_rate=1,
       activation=None,dropout_rate=dropout_rate)
        
    L11=highwaynet(input_tensor=L10,filters=None,kernel_size=3,
                         strides=1,padding="SAME",dilation_rate=1,
                         activation=None,dropout_rate=dropout_rate,scope_name="SRNET_highwaynet_Block"+str(7))
    
    L12=highwaynet(input_tensor=L11,filters=None,kernel_size=3,
                         strides=1,padding="SAME",dilation_rate=1,
                         activation=None,dropout_rate=dropout_rate,scope_name="SRNET_highwaynet_Block"+str(8))
        
    #????1+n_fft//2
    L13=conv1d(input_tensor=L12,filters=1+n_fft//2,kernel_size=1,strides=1,padding="SAME",dilation_rate=1,
       activation=None,dropout_rate=dropout_rate)
    
    L14=conv1d(input_tensor=L13,filters=None,kernel_size=1,strides=1,padding="SAME",dilation_rate=1,
             activation=tf.nn.relu,dropout_rate=dropout_rate)
    L15=conv1d(input_tensor=L14,filters=None,kernel_size=1,strides=1,padding="SAME",dilation_rate=1,
             activation=tf.nn.relu,dropout_rate=dropout_rate)
       

        
    logits=conv1d(input_tensor=L15,filters=None,kernel_size=1,strides=1,padding="SAME",dilation_rate=1,
       activation=None,dropout_rate=dropout_rate)
    
    Z=tf.nn.sigmoid(logits)

    return logits,Z
    