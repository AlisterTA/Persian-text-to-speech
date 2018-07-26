import sys
import tensorflow as tf
sys.path.append('tools')
from hp import HP
def attention(K,V,Q,num_hidden_units,mononotic_attention=False, prev_max_attentions=None):
    A=tf.matmul(Q,tf.transpose(K,[0,2,1]))/tf.sqrt(tf.to_float(num_hidden_units))
    A=tf.nn.softmax(A)
    R=tf.matmul(A,V)
    #they found it benefical for whatever reasons?
    R = tf.concat((R, Q), -1)
    #max_attentions = tf.argmax(A, -1)
    #alignments = tf.transpose(A, [0, 2, 1])
    return R,A
    