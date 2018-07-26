import sys
import tensorflow as tf
sys.path.append('tools')
sys.path.append('network')
from read_data import load_data_main,get_batch,get_batch_npz
from hp import HP
from text_encoder import textencoder,embeding_layer
from audio_encoder import audioencoder
from audio_decoder import audiodecoder
from attention import attention
from SuperResolution import super_resolution
from scipy.io.wavfile import write
from read_data import load_data_synthesize
from wavprepro import spectrogram2wav
import numpy as np
from tqdm import tqdm


'''
if you're not sure how to structure your tensorFlow model especially when it comes to working with multiple graphs in TensorFlow I highly recommend reading the following blog posts,they help you to tease out the TensorFlow Graph Mess.(at least you wouldn't get lost when you look through your TensorBoard graph)

[1]https://danijar.com/structuring-your-tensorflow-models/
[2]https://gist.github.com/Breta01/cabbb5c7d9bbd3d9b4ec404828ac24bb

'''
class model(object):
    def __init__(self, data_path,mode):
        
        if mode=='training_superresolution':
            self.mels, self.mags, self.fnames,self.num_batch=get_batch_npz(data_path,
                                                                           'metadata.csv',HP.batch_size,HP.n_mels,HP.n_fft,mode=1)
            
            with tf.variable_scope('Super_Resolution_Network'):
                self.logits,self.Z=super_resolution(input_tensor=self.mels,dropout_rate=
                                                    HP.dropout_rate,num_hidden_layers=HP.c,n_fft=HP.n_fft)
                
        if mode=='training_text2sp':
            self.texts, self.mels, self.fnames,self.num_batch=get_batch_npz(data_path,
                                                                                  'metadata.csv',HP.batch_size,HP.n_mels,HP.n_fft,mode=2)
            self.L=embeding_layer(inputtextids=self.texts,emdeding_size=HP.embeding_num_units,
                             vocab_size=len(HP.persianvocab),scope_name="embeding")
            #one preview shifted target input
            self.S = tf.pad(self.mels[:,:-1,:],[[0,0],[1,0],[0,0]])

            with tf.variable_scope("Text_Encoder"):
                self.K,self.V = textencoder(embeding_tensor=self.L,dropout_rate=HP.dropout_rate,num_hidden_layers=HP.d)
            
            with tf.variable_scope("Audio_Encoder"):
                self.Q = audioencoder(input_tensor=self.S,dropout_rate=HP.dropout_rate,num_hidden_layers=HP.d)

            with tf.variable_scope("Attention"):
                 self.R,self.A = attention(self.K,self.V, self.Q,HP.d)

            with tf.variable_scope("Audio_Decoder"):
                self.logits,self.Y = audiodecoder(self.R,dropout_rate=HP.dropout_rate,num_hidden_layers=HP.d,num_mels=HP.n_mels) 
        if mode=='training_superresolution' or mode=='training_text2sp':    
            #In order to keep track of how far we are in the training, we use one of Tensorflow’s training utilities, the global_step.
            with tf.variable_scope("gs"):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
           
        if mode=='training_superresolution' or mode=='training_text2sp':    
            with tf.variable_scope('Loss_Operations'):
                if mode=='training_text2sp':
                    # mel L1 loss
                    self.l1_distance_loss= tf.reduce_mean(tf.abs(self.mels-self.Y ))
                    self.binary_divergence_loss= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                                (logits=self.logits, labels=self.mels))
                
                    #attention loss(guided attention) Refer to page 3 on the paper
                    N, T = tf.cast(tf.shape(self.A)[1],tf.float32), tf.cast(tf.shape(self.A)[2],tf.float32)
                    W = tf.fill(tf.shape(self.A),0.0) 
                    W = W + tf.expand_dims(tf.range(N),1)/N - tf.expand_dims(tf.range(T),0)/T 
                    self.att_W = 1.0 - tf.exp(-tf.square(W)/(2*0.2)**2) 
                    self.loss_att = tf.reduce_mean(tf.multiply(self.A,self.att_W))

                    # total loss
                    self.total_loss = self.l1_distance_loss+self.binary_divergence_loss  + self.loss_att
                    tf.summary.scalar('l1_distance_loss', self.l1_distance_loss)
                    tf.summary.scalar('binary_divergence_loss', self.binary_divergence_loss)
                    tf.summary.scalar('loss_att', self.loss_att)
                    tf.summary.scalar('total_loss', self.total_loss)
                    tf.summary.image('mel', tf.expand_dims(tf.transpose(self.mels[:1], [0, 2, 1]), -1))
                    tf.summary.image('predicted_mel', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))
                    tf.summary.image('plot_attention', tf.expand_dims(tf.transpose(self.A[:1], [0, 2, 1]), -1))
                
                if mode=='training_superresolution':
                    self.l1_distance_loss = tf.reduce_mean(tf.abs(self.mags-self.Z ))
                    self.binary_divergence_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                                 (logits=self.logits, labels=self.mags))
                    self.total_loss=self.l1_distance_loss+self.binary_divergence_loss
                    #tensorboard information
                    tf.summary.scalar('l1_distance_loss', self.l1_distance_loss)
                    tf.summary.scalar('binary_divergence_loss', self.binary_divergence_loss)
                    tf.summary.scalar('total_loss', self.total_loss)
                    tf.summary.image('mags', tf.expand_dims(tf.transpose(self.mags[:1], [0, 2, 1]), -1))
                    tf.summary.image('predicted_mag', tf.expand_dims(tf.transpose(self.Z[:1], [0, 2, 1]), -1))
        if mode=='training_superresolution' or mode=='training_text2sp':    
        
            # Define training step that minimizes the loss with the Adam optimizer
            with tf.variable_scope('optimizer_scope'):
                #for more information :
                #https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer    
                step = tf.to_float(self.global_step + 1)
                lr=HP.init_learinig_rate * 4000.0**0.5 * tf.minimum(step * 4000.0**-1.5, step**-0.5)
                #optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
                #accordint to the ADAM paper Good default settings for the tested machine 
                #learning problems are α = 0.001,β1 = 0.9, β2 = 0.999             
                #and  = 10−8 but in my case it didn't work!
                optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1= 0.6,beta2=0.95)
                tf.summary.scalar("lr", lr)
                #do gradient cliping
                grts = optimizer.compute_gradients(self.total_loss)
                clipped = []
                for grad, var in grts:
                    grad = tf.clip_by_value(grad, -1., 1.)
                    clipped.append((grad, var))
                    
                self.train_operation = optimizer.apply_gradients(clipped, global_step=self.global_step)
        if mode=='demo':
            self.INP = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, HP.n_mels))
            L=embeding_layer(inputtextids=self.INP,emdeding_size=HP.embeding_num_units,
                             vocab_size=len(HP.persianvocab),scope_name="embeding")
            with tf.variable_scope("Text_Encoder"):
                K,V = textencoder(embeding_tensor=L,dropout_rate=HP.dropout_rate,num_hidden_layers=HP.d)
                
            with tf.variable_scope("Audio_Encoder"):
                Q = audioencoder(input_tensor=self.mels,dropout_rate=HP.dropout_rate,num_hidden_layers=HP.d)

            with tf.variable_scope("Attention"):
                R,A = attention(K,V, Q,HP.d)
            with tf.variable_scope("Audio_Decoder"):
                Ylogit, self.Y = audiodecoder(R,dropout_rate=HP.dropout_rate ,num_hidden_layers=HP.d,num_mels=HP.n_mels) 
            with tf.variable_scope('Super_Resolution_Network'):
                logits,self.Z=super_resolution(input_tensor=self.Y ,dropout_rate=
                                                    HP.dropout_rate,num_hidden_layers=HP.c,n_fft=HP.n_fft)
            self.sess=tf.Session()
            self.sess.run(tf.global_variables_initializer())

            text2sp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text_Encoder|Audio_Encoder|Audio_Decoder|embeding')

            superresolution_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Super_Resolution_Network')

            sv1 = tf.train.Saver(var_list=text2sp_vars)
            sv1.restore(self.sess, tf.train.latest_checkpoint('logs/text-to-spec'))

            sv2 = tf.train.Saver(var_list=superresolution_vars)
            sv2.restore(self.sess, tf.train.latest_checkpoint('logs/super_resolution'))
            print('model loaded :)')
        tf.summary.merge_all()
    
    def predict(self,lines):
        lines = [item.replace("آ","آ").replace("أ","ا").replace("ئ","ی").replace("ؤ","و") for item in lines]
        Input_Text=load_data_synthesize(lines,HP.Max_Number_Of_Chars)
        predicted_mel = np.zeros((Input_Text.shape[0],HP.Max_Number_Of_MelFrames,HP.n_mels)) 
        predicted_mag = np.zeros((Input_Text.shape[0],HP.Max_Number_Of_MelFrames,HP.c+1))
        print('predicting ( . •́ _ʖ •̀ .) ...')
        #TODO:Implement mononotic attention
        for i in tqdm(range(1,HP.Max_Number_Of_MelFrames)):
            previous_slice = predicted_mel[:,:i,:]
            model_out = self.sess.run(self.Y,{
                          self.mels: previous_slice,self.INP: Input_Text})
            predicted_mel[:,i,:] = model_out[:,-1,:]
        _Z = self.sess.run(self.Z, {self.Y: predicted_mel})
        
        print('converting the generated spectrogram to audio ｖ(⌒ｏ⌒)ｖ♪ ...')
        for i, mag in enumerate(_Z):
            wav = spectrogram2wav(mag)
            write('generated_samples/' + "/{}.wav".format(i+1),HP.sr,wav.astype(np.float32))
        print('DONE!')