import numpy as np
from hp import HP
import unicodedata
import re
import os
import codecs
import tensorflow as tf
from wavprepro import load_spectrograms,load_spectrograms_npz

def load_vocab():
    char2index={char:indx for indx,char in enumerate(HP.persianvocab)}
    index2char={indx:char for indx,char in enumerate(HP.persianvocab)}
    return char2index,index2char

def text_norm(text):
    #remove all the "diacritic"
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn')
    
    text = text.lower()
    text = re.sub("[^{}]".format(HP.persianvocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(text):
    char2index,index2char=load_vocab()
    text+='E'
    return np.asarray([char2index[i] for i in text],np.int32)

def load_data_synthesize(lines,Max_Number_Of_Chars):
    char2index,index2char=load_vocab()
    sents = [text_norm(line) + "E" for line in lines]
    print(sents)
    texts = np.zeros((len(sents), Max_Number_Of_Chars), np.int32)
    for i, sent in enumerate(sents):
        texts[i, :len(sent)] = [char2index[char] for char in sent]
    return texts

def load_data_main(path,text_corpus_name):
    char2index,index2char=load_vocab()
    wavpaths,sentense_lenghts,sentenses=[],[],[]
    lines = codecs.open(os.path.join(path,text_corpus_name), 'r', 'utf-8').readlines()
    for line in lines:
            wname,sentense=line.strip().split('|')
            wpath = os.path.join(path,'wavs', wname + ".wav")
            wavpaths.append(wpath)
            norm_text=text_norm(line)+'E'
            convertedtoindx=[char2index[i] for i in norm_text]
            #i convert it to string because slice_input_producer expects it to be string
            sentenses.append(np.array(convertedtoindx, np.int32).tostring())
            sentense_lenghts.append(len(convertedtoindx))
    return wavpaths,sentense_lenghts,sentenses

def load_data_main_npz(path,text_corpus_name):
    char2index,index2char=load_vocab()
    wavpaths,sentense_lenghts,sentenses=[],[],[]
    lines = codecs.open(os.path.join(path,text_corpus_name), 'r', 'utf-8').readlines()
    for line in lines:
            wname,sentense=line.strip().split('|')
            wpath = os.path.join(path,'pickle', wname + ".npz")
            wavpaths.append(wpath)
            norm_text=text_norm(line)+'E'
            convertedtoindx=[char2index[i] for i in norm_text]
            #i convert it to string because slice_input_producer expects it to be string
            sentenses.append(np.array(convertedtoindx, np.int32).tostring())
            sentense_lenghts.append(len(convertedtoindx))
    return wavpaths,sentense_lenghts,sentenses
     
            
def get_batch(path,text_corpus_name,batch_size,n_mels,n_fft):
    with tf.device('/cpu:0'):
        fpaths, text_lengths, texts = load_data_main(path,text_corpus_name) # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        num_batch = len(fpaths) // batch_size

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        text = tf.decode_raw(text, tf.int32)  # (None,)
            
        parse_func = lambda path: load_spectrograms(path)
        fname, mel, mag = tf.py_func(parse_func, [fpath], [tf.string, tf.float32, tf.float32])  # (None, F)
        
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, n_mels))
        mag.set_shape((None, n_fft//2+1))

        # Batching for more details :https://github.com/wcarvalho/jupyter_notebooks/blob/ebe762436e2eea1dff34bbd034898b64e4465fe4/tf.bucket_by_sequence_length/bucketing%20practice.ipynb
        _, (texts, mels, mags, fnames)  = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=batch_size,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=batch_size*4,
                                            dynamic_pad=True)

    return  texts, mels, mags, fnames,num_batch

def get_batch_npz(path,text_corpus_name,batch_size,n_mels,n_fft,mode):
    with tf.device('/cpu:0'):
        fpaths, text_lengths, texts = load_data_main_npz(path,text_corpus_name) # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        num_batch = len(fpaths) // batch_size

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        text = tf.decode_raw(text, tf.int32)  # (None,)
        if mode==1:    
            fname, mel, mag = tf.py_func(load_spectrograms_npz, [fpath,mode], [tf.string, tf.float32, tf.float32])  # (None, n_mels)
            fname.set_shape(())
            text.set_shape((None,))
            mel.set_shape((None, n_mels))
            mag.set_shape((None, n_fft//2+1))
        if mode==2:    
            fname, mel = tf.py_func(load_spectrograms_npz, [fpath,mode], [tf.string, tf.float32])  # (None, n_mels)
            fname.set_shape(())
            text.set_shape((None,))
            mel.set_shape((None, n_mels))
            


        # Batching for more details :https://github.com/wcarvalho/jupyter_notebooks/blob/ebe762436e2eea1dff34bbd034898b64e4465fe4/tf.bucket_by_sequence_length/bucketing%20practice.ipynb
        if mode==1:
            _, (mels, mags, fnames)  = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[mel, mag, fname],
                                            batch_size=batch_size,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=batch_size*4,
                                            dynamic_pad=True)
            return  mels, mags, fnames,num_batch
        elif mode==2:
            _, (texts, mels, fnames)  = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, fname],
                                            batch_size=batch_size,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=batch_size*4,
                                            dynamic_pad=True)
            return  texts, mels, fnames,num_batch
            
            