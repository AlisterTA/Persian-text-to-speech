# Persian-text-to-speech

**A convolutional sequence to sequence model for Persian text to speech based on [Tachibana et al](https://arxiv.org/abs/1710.08969) with a few modifications:**

1)	The article didn’t mention position embedding, but in order to give the model a sense of position awareness I added it (I turned out to be useful in the original [convolutional seq2seq paper](https://arxiv.org/abs/1705.03122)

2)	In the paper they trained both networks with a combination of the L1 loss and an additional binary cross entropy loss , they claim it was beneficial, I found it to be an odd choice of loss function.to validate their idea I trained networks with and without binary cross entropy loss , but adding binary cross entropy loss didn’t make much difference.

3)	In the original paper they used a fixed learning rate of 0.001, but I decayed it.according to the [ADAM](https://arxiv.org/abs/1412.6980) paper Good default settings for the tested machine learning problems are α = 0.001,β1 = 0.9, β2 = 0.999 but in my case it didn't work!(gradients keep exploding)

4)	I implemented a standard scaled dot-product attention but it mostly failed to converge. Guided attention is a simple but good idea such that the model converge way faster than a standard scaled dot-product attention. Training the attention part was a bottleneck.


In the following figure a schematic of the model architecture is presented:

![text to mel](/imgs/texttomel.jpg)

**Dataset: (a Persian single speaker speech dataset that last more than 30 hours [narrated by Maryam Mahboub])**

There aren’t any available datasets for Persian text to speech so I decided to make my own. I chose to use audio books from [navar](www.navaar.ir) (I couldn’t find any websites for free public domain audiobooks) then I checked for text availability, if text is not available the audio book is excluded. 
None of them were available so I $$$$ a few online bookstores like [Fidibo](http://fidibo.com/) to get them. All audio files are sampled at 44 kHz which means there are 44100 values stored for every second of audio, too much information, WAY TOO MUCH! So I down sampled the pcm/wav audio samples from 44 kHz to 22 kHz. I also discarded the stereo channel as it contains highly redundant information.

The audio from  [navar](www.navaar.ir) comes in large files which don’t suit the text to speech task. At first I decided to use automatic force-alignment technique (given an audio clip containing speech [without environmental noise] and the corresponding transcript, computing a forced alignment is the process of determining, for each fragment of the transcript, the time interval containing the spoken text of the fragment) to align the text corpus with audio clips but it didn’t guarantee the correct alignments because texts were slightly different than speech. I finally ended up pragmatically splitting large chunk of audio files into smaller parts by using silence detection and manually aligned text with audio segments.

The distribution of audio lengths for my dataset is shown in the following figure: (90 percentage of the samples have a duration between 4 and 10 seconds)
imageeee



To speed up the training speech and reduce CPU usage I first preprocessed all audio files (using Fourier’s Transform to convert our audio data to the frequency domain). Extracting the audio spectrogram on the fly is expensive. 

I implemented the models in Tensorflow and they were trained on a single Nvidia GTX 1050Ti with 4GB memory (a batch size of 32 for TEXT2MEL and 8 for SSRN ).


In the following figure the learned character embeddings is shown:

![text to mel](/imgs/char-embedding.jpg)

**Some generated samples:**

**Pre-trained model:**

You can download pre-trained weights from here

**Script files**

if you want to train and test your own datasets:
Modify train.ipynb including args and data path

demo.ipynb:Enter your sentences and listen to the generated audio
