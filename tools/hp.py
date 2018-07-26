class HP:
    
    persianvocab='PE !.:،؟ءآئابتثجحخدذرزسشصضطظعغفقلمنهوىًَُِّْپچژکگی'

    embeding_num_units=128
    d=256#Hidden units for text 2 mel according to the paper
    c=512#Hidden units for super resution network according to the paper
    dropout_rate=0
    
    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 1024  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = 254  # samples.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 150  # Number of inversion iterations
    preemphasis = .97
    sharpening_factor=1.3
    max_db = 100
    min_db=-100
    ref_db = 20
    r = 4 # Reduction factor
    logdir = None
    
    init_learinig_rate=4e-4
    batch_size=4
    Max_Number_Of_Chars = 180 
    Max_Number_Of_MelFrames = 210