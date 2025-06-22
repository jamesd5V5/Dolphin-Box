1) Introduction

  Humans hear from 20-20,000hz
  Dolphins hear from 0-150,000hz
  Meaning, we only hear a fraction of their range!
  Dolphin Communication has yet to be proven. But maybe machines can Learn?
  

2) Data Collection

  Thanks to Professor Christine Johnson (Cog Sci) and her lab, dolphins pairs were recorded using hydrophones at the San Diego Navy Base (192kHz SR / 91kHz BW)
  

3) Setting Up the Data

  LOTS OF NOISE: Machinery, equipment & even other dolphins. Hand picked clips of audio, ranging from 0.15-3 seconds. Ideally with classifications in pure Isolation. 
  
  Whistles, Clicks and Burst Pulses

  Loaded Mel Spectorgrams. Spliced tracks into more data. If over 0.25s, added padding if needed.
  EQs, CCA (Morphological filtering)
  Normalized (min-max scaling)
  Harsh db threshold to limit noise
  

4)  MFCCs + PCA

  Spectrograms & MFCCs contain a lot of information!
  With the help of PCA we are trying to find the best components (or combination of) that generate the maximum variance between   classifications 
  Higher Variance, Better Features

  Reduced to only 57 components
  

5) Train The Model

  Convert to Muli-hot encoding (Categorical Data to Numerical Data)
  Added Weights to Classifications due to imbalnce in amount of data 
  Whistle: 2 Click: 1.5 BP: 2 Noise: 4

  Optimizer: Adam 0.0005
  Weight decay: 1e-3 (L2 regularization) 
  Learning Rate Scheduler: ReduceLrOnPlateus (Reduced LR by half with loss plateaus)
  Gradient Clipping: max_norm = 0.5
  Batch Size: 32, Epochs: 40

MultiLabel CNN
  
  (57 features)
  3 Fully Connected Layers
  Each containing batch normalization
  FC1: 128u, FC2: 64u, FC3: 32u
  Dropout layers 0.4,0.4,0.3
  Dual Head Output
  Classification head (4 classes)
  Confidence head (probability score)
  

Loss Function
  
  Classification Loss: CrossEntropyLoss 
  Confidence Loss: BCEWithLogistsLoss 
  Total loss 
  Class Loss + 0.5 * Confidence Loss
  

6) Predict

  Splice Recordings into segments, Classifying Each Segment Individually with the Trained Model 
  Load Windows in Real Time

  Classifcation Accuracy 0.94%
  Whistles 0.88%
  Clicks 0.95%
  Burst Pulses 0.93%
  Noise 0.96%
