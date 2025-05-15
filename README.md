ToDo

Had: 2.37Million parameters
Down: 403

Orginally 50 components
Variance explained by each component:
PC1: 15.97%
PC2: 9.09%
PC3: 3.94%

Total variance explained by 3 components: 28.99%
This is quite revealing! The fact that 50 components explain the same variance as 3 components suggests that:
Most of the important information is captured in the first few components




Whistles, Clicks, BPs
- Create Spectrogram Datset
    - Reduce Noise, Like Audaicty, follow Paper routine
- Create Latent Space with VAEs
- Generative Adversarial Networks (GANs)
    - WaveGAN: train on dolphin sounds dataset to generate
    - Convert Newly generated spectorgrams back to audio using Griffin-Lim Algorithm

Latent Space Creation:
- VAE: dimsionality reduction (can be revetred unlike PCA, t-SNE)
      - Encode: latent vector (maybe 32D)
      - Latent Space
      - Decode: Reconsturcuts the spectorgam from the latent Vecotr
      - Back to Audio: Inversel mel or Griffin-Lim Algorithm
- Could also do K-Means, DBSCAN, HDBSCAN for One vs all method
Classification
- One vs All: Whistles vs CLicks/BPs
      - Tonal vs non-tonal (impulsive)
      - Binary Classifier (SVM, logestic regression, shallow NN)
- Finetune for clicks/BPs
      - Zoom in:
        -Clicks: isolated, regular timingm echolocation
        -BPS:  grouped, less regular, communcation
      - Methods: Inter click intervals (ICIs), Envelope Shapes, Specteral centroid/flatness


  Jan 11: Whistles, Clicks
  


 
Generate More Data?
- Pitch shifting
- Speed/Slow Down
- Gaussian Noise
- Reverberation

- 
  
