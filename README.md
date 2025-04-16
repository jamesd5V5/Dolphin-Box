ToDo

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
  
