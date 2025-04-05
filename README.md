ToDo

Whistles, Clicks, BPs
- Create Spectrogram Datset
    - Reduce Noise, Like Audaicty, follow Paper routine
- Create Latent Space with VAEs
- Generative Adversarial Networks (GANs)
    - WaveGAN: train on dolphin sounds dataset to generate
    - Convert Newly generated spectorgrams back to audio using Griffin-Lim Algorithm

Latent Space Creation:
- PCA, t-SNE, UMAP, AE
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
  


 
Generate More Data?
- Pitch shifting
- Speed/Slow Down
- Gaussian Noise
- Reverberation

- 
  
