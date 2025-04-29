from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from CNN import CNN
from PrepData import PrepData
from DataExtraction import get_all_spectrograms
from Classification import train
from Classification import get_predictions

from sklearn.metrics import accuracy_score, classification_report
from copy import deepcopy

def run_k_fold(data, k=5, batch_size=16, epochs=10):
    labels = [label for _, label in data]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    all_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(data)), labels)):
        print(f"\n--- Fold {fold + 1} ---")

        train_subset = [data[i] for i in train_idx]
        test_subset = [data[i] for i in test_idx]

        train_dataset = PrepData(train_subset)
        test_dataset = PrepData(test_subset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Create a new model instance for each fold
        model = CNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        # Train model
        train(model, train_loader, optimizer, criterion, epochs=epochs)

        # Evaluate
        y_true, y_pred = get_predictions(model, test_loader)
        acc = accuracy_score(y_true, y_pred)
        print(f"Fold {fold+1} Accuracy: {acc*100:.2f}%")
        all_accuracies.append(acc)

        # Optional: print class report
        print(classification_report(y_true, y_pred, target_names=["Click", "Whistle"]))

    print(f"\nAverage Accuracy over {k} folds: {np.mean(all_accuracies)*100:.2f}%")

data = get_all_spectrograms()
run_k_fold(data, k=5, epochs=10)