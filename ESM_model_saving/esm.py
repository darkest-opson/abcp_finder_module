#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

############################################################
# GPU SETUP
############################################################

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Available GPUs:", torch.cuda.device_count())
    print("Current device ID:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA not available, using CPU.")


############################################################
# LOAD DATA
############################################################

df = pd.read_csv("../ESM_works/ESM2_embeddings_training.csv")

X = df.drop("Target", axis=1)
y = df["Target"]

X_train = np.array(X)
y_train = np.array(y)

df_test = pd.read_csv("../ESM_works/ESM2_embeddings_testing.csv")

X_test = np.array(df_test.drop("Target", axis=1))
y_test = np.array(df_test["Target"])

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


############################################################
# MODEL
############################################################

class TorchMLPClassifier:

    def __init__(
        self,
        input_dim,
        num_classes=2,
        hidden_dim=128,
        num_layers=2,
        activation="relu",
        dropout=0.1,
        batch_norm=True,
        lr=0.001,
        weight_decay=0,
        batch_size=16,
        epochs=30,
    ):

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.train_losses = []
        self.train_accuracies = []

    def _activation(self):

        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "gelu":
            return nn.GELU()
        else:
            return nn.LeakyReLU()

    def _build_model(self):

        layers = []
        in_dim = self.input_dim

        for _ in range(self.num_layers):

            layers.append(nn.Linear(in_dim, self.hidden_dim))

            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))

            layers.append(self._activation())
            layers.append(nn.Dropout(self.dropout))

            in_dim = self.hidden_dim

        layers.append(nn.Linear(in_dim, self.num_classes))

        return nn.Sequential(*layers)

    def fit(self, X, y):

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model = self._build_model().to(self.device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        for epoch in range(self.epochs):

            self.model.train()

            total_loss = 0
            correct = 0
            total = 0

            for xb, yb in loader:

                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(xb)

                loss = criterion(outputs, yb)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                correct += (predicted == yb).sum().item()
                total += yb.size(0)

            epoch_loss = total_loss / len(loader)
            epoch_acc = correct / total

            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)

            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

        return self

    def predict_proba(self, X):

        self.model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():

            outputs = self.model(X_tensor)

            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def predict(self, X):

        probs = self.predict_proba(X)

        return np.argmax(probs, axis=1)


############################################################
# TRAIN MODEL
############################################################

model = TorchMLPClassifier(
    input_dim=X_train.shape[1],
    hidden_dim=128,
    num_layers=2,
    activation="relu",
    dropout=0.1,
    batch_norm=True,
    lr=0.001,
    weight_decay=0,
    batch_size=16,
    epochs=15,
)

model.fit(X_train, y_train)


############################################################
# SMOOTH TRAINING CURVES
############################################################

def smooth_curve(values, window=10):

    values = pd.Series(values)

    return values.rolling(window, min_periods=1).mean()


smooth_loss = smooth_curve(model.train_losses, 10)
smooth_acc = smooth_curve(model.train_accuracies, 10)


############################################################
# EVALUATE MODEL
############################################################

probs = model.predict_proba(X_test)[:, 1]

threshold = 0.6400

preds = (probs >= threshold).astype(int)

print("\nTest Accuracy:", accuracy_score(y_test, preds))

print("\nClassification Report:\n")
print(classification_report(y_test, preds))


############################################################
# COMPUTE METRICS
############################################################

auc = roc_auc_score(y_test, probs)

fpr, tpr, _ = roc_curve(y_test, probs)

cm = confusion_matrix(y_test, preds)

prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)


############################################################
# MULTI PANEL FIGURE
############################################################

plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

################ ROC ################

axes[0,0].plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
axes[0,0].plot([0,1],[0,1],'k--')

axes[0,0].set_title("ROC Curve")
axes[0,0].set_xlabel("False Positive Rate")
axes[0,0].set_ylabel("True Positive Rate")
axes[0,0].legend()

################ CONFUSION MATRIX ################

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0,1]
)

axes[0,1].set_title("Confusion Matrix")
axes[0,1].set_xlabel("Predicted")
axes[0,1].set_ylabel("Actual")

################ CALIBRATION ################

axes[1,0].plot(prob_pred, prob_true, marker="o", label="Model")
axes[1,0].plot([0,1],[0,1],'k--', label="Perfect calibration")

axes[1,0].set_title("Calibration Curve")
axes[1,0].set_xlabel("Mean Predicted Probability")
axes[1,0].set_ylabel("Fraction of Positives")
axes[1,0].legend()

################ TRAINING CURVES ################

ax1 = axes[1,1]

ax1.plot(smooth_loss, color="blue", label="Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="blue")

ax2 = ax1.twinx()

ax2.plot(smooth_acc, color="red", label="Training Accuracy")
ax2.set_ylabel("Accuracy", color="red")

axes[1,1].set_title("Training Curves")

################################################

plt.tight_layout()

plt.savefig("model_evaluation_panel.tiff", dpi=600)

plt.show()



accuracy = accuracy_score(y_test, preds)

precision = precision_score(y_test, preds)
sensitivity = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

mcc = matthews_corrcoef(y_test, preds)

brier = brier_score_loss(y_test, probs)

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

specificity = tn / (tn + fp)

############################################################
# PROBABILITY DISTRIBUTION PLOT
############################################################

plt.figure(figsize=(8,6))

plt.hist(probs[y_test == 0], bins=30, alpha=0.6, label="Negative")
plt.hist(probs[y_test == 1], bins=30, alpha=0.6, label="Positive")

plt.axvline(threshold, color="red", linestyle="--", label="Threshold")

plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Prediction Probability Distribution")

plt.legend()

plt.savefig(
    "ESM_test_probability_distribution.tiff",
    dpi=600,
    format="tiff"
)

plt.show()
plt.close()

############################################################
# SAVE METRICS TO CSV
############################################################

metrics_dict = {

    "Metric":[
        "Accuracy",
        "Specificity",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC",
        "AUC",
        "Brier Score"
    ],

    "Value":[
        accuracy,
        specificity,
        precision,
        sensitivity,
        f1,
        mcc,
        auc,
        brier
    ]
}

metrics_df = pd.DataFrame(metrics_dict)

metrics_df.to_csv("ESM_test_metrics.csv", index=False)

print("\nPerformance Metrics")
print(metrics_df)

print("\nMetrics saved to ESM_test_metrics.csv")

############################################################
# COLORFUL METRICS BAR PLOT
############################################################

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.size":14,
    "axes.labelweight":"bold",
    "axes.titleweight":"bold"
})

colors = sns.color_palette("viridis", len(metrics_df))

plt.figure(figsize=(11,6))

bars = plt.bar(
    metrics_df["Metric"],
    metrics_df["Value"],
    color=colors,
    edgecolor="black",
    linewidth=1.2
)

plt.ylim(0,1.05)

plt.xticks(rotation=40, ha="right")

plt.ylabel("Performance Score", fontsize=16)

plt.title("ESM-MLP Model Performance Metrics", fontsize=18, weight="bold")

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.02,
        f"{height:.3f}",
        ha="center",
        fontsize=12,
        weight="bold"
    )

sns.despine()

plt.tight_layout()

plt.savefig(
    "ESM_test_metrics_colorful_barplot.tiff",
    dpi=600,
    format="tiff"
)

plt.show()

print("Colorful bar plot saved as ESM_test_metrics_colorful_barplot.tiff")
############################################################
# SAVE MODEL
############################################################

checkpoint = {

    "input_dim": model.input_dim,
    "hidden_dim": model.hidden_dim,
    "num_layers": model.num_layers,
    "activation": model.activation,
    "dropout": model.dropout,
    "batch_norm": model.batch_norm,
    "state_dict": model.model.state_dict(),
    "threshold": threshold
}

torch.save(checkpoint, "ABCP_mlp_model.pt")

print("\nModel saved successfully: ABCP_mlp_model.pt")