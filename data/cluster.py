import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import random
from scipy.stats import gaussian_kde
import seaborn as sns

# -----------------------------
# STYLE (IMPORTANT)
# -----------------------------
sns.set(style="whitegrid", context="paper", font_scale=1.5)

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_FILES = ["ABCP_train.fasta", "Non_ABCP_train.fasta"]
TEST_FILES = ["ABCP_test.fasta", "Non_ABCP_test.fasta"]

MAX_SAMPLES = 150


# -----------------------------
# LOAD FASTA
# -----------------------------
def load_sequences(files):
    seqs = []
    for f in files:
        for record in SeqIO.parse(f, "fasta"):
            seqs.append(str(record.seq))
    return seqs

train_seqs = load_sequences(TRAIN_FILES)
test_seqs = load_sequences(TEST_FILES)

# -----------------------------
# SUBSAMPLING
# -----------------------------
random.seed(42)
train_sample = random.sample(train_seqs, min(MAX_SAMPLES, len(train_seqs)))
test_sample = random.sample(test_seqs, min(MAX_SAMPLES, len(test_seqs)))

# -----------------------------
# IDENTITY FUNCTION
# -----------------------------
def seq_identity(s1, s2):
    min_len = min(len(s1), len(s2))
    matches = sum(1 for i in range(min_len) if s1[i] == s2[i])
    return (matches / min_len) * 100

# -----------------------------
# COMPUTE MATRIX
# -----------------------------
similarity_matrix = np.zeros((len(train_sample), len(test_sample)))

for i, t in enumerate(train_sample):
    for j, s in enumerate(test_sample):
        similarity_matrix[i, j] = seq_identity(t, s)

similarities = similarity_matrix.flatten()

plt.figure(figsize=(8,6))

# Histogram
sns.histplot(similarities[similarities <= 30-2], bins=50, stat="density",
             color="#4C72B0", edgecolor=None, alpha=0.6)

# KDE
sns.kdeplot(similarities[similarities <= 30-2], color="#DD8452", linewidth=2.5)

# Threshold line (right boundary now)
plt.axvline(30, color="#C44E52",
            linestyle="--", linewidth=2,
            label=f"Identity Threshold")

# Labels
plt.xlabel("Sequence Identity (%)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.title("Train–Test Sequence Identity Distribution (≤30%)", fontsize=16)

plt.legend(frameon=False)
sns.despine()
plt.tight_layout()

plt.savefig("identity_distribution_filtered.png", dpi=600, bbox_inches='tight')
plt.show()

