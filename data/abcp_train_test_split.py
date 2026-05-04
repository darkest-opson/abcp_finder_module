from Bio import SeqIO
import random
import os

# -----------------------------
# Step 1: Parse CD-HIT clusters
# -----------------------------
def parse_cdhit_clusters(clstr_file):
    clusters = {}
    cluster_id = None
    
    with open(clstr_file, 'r') as f:
        for line in f:
            if line.startswith(">Cluster"):
                cluster_id = line.strip()
                clusters[cluster_id] = []
            else:
                seq_id = line.split(">")[1].split("...")[0]
                clusters[cluster_id].append(seq_id)
    
    return clusters


# -----------------------------
# Step 2: Assign clusters to folds
# -----------------------------
def assign_clusters_to_folds(clusters, n_folds=5):
    cluster_list = list(clusters.keys())
    random.shuffle(cluster_list)
    
    folds = {i: [] for i in range(n_folds)}
    
    for i, cluster in enumerate(cluster_list):
        fold_id = i % n_folds
        folds[fold_id].extend(clusters[cluster])
    
    return folds


# -----------------------------
# Step 3: Write train/test FASTA
# -----------------------------
def write_split_fasta(input_fasta, folds, test_fold=0):
    records = {rec.id: rec for rec in SeqIO.parse(input_fasta, "fasta")}
    
    train_ids = []
    test_ids = folds[test_fold]
    
    for fold_id, ids in folds.items():
        if fold_id != test_fold:
            train_ids.extend(ids)
    
    train_records = [records[i] for i in train_ids if i in records]
    test_records = [records[i] for i in test_ids if i in records]
    
    SeqIO.write(train_records, "ABCP_train.fasta", "fasta")
    SeqIO.write(test_records, "ABCP_test.fasta", "fasta")
    
    print(f"Train sequences: {len(train_records)}")
    print(f"Test sequences: {len(test_records)}")


# -----------------------------
# Main
# -----------------------------
clstr_file = "ABCP_clustered.fasta.clstr"
input_fasta = "ABCP_clustered.fasta"

clusters = parse_cdhit_clusters(clstr_file)
folds = assign_clusters_to_folds(clusters, n_folds=5)
write_split_fasta(input_fasta, folds, test_fold=0)
