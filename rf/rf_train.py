"""
Train Random Forest and SVM classifiers using HSI_Labeled_Dataset
"""

import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hsi_labeled_dataset import HSI_Labeled_Dataset
from core.hsi_trainer import HSI_Trainer
from sklearn.ensemble import RandomForestClassifier


# Load dataset from saved files
print("Loading dataset...")

dataset = HSI_Labeled_Dataset(
    molecule_dataset_path='molecule_dataset/lipid_subtype_wn_61_test',
    srs_params_path='params_dataset/srs_params_61',
    num_samples_per_class=10000,
    normalize_per_molecule=False,
    compute_min_max=True,
    noise_multiplier=0.5
)

# Visualize the dataset (optional)
# print("\n" + "=" * 80)
# print("VISUALIZING DATASET SAMPLES")
# print("=" * 80)
# dataset.visualize_dataset_samples(train_ratio=0.7, val_ratio=0.15, num_samples_per_class=3)

# Training parameters
num_estimators = 200
max_depth = 20
min_samples_split = 2
min_samples_leaf = 2



# Train baseline RF for comparison
rf_model = RandomForestClassifier(
    n_estimators=num_estimators,
    min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,
    max_depth=max_depth,
    random_state=42,
    n_jobs=-1
)

rf_trainer = HSI_Trainer(rf_model, model_type='sklearn')
rf_results = rf_trainer.train_sklearn(dataset, train_ratio=0.7, val_ratio=0.15, verbose=True)

# Evaluate on test set and get confusion matrix
print("\n" + "=" * 80)
print("EVALUATING ON TEST SET")
print("=" * 80)
test_results = rf_trainer.evaluate(dataset, batch_size=32)
print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")
print("\nClassification Report:")
print(test_results['classification_report'])

# Plot and save confusion matrix
print("\n" + "=" * 80)
print("GENERATING CONFUSION MATRIX")
print("=" * 80)

# Get molecule names from dataset
molecule_names = dataset.molecule_names

# Create both normalized and unnormalized confusion matrices
fig1, ax1 = rf_trainer.plot_confusion_matrix(
    test_results, 
    class_names=molecule_names,
    normalize=False,
    save_path='plots/rf_confusion_matrix.png',
    title='Random Forest Confusion Matrix',
    figsize=(len(dataset.molecule_names)*2, len(dataset.molecule_names)*2) #adjust size of the plots based on number of classes
)

fig2, ax2 = rf_trainer.plot_confusion_matrix(
    test_results,
    class_names=molecule_names,
    normalize=True,
    save_path='plots/rf_confusion_matrix_normalized.png',
    title='Random Forest Confusion Matrix (Normalized)',
    figsize=(len(dataset.molecule_names)*2, len(dataset.molecule_names)*2)
)

print("\nConfusion matrices saved to:")
print("  - plots/rf_confusion_matrix.png")
print("  - plots/rf_confusion_matrix_normalized.png")

# Save model
model_name = 'best_model'
rf_trainer.save(f'rf/models/{model_name}.joblib')

# Display output
print("=" * 80)
print("\nSaved models:")
print(f"  - rf/models/{model_name}.joblib")

