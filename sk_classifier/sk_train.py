"""
Train Random Forest and SVM classifiers using HSI_Labeled_Dataset
"""

import sys 
from pathlib import Path
from unittest import loader

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hsi_labeled_dataset import HSI_Labeled_Dataset
from core.hsi_trainer import HSI_Trainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



def main(): 


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


    model_configurations = {
        'Random Forest': {
            'model_class': RandomForestClassifier,
            'parameters': {
                'n_estimators': 150,
                'min_samples_leaf': 9,
                'min_samples_split': 2,
                'max_depth': 40
            },
        },
        # 'SVM': {
        #     'model_class': SVC,
        #     'parameters': {
        #         'C': 1.0,
        #         'kernel': 'rbf',
        #     }
        # }
    }

    def create_model(model_name, config):
        model_class = config['model_class']
        parameters = config['parameters']
        return model_class(**parameters)


    # Train baseline RF for comparison
    rf_model = create_model('Random Forest', model_configurations['Random Forest'])
    rf_trainer = HSI_Trainer(model=rf_model, dataset=dataset, model_type='sklearn')
    rf_results = rf_trainer.train_sklearn_classifier(train_ratio=0.7, val_ratio=0.15, verbose=True, sam_weighting=True, alpha=10, use_platt=True)

    print("Training completed.")
    print("Training results:")
    print(f"\nRandom Forest Training Accuracy: {rf_results['train_accuracy']:.4f}")
    print(f"Random Forest Training F1 Score: {rf_results['train_score']:.4f}")


    print("Validation results:")
    print(f"\nRandom Forest Validation Accuracy: {rf_results['val_accuracy']:.4f}")
    print(f"Random Forest Validation F1 Score: {rf_results['val_score']:.4f}")

    
    # Save model
    model_name = 'rf_best_model_cos_sim'
    rf_trainer.save(f'rf/models/{model_name}.joblib')

    # Train baseline SVM for comparison
    # svm_model = create_model('SVM', model_configurations['SVM'])
    # svm_trainer = HSI_Trainer(svm_model, dataset=dataset, model_type='sklearn')
    # svm_results = svm_trainer.train_sklearn_classifier( train_ratio=0.7, val_ratio=0.15, verbose=True)


    # Evaluate on test set and get confusion matrix

    test_results = rf_trainer.evaluate(batch_size=32)
    print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(test_results['classification_report'])

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

if __name__ == "__main__":
    main()