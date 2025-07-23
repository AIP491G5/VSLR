"""
Visualization utilities for VSLR project
Contains plotting functions for training analysis and model performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from pathlib import Path


def visualize_training_process(history, config, save_plots=True):
    """
    Simple visualization of training process - only essential plots
    Args:
        history: training history dict with train_loss, val_loss, train_acc, val_acc
        config: configuration object
        save_plots: whether to save plots to disk
    """
    # Create plots directory
    plots_dir = Path("training_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Extract data
    epochs = range(1, len(history['train_loss']) + 1)
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    train_accs = history['train_acc']
    val_accs = history['val_acc']
    
    # Create simple figure with 2 plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('HGC-LSTM Training Process', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    if save_plots:
        plot_path = plots_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Training curves saved to: {plot_path}")
    
    plt.show()
    
    # Print summary
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]
    best_val_acc = max(val_accs)
    best_epoch = val_accs.index(best_val_acc) + 1
    
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"   Final Training Accuracy: {final_train_acc:.2f}%")
    
    return fig


def analyze_model_performance(model, val_loader, device, config, unique_labels, id_to_label_mapping):
    """
    Simple model performance analysis - only confusion matrix
    Args:
        model: trained model
        val_loader: validation data loader
        device: computation device
        config: configuration object
        unique_labels: list of unique label IDs
        id_to_label_mapping: mapping from ID to label name
    """
    print(" Analyzing model performance on validation set...")
    
    # Create plots directory
    plots_dir = Path("training_plots")
    plots_dir.mkdir(exist_ok=True)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for keypoints, labels in val_loader:
            keypoints, labels = keypoints.to(device), labels.to(device)
            outputs = model(keypoints)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # Create confusion matrix plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Create label names for display
    label_names = []
    for label_idx in range(len(unique_labels)):
        label_id = unique_labels[label_idx]
        label_name = id_to_label_mapping.get(label_id, f"{label_id}")
        label_names.append(f"{label_id}")  # Keep it simple for readability
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                square=True, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    
    plt.tight_layout()
    
    # Save plot
    confusion_path = plots_dir / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Confusion matrix saved to: {confusion_path}")
    
    plt.show()
    
    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    # Create class name mapping for report
    target_names = [f"Class_{uid}" for uid in unique_labels]
    print(classification_report(all_labels, all_predictions, target_names=target_names))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels
    }
