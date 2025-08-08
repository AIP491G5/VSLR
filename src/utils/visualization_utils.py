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

def visualize_training_triplet_process(history, config, save_plots=True):
    """
    Visualize the training process for the triplet model
    """
    # Create plots directory
    plots_dir = Path("training_plots")
    plots_dir.mkdir(exist_ok=True)

    # Extract data
    epochs = range(1, len(history['train_loss']) + 1)
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    # train_accs = history['train_acc']
    # val_accs = history['val_acc']

    # Create simple figure with 1 plots
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # 1. Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy curves
    # ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    # ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    # ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Accuracy (%)')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)

    plt.suptitle('HGC-LSTM Training Process', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    if save_plots:
        plot_path = plots_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Training curves saved to: {plot_path}")

    plt.show()

    # Print summary

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

def visualize_dual_attention_weights(model, data_loader, device, config, num_samples=3, save_plots=True):
    """
    Visualize both joint attention and temporal attention weights
    Args:
        model: trained model with dual attention
        data_loader: data loader for samples
        device: computation device
        config: configuration object
        num_samples: number of samples to visualize
        save_plots: whether to save plots to disk
    """
    # Create plots directory
    plots_dir = Path("attention_plots")
    plots_dir.mkdir(exist_ok=True)
    
    model.eval()
    
    # Hook functions to capture attention weights
    joint_attention_weights = []
    temporal_attention_weights = []
    
    def hook_joint_attention(module, input, output):
        # output is (pooled_output, attention_weights)
        joint_attention_weights.append(output[1].detach())
    
    def hook_temporal_attention(module, input, output):
        # output is (pooled_output, attention_weights)
        temporal_attention_weights.append(output[1].detach())
    
    # Register hooks
    joint_hook = model.joint_attention.register_forward_hook(hook_joint_attention)
    temporal_hook = model.temporal_attention.register_forward_hook(hook_temporal_attention)
    
    try:
        with torch.no_grad():
            sample_count = 0
            for batch_idx, (data, labels) in enumerate(data_loader):
                if sample_count >= num_samples:
                    break
                
                data = data.to(device)
                labels = labels.to(device)
                
                # Clear previous attention weights
                joint_attention_weights.clear()
                temporal_attention_weights.clear()
                
                # Forward pass
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                
                # Process each sample in the batch
                batch_size = min(data.shape[0], num_samples - sample_count)
                
                for i in range(batch_size):
                    # Create subplot for this sample
                    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # 1. Joint Attention Weights (B, T, V, 1) -> average over time
                    joint_att = joint_attention_weights[0][i].squeeze(-1)  # (T, V)
                    joint_att_avg = joint_att.mean(dim=0).cpu().numpy()  # (V,) - average over time
                    
                    # Plot joint attention as bar chart
                    axes[0].bar(range(len(joint_att_avg)), joint_att_avg, color='skyblue', alpha=0.7)
                    axes[0].set_title(f'Joint Attention Weights\nSample {sample_count + 1} - Pred: {predictions[i].item()}, True: {labels[i].item()}', 
                                    fontsize=12, fontweight='bold')
                    axes[0].set_xlabel('Keypoint Index')
                    axes[0].set_ylabel('Attention Weight')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Add keypoint group labels
                    axes[0].axvline(x=33, color='red', linestyle='--', alpha=0.5, label='Pose|Hand boundary')
                    axes[0].axvline(x=54, color='red', linestyle='--', alpha=0.5, label='Left|Right hand boundary')
                    axes[0].legend()
                    
                    # 2. Temporal Attention Weights (B, T, 1) -> plot over time
                    temporal_att = temporal_attention_weights[0][i].squeeze(-1).cpu().numpy()  # (T,)
                    
                    axes[1].plot(range(len(temporal_att)), temporal_att, marker='o', linewidth=2, markersize=4, color='orange')
                    axes[1].set_title(f'Temporal Attention Weights\nSample {sample_count + 1} - Pred: {predictions[i].item()}, True: {labels[i].item()}', 
                                    fontsize=12, fontweight='bold')
                    axes[1].set_xlabel('Time Step (Frame)')
                    axes[1].set_ylabel('Attention Weight')
                    axes[1].grid(True, alpha=0.3)
                    axes[1].fill_between(range(len(temporal_att)), temporal_att, alpha=0.3, color='orange')
                    
                    plt.tight_layout()
                    
                    # Save plot
                    if save_plots:
                        plot_path = plots_dir / f"dual_attention_sample_{sample_count + 1}.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                        print(f"Dual attention weights for sample {sample_count + 1} saved to: {plot_path}")
                    
                    plt.show()
                    
                    sample_count += 1
                    if sample_count >= num_samples:
                        break
    
    finally:
        # Remove hooks
        joint_hook.remove()
        temporal_hook.remove()

def visualize_attention_heatmap(model, data_loader, device, config, num_samples=2, save_plots=True):
    """
    Visualize joint attention as heatmap across time and keypoints
    Args:
        model: trained model with dual attention
        data_loader: data loader for samples
        device: computation device
        config: configuration object
        num_samples: number of samples to visualize
        save_plots: whether to save plots to disk
    """
    # Create plots directory
    plots_dir = Path("attention_plots")
    plots_dir.mkdir(exist_ok=True)
    
    model.eval()
    
    # Hook function to capture joint attention weights
    joint_attention_weights = []
    
    def hook_joint_attention(module, input, output):
        joint_attention_weights.append(output[1].detach())
    
    # Register hook
    joint_hook = model.joint_attention.register_forward_hook(hook_joint_attention)
    
    try:
        with torch.no_grad():
            sample_count = 0
            
            for batch_idx, (data, labels) in enumerate(data_loader):
                if sample_count >= num_samples:
                    break
                
                data = data.to(device)
                labels = labels.to(device)
                
                # Clear previous attention weights
                joint_attention_weights.clear()
                
                # Forward pass
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                
                # Process samples
                batch_size = min(data.shape[0], num_samples - sample_count)
                
                for i in range(batch_size):
                    # Joint attention heatmap (T, V)
                    joint_att = joint_attention_weights[0][i].squeeze(-1).cpu().numpy()  # (T, V)
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    sns.heatmap(joint_att.T, annot=False, cmap='YlOrRd', ax=ax, cbar_kws={'shrink': 0.8})
                    
                    ax.set_title(f'Joint Attention Heatmap - Sample {sample_count + 1}\n'
                               f'Predicted: {predictions[i].item()}, True: {labels[i].item()}', 
                               fontsize=14, fontweight='bold')
                    ax.set_xlabel('Time Steps (Frames)')
                    ax.set_ylabel('Keypoint Index')
                    
                    # Add keypoint group boundaries
                    ax.axhline(y=33, color='blue', linestyle='--', alpha=0.7, linewidth=2)
                    ax.axhline(y=54, color='blue', linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Add text labels for keypoint groups
                    ax.text(joint_att.shape[0] * 0.02, 16, 'Pose', fontsize=10, fontweight='bold', color='blue')
                    ax.text(joint_att.shape[0] * 0.02, 43, 'Left Hand', fontsize=10, fontweight='bold', color='blue')
                    ax.text(joint_att.shape[0] * 0.02, 64, 'Right Hand', fontsize=10, fontweight='bold', color='blue')
                    
                    plt.tight_layout()
                    
                    # Save plot
                    if save_plots:
                        plot_path = plots_dir / f"joint_attention_heatmap_sample_{sample_count + 1}.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                        print(f"Joint attention heatmap for sample {sample_count + 1} saved to: {plot_path}")
                    
                    plt.show()
                    
                    sample_count += 1
                    if sample_count >= num_samples:
                        break
    
    finally:
        # Remove hook
        joint_hook.remove()
