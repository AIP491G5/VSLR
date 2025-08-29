"""
Training utilities for VSLR project
Contains training functions and optimization utilities
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_model(model, train_loader, val_loader, config, device):
    """Train the HGC-LSTM model"""
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    if config.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay
        )
    else:  # SGD
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay,
            momentum=config.training.momentum
        )
    
    # Setup scheduler
    if config.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.training.scheduler_step_size, 
            gamma=config.training.scheduler_gamma
        )
    elif config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.training.num_epochs
        )
    else:
        scheduler = None
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_train_acc = 0.0 
    patience_counter = 0
    
    # Create save directory
    os.makedirs(config.training.save_dir, exist_ok=True)
    
    for epoch in range(config.training.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (keypoints, labels) in enumerate(train_loader):
            keypoints, labels = keypoints.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            if config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for keypoints, labels in val_loader:
                keypoints, labels = keypoints.to(device), labels.to(device)
                outputs = model(keypoints)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Save best model with improved logic
        save_model = False
        
        if val_acc > best_val_acc:
            # Val accuracy improved
            save_model = True
            best_val_acc = val_acc
            best_train_acc = train_acc
            patience_counter = 0
        elif val_acc == best_val_acc and train_acc > best_train_acc:
            save_model = True
            best_train_acc = train_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if save_model:
            model_path = os.path.join(config.training.save_dir, config.training.model_save_name)
            torch.save(model, model_path)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{config.training.num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:05.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:05.2f}% | LR: {current_lr:.8f}")
        
        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}% (Train: {best_train_acc:.2f}%)")
    return history

def train_model_triplet(model, train_loader, val_loader, config, device):
    """Train model using Triplet Loss"""
    criterion = nn.TripletMarginLoss(margin=config.training.triplet_margin)
    
    # Optimizer
    if config.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay,
            momentum=config.training.momentum
        )
    
    # Scheduler
    if config.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.training.scheduler_step_size, 
            gamma=config.training.scheduler_gamma
        )
    elif config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.training.num_epochs
        )
    else:
        scheduler = None

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0

    os.makedirs(config.training.save_dir, exist_ok=True)

    for epoch in range(config.training.num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a, emb_p, emb_n = model(anchor, positive, negative)
            emb_a = F.normalize(emb_a, p=2, dim=1)
            emb_p = F.normalize(emb_p, p=2, dim=1)
            emb_n = F.normalize(emb_n, p=2, dim=1)
            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                emb_a, emb_p, emb_n = model(anchor, positive, negative)
                emb_a = F.normalize(emb_a, p=2, dim=1)
                emb_p = F.normalize(emb_p, p=2, dim=1)
                emb_n = F.normalize(emb_n, p=2, dim=1)
                loss = criterion(emb_a, emb_p, emb_n)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Save best model
        save_model = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model = True
        else:
            patience_counter += 1

        if save_model:
            model_path = os.path.join(config.training.save_dir, config.training.model_triplet_save_name)
            torch.save(model, model_path)

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Update LR
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{config.training.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.8f}")

        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return history


