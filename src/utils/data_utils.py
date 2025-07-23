"""
Data utilities for VSLR project
Contains dataset classes, data augmentation functions, and data loading utilities
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def load_labels_from_csv(csv_file, config):
    """Load labels from CSV file"""
    if csv_file is None:
        csv_file = config.data.input_csv_file
    
    df = pd.read_csv(csv_file)
    video_to_label_mapping = {}
    id_to_label_mapping = {}

    for _, row in df.iterrows():
        label_id = row['id']
        label_name = row['label']
        videos_str = row['videos']
        video_files = [v.strip() for v in videos_str.split(',')]

        id_to_label_mapping[label_id] = label_name
        
        for video_file in video_files:
            video_base = os.path.splitext(video_file)[0]
            video_to_label_mapping[video_base] = label_id
    
    unique_labels = sorted(df['id'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f" {len(unique_labels)} classes: {unique_labels}")
    print(f" {len(video_to_label_mapping)} videos")
    
    return video_to_label_mapping, label_to_idx, unique_labels, id_to_label_mapping


def flip_keypoints(keypoints):
    """
    Flip keypoints horizontally for data augmentation
    Args:
        keypoints: numpy array of shape (T, V, C) where V=75 (33 pose + 21 left hand + 21 right hand)
    Returns:
        flipped_keypoints: horizontally flipped keypoints
    """
    flipped_keypoints = keypoints.copy()
    
    # Flip x coordinates (assuming x is the first channel)
    flipped_keypoints[:, :, 0] = 1.0 - flipped_keypoints[:, :, 0]
    
    # Define swap mappings for MediaPipe landmarks
    # Pose landmarks that need to be swapped (left <-> right)
    pose_swap_pairs = [
        (1, 4),   # left_eye_inner <-> right_eye_inner
        (2, 5),   # left_eye <-> right_eye  
        (3, 6),   # left_eye_outer <-> right_eye_outer
        (7, 8),   # left_ear <-> right_ear
        (11, 12), # left_shoulder <-> right_shoulder
        (13, 14), # left_elbow <-> right_elbow
        (15, 16), # left_wrist <-> right_wrist
        (17, 18), # left_pinky <-> right_pinky
        (19, 20), # left_index <-> right_index
        (21, 22), # left_thumb <-> right_thumb
        (23, 24), # left_hip <-> right_hip
        (25, 26), # left_knee <-> right_knee
        (27, 28), # left_ankle <-> right_ankle
        (29, 30), # left_heel <-> right_heel
        (31, 32)  # left_foot_index <-> right_foot_index
    ]
    
    # Swap pose landmarks
    for left_idx, right_idx in pose_swap_pairs:
        flipped_keypoints[:, left_idx, :], flipped_keypoints[:, right_idx, :] = \
            flipped_keypoints[:, right_idx, :].copy(), flipped_keypoints[:, left_idx, :].copy()
    
    # Swap left and right hand landmarks
    # Left hand (indices 33-53) <-> Right hand (indices 54-74)
    left_hand_start, left_hand_end = 33, 54
    right_hand_start, right_hand_end = 54, 75
    
    temp_left_hand = flipped_keypoints[:, left_hand_start:left_hand_end, :].copy()
    flipped_keypoints[:, left_hand_start:left_hand_end, :] = flipped_keypoints[:, right_hand_start:right_hand_end, :]
    flipped_keypoints[:, right_hand_start:right_hand_end, :] = temp_left_hand
    
    return flipped_keypoints


def translate_keypoints(keypoints, translation_range=0.1):
    """
    Apply translation augmentation to keypoints
    Args:
        keypoints: numpy array of shape (T, V, C) where V=75, C=2 (x, y)
        translation_range: range for random translation (-range to +range)
    Returns:
        translated_keypoints: translated keypoints
    """
    translated_keypoints = keypoints.copy()
    
    # Random translation for x and y coordinates
    # Generate random translation values for the entire sequence
    tx = np.random.uniform(-translation_range, translation_range)
    ty = np.random.uniform(-translation_range, translation_range)
    
    # Apply translation
    translated_keypoints[:, :, 0] += tx  # x coordinates
    translated_keypoints[:, :, 1] += ty  # y coordinates
    
    # Clamp values to [0, 1] range (assuming normalized coordinates)
    translated_keypoints = np.clip(translated_keypoints, 0.0, 1.0)
    
    return translated_keypoints


def scale_keypoints(keypoints, scale_range=0.1):
    """
    Apply scaling augmentation to keypoints
    Args:
        keypoints: numpy array of shape (T, V, C) where V=75, C=2 (x, y)
        scale_range: range for random scaling (1-range to 1+range)
    Returns:
        scaled_keypoints: scaled keypoints
    """
    scaled_keypoints = keypoints.copy()
    
    # Random scaling
    # Generate random scale factors
    scale_x = np.random.uniform(1 - scale_range, 1 + scale_range)
    scale_y = np.random.uniform(1 - scale_range, 1 + scale_range)
    
    # Find the center of keypoints for scaling
    center_x = np.mean(scaled_keypoints[:, :, 0])
    center_y = np.mean(scaled_keypoints[:, :, 1])
    
    # Apply scaling around center
    scaled_keypoints[:, :, 0] = center_x + (scaled_keypoints[:, :, 0] - center_x) * scale_x
    scaled_keypoints[:, :, 1] = center_y + (scaled_keypoints[:, :, 1] - center_y) * scale_y
    
    # Clamp values to [0, 1] range (assuming normalized coordinates)
    scaled_keypoints = np.clip(scaled_keypoints, 0.0, 1.0)
    
    return scaled_keypoints


def transform_keypoints(keypoints, translation_range=0.1, scale_range=0.1):
    """
    Apply both translation and scaling augmentation to keypoints (legacy function)
    Args:
        keypoints: numpy array of shape (T, V, C) where V=75, C=2 (x, y)
        translation_range: range for random translation (-range to +range)
        scale_range: range for random scaling (1-range to 1+range)
    Returns:
        transformed_keypoints: augmented keypoints
    """
    # Apply translation first, then scaling
    transformed_keypoints = translate_keypoints(keypoints, translation_range)
    transformed_keypoints = scale_keypoints(transformed_keypoints, scale_range)
    
    return transformed_keypoints


class SignLanguageDataset(Dataset):
    """Dataset for Sign Language Recognition with stratified split and multiple data augmentation options"""
    
    def __init__(self, keypoints_dir, video_to_label_mapping, label_to_idx, config, 
                 sequence_length=None, split_type='train', train_split=None, 
                 augmentations=None, use_strategy=True):
        
        self.sequence_length = sequence_length or config.hgc_lstm.sequence_length
        train_split = train_split or config.training.train_split
        self.use_strategy = use_strategy
        self.config = config
        
        # Handle augmentations array
        if augmentations is not None:
            # Use provided augmentations array
            self.augmentations = augmentations
        else:
            # Use augmentations array from config (default: ['flip', 'translation', 'scaling'])
            self.augmentations = getattr(config.data, 'augmentations', [])
        
        # Set individual flags for backward compatibility (if needed elsewhere)
        self.use_flip_augmentation = 'flip' in self.augmentations
        self.use_translation_augmentation = 'translation' in self.augmentations
        self.use_scaling_augmentation = 'scaling' in self.augmentations
        self.use_combined_augmentation = len(self.augmentations) > 1
        
        # Get augmentation parameters from config
        self.translation_range = config.data.translation_range
        self.scale_range = config.data.scale_range
        
        self.keypoints_dir = keypoints_dir
        self.video_to_label_mapping = video_to_label_mapping
        self.label_to_idx = label_to_idx
        
        # Get all available files
        available_files = [f for f in os.listdir(keypoints_dir) if f.endswith(config.data.keypoints_ext)]
        valid_files = []
        for file in available_files:
            base_name = os.path.splitext(file)[0]
            if base_name in video_to_label_mapping:
                valid_files.append(base_name)
        
        if self.use_strategy:
            # Stratified split - ensure each class has balanced samples in train/val
            print(f" Using stratified split strategy")
            
            # Group files by label_id for stratified split
            label_to_files = {}
            for file in valid_files:
                label_id = video_to_label_mapping[file]
                if label_id not in label_to_files:
                    label_to_files[label_id] = []
                label_to_files[label_id].append(file)
            
            # Perform stratified split
            train_files = []
            val_files = []
            
            for label_id, files in label_to_files.items():
                # Shuffle files for this class
                np.random.shuffle(files)
                
                # Calculate split for this class
                n_files = len(files)
                n_train = int(n_files * train_split)
                n_val = n_files - n_train
                
                # Split files for this class
                class_train = files[:n_train]
                class_val = files[n_train:]
                
                train_files.extend(class_train)
                val_files.extend(class_val)
            
            # Shuffle the final lists
            np.random.shuffle(train_files)
            np.random.shuffle(val_files)
        else:
            # Simple random split - no stratification
            print(f"Using simple random split")
            
            # Shuffle all files
            np.random.shuffle(valid_files)
            
            # Calculate split point
            n_total = len(valid_files)
            n_train = int(n_total * train_split)
            
            # Split files
            train_files = valid_files[:n_train]
            val_files = valid_files[n_train:]
        
        # Select files based on split type
        if split_type == 'train':
            self.files = train_files
        else:
            self.files = val_files
        
        # Calculate total samples with augmentation
        self.original_samples = len(self.files)
        
        # Calculate all possible augmentation combinations
        augmentation_combinations = self._get_augmentation_combinations()
        augmentation_multiplier = len(augmentation_combinations)
        
        # Store augmentation combinations for __getitem__
        self.augmentation_combinations = augmentation_combinations
        
        # Create augmented file indices
        self.augmented_samples = self.original_samples * augmentation_multiplier
        
        print(f" {split_type.upper()} dataset: {self.original_samples} original files")
        if self.augmentations:
            print(f" Augmentation enabled: {self.augmented_samples} total samples (x{augmentation_multiplier})")
            print(f"  - Augmentations: {self.augmentations}")
            print(f"  - Total combinations: {len(self.augmentation_combinations)}")
            print(f"  - Combinations: {['+'.join(combo) if combo else 'original' for combo in self.augmentation_combinations]}")
        else:
            print(f" No augmentation enabled: {self.augmented_samples} total samples")
        
        # Print class distribution for verification
        self._print_class_distribution(split_type)

    def _get_augmentation_combinations(self):
        """Generate all possible augmentation combinations"""
        from itertools import combinations, chain
        
        combinations_list = []
        
        # Always include original (no augmentation)
        combinations_list.append([])
        
        # Single augmentations
        for aug in self.augmentations:
            combinations_list.append([aug])
        
        # Multiple augmentations (combinations of 2, 3, etc.)
        for r in range(2, len(self.augmentations) + 1):
            for combo in combinations(self.augmentations, r):
                combinations_list.append(list(combo))
        
        return combinations_list

    def _print_class_distribution(self, split_type):
        """Print class distribution to verify balanced split"""
        class_counts = {}
        for file in self.files:
            label_id = self.video_to_label_mapping[file]
            class_counts[label_id] = class_counts.get(label_id, 0) + 1
        
        print(f" {split_type.upper()} class distribution:")
        for label_id, count in sorted(class_counts.items()):
            print(f"  Class {label_id}: {count} samples")

        counts = list(class_counts.values())
        if len(set(counts)) != 1:
            min_count, max_count = min(counts), max(counts)
            print(f" ⚠ Imbalanced: {min_count}-{max_count} samples per class")
        else:
            print(f" ✓ Balanced: {counts[0]} samples per class")

    def __len__(self):
        return self.augmented_samples

    def __getitem__(self, idx):
        # Determine augmentation type based on index
        file_idx = idx % self.original_samples
        augmentation_idx = idx // self.original_samples
        
        base_filename = self.files[file_idx]
        
        kp_path = os.path.join(self.keypoints_dir, f"{base_filename}{self.config.data.keypoints_ext}")
        kp_sequence = np.load(kp_path)
        
        if len(kp_sequence.shape) == 2:
            T, features = kp_sequence.shape
            expected_features = self.config.hgc_lstm.num_vertices * self.config.hgc_lstm.in_channels
            if features == expected_features:
                kp_sequence = kp_sequence.reshape(T, self.config.hgc_lstm.num_vertices, self.config.hgc_lstm.in_channels)
        
        # Apply augmentations based on combination
        augmentation_combo = self.augmentation_combinations[augmentation_idx]
        
        for aug_type in augmentation_combo:
            if aug_type == 'flip':
                kp_sequence = flip_keypoints(kp_sequence)
            elif aug_type == 'translation':
                kp_sequence = translate_keypoints(kp_sequence, self.translation_range)
            elif aug_type == 'scaling':
                kp_sequence = scale_keypoints(kp_sequence, self.scale_range)
        
        label_id = self.video_to_label_mapping[base_filename]
        label_idx = self.label_to_idx[label_id]
        
        return torch.from_numpy(kp_sequence).float(), torch.tensor(label_idx, dtype=torch.long)


def create_data_loaders(train_dataset, val_dataset, config, batch_size=None):
    """Create data loaders"""
    batch_size = batch_size or config.training.batch_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f" Train batches: {len(train_loader)}")
    print(f" Valid batches: {len(val_loader)}")
    print(f" Batch size: {batch_size}")
    
    # Show detailed augmentation info
    if hasattr(train_dataset, 'augmentations') and train_dataset.augmentations:
        original_samples = train_dataset.original_samples
        total_samples = len(train_dataset)
        
        aug_str = " + ".join(train_dataset.augmentations)
        print(f" Data Augmentation ({aug_str}): {original_samples} original → {total_samples} total samples")
        print(f" Augmentation combinations: {['+'.join(combo) if combo else 'original' for combo in train_dataset.augmentation_combinations]}")
    else:
        print(f" No augmentation: {len(train_dataset)} samples")
    
    sample_kp, sample_lbl = next(iter(train_loader))
    print(f" Sample keypoints shape: {sample_kp.shape}")  # (B, T, V, C)
    print(f" Sample labels shape: {sample_lbl.shape}")    # (B,)
    return train_loader, val_loader