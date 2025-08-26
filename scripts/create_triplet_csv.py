import pandas as pd
import random
import numpy as np
from collections import defaultdict

def generate_triplet_dataset(input_csv: str, output_csv: str = "triplet_dataset.csv", 
                           max_triplets_per_class: int = None, random_seed: int = 42):
    """
    Tạo triplet dataset từ file labels.csv với format:
    - Mỗi dòng gồm: anchor, positive (cùng gloss), negative (khác gloss)
    - Đảm bảo positive != negative
    - Đảm bảo negative từ class khác hoàn toàn
    - Cân bằng số lượng triplet cho mỗi class
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    df = pd.read_csv(input_csv)
    triplets = []
    
    # Tạo danh sách tất cả video theo class
    class_videos = defaultdict(list)
    for _, row in df.iterrows():
        label = row['label']
        videos = row['videos'].split(', ')
        class_videos[label].extend([(row['id'], video.strip()) for video in videos])
    
    # Lọc ra classes có ít nhất 2 video
    valid_classes = {label: videos for label, videos in class_videos.items() if len(videos) >= 2}
    
    print(f"📊 Found {len(valid_classes)} valid classes with >=2 videos")
    for label, videos in valid_classes.items():
        print(f"   Class {label}: {len(videos)} videos")
    
    # Tạo triplets
    triplet_counts = defaultdict(int)
    
    for current_label, current_videos in valid_classes.items():
        class_triplets = []
        
        # Tạo tất cả cặp (anchor, positive) trong class hiện tại
        for i in range(len(current_videos)):
            for j in range(i + 1, len(current_videos)):
                anchor_id, anchor_video = current_videos[i]
                positive_id, positive_video = current_videos[j]
                
                # Đảm bảo anchor != positive
                if anchor_video == positive_video:
                    continue
                
                # Chọn negative từ classes khác
                other_classes = [label for label in valid_classes.keys() if label != current_label]
                if not other_classes:
                    continue
                
                # Chọn random class khác
                negative_class = random.choice(other_classes)
                negative_id, negative_video = random.choice(valid_classes[negative_class])
                
                # Đảm bảo negative khác positive và anchor
                max_attempts = 10
                attempts = 0
                while (negative_video == positive_video or negative_video == anchor_video) and attempts < max_attempts:
                    negative_class = random.choice(other_classes)
                    negative_id, negative_video = random.choice(valid_classes[negative_class])
                    attempts += 1
                
                # Skip nếu không tìm được negative hợp lệ
                if negative_video == positive_video or negative_video == anchor_video:
                    continue
                
                class_triplets.append((anchor_id, anchor_video, positive_video, negative_video))
        
        # Giới hạn số lượng triplets per class nếu cần
        if max_triplets_per_class and len(class_triplets) > max_triplets_per_class:
            class_triplets = random.sample(class_triplets, max_triplets_per_class)
        
        triplets.extend(class_triplets)
        triplet_counts[current_label] = len(class_triplets)
        print(f"   Generated {len(class_triplets)} triplets for class {current_label}")
    
    # Shuffle để tránh bias
    random.shuffle(triplets)
    
    # Validation: Kiểm tra quality
    print(f"\n🔍 VALIDATION:")
    print(f"   Total triplets: {len(triplets)}")
    
    # Kiểm tra positive == negative
    invalid_count = 0
    same_class_negative = 0
    
    for id, anchor, positive, negative in triplets:
        if positive == negative:
            invalid_count += 1
        
        # Kiểm tra negative có cùng class với anchor không
        anchor_class = anchor.split('_')[0] if '_' in anchor else anchor[:2]
        negative_class = negative.split('_')[0] if '_' in negative else negative[:2]
        if anchor_class == negative_class:
            same_class_negative += 1
    
    print(f"   ❌ Invalid (positive==negative): {invalid_count}")
    print(f"   ⚠️  Same class negative: {same_class_negative}")
    
    if invalid_count == 0 and same_class_negative == 0:
        print(f"   ✅ All triplets are valid!")
    
    triplet_df = pd.DataFrame(triplets, columns=["id", "anchor", "positive", "negative"])
    triplet_df.to_csv(output_csv, index=False)
    print(f"\n✅ Đã tạo file triplet dataset: {output_csv}")
    print(f"📈 Distribution: {dict(triplet_counts)}")

if __name__ == "__main__":
    generate_triplet_dataset(
        input_csv="labels.csv", 
        output_csv="triplet_dataset.csv",
        max_triplets_per_class=200,  
        random_seed=42
    )
