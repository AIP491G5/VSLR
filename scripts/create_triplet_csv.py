import pandas as pd
import random

def generate_triplet_dataset(input_csv: str, output_csv: str = "triplet_dataset.csv"):
    """
    Tạo triplet dataset từ file labels.csv với format:
    - Mỗi dòng gồm: anchor, positive (cùng gloss), negative (khác gloss)
    """
    df = pd.read_csv(input_csv)
    triplets = []

    for _, row in df.iterrows():
        label = row['label']
        videos = row['videos'].split(', ')

        if len(videos) < 2:
            continue  # Bỏ qua gloss nếu có < 2 video

        # Tạo tất cả cặp (anchor, positive)
        for i in range(len(videos)):
            for j in range(i + 1, len(videos)):
                anchor = videos[i]
                positive = videos[j]
                id = row['id']
                # Chọn ngẫu nhiên 1 negative từ gloss khác
                negative_row = df[df['label'] != label].sample(n=1).iloc[0]
                negative_video = random.choice(negative_row['videos'].split(', '))

                triplets.append((id, anchor, positive, negative_video))

    # Ghi ra file CSV
    triplet_df = pd.DataFrame(triplets, columns=["id", "anchor", "positive", "negative"])
    triplet_df.to_csv(output_csv, index=False)
    print(f"✅ Đã tạo file triplet dataset: {output_csv}")

if __name__ == "__main__":
    generate_triplet_dataset("labels.csv")  # hoặc đường dẫn tùy ý
