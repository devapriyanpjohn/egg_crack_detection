import cv2
import os
import numpy as np
import pandas as pd

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    edge_pixel_count = np.sum(edges > 0)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_contours = [c for c in contours if cv2.contourArea(c) < 5000]
    num_crack_contours = len(small_contours)

    if not contours:
        return (edge_pixel_count, 0, 0, 0, np.std(gray), num_crack_contours)

    largest = max(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    circularity = (4 * np.pi * max_area) / (perimeter ** 2) if perimeter > 0 else 0
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = max_area / hull_area if hull_area > 0 else 0
    std_intensity = np.std(gray)

    return (edge_pixel_count, max_area, circularity, solidity, std_intensity, num_crack_contours)

def process_directory(base_dir):
    data = []
    print(f"📂 Scanning directory: {base_dir}")

    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, filename)
                relpath = os.path.relpath(full_path, base_dir)  # <--- Relative path for later use
                print(f"🔍 Processing: {relpath}")

                lower = relpath.lower()
                if "normal" in lower or "not_damaged" in lower:
                    label = 0
                elif "cracked" in lower or "damaged" in lower:
                    label = 1
                else:
                    print(f"⚠️ Skipping (unknown label): {relpath}")
                    continue

                features = extract_features(full_path)
                if features is None:
                    continue

                (edge_pixels, max_area, circularity,
                 solidity, std_intensity, num_crack_contours) = features

                data.append([
                    relpath, edge_pixels, max_area, circularity,
                    solidity, std_intensity, num_crack_contours, label
                ])

    print(f"✅ Processed {len(data)} valid images.")
    return data

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    image_dir = os.path.abspath(os.path.join(current_dir, "..", "Data"))

    dataset = process_directory(image_dir)

    if dataset:
        output_path = os.path.join(current_dir, "features.csv")
        df = pd.DataFrame(dataset, columns=[
            "relpath", "edge_pixels", "max_area", "circularity",
            "solidity", "std_intensity", "num_crack_contours", "label"
        ])
        df.to_csv(output_path, index=False)
        print(f"📄 Feature extraction complete → {output_path}")
    else:
        print("⚠️ No valid data found; CSV not created.")




