import os
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import transforms
from PIL import Image

# Load pretrained MobileNet-V3 (feature extractor)
model = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True)
model.eval()
cfg = model.default_cfg

preprocess = transforms.Compose([
    transforms.Resize((cfg['input_size'][1], cfg['input_size'][2])),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg['mean'], std=cfg['std']),
])


def get_embedding(path):
    img = Image.open(path).convert('RGB')
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feats = model(x)
    return feats[-1].flatten().cpu().numpy()

if __name__ == "__main__":
    df = pd.read_csv("features.csv")
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "Data"))

    embeddings = []
    skipped = []

    for relpath in df["relpath"]:  # Now using proper relative path
        img_path = os.path.join(data_dir, relpath)
        if not os.path.isfile(img_path):
            skipped.append(relpath)
            print(f"⚠️ Skipping missing file: {img_path}")
            continue
        embeddings.append(get_embedding(img_path))

    if embeddings:
        embeddings = np.vstack(embeddings)
        pd.DataFrame(embeddings).to_csv("cnn_embeddings.csv", index=False)
        print(f"✅ Saved embeddings for {len(embeddings)} images.")
    else:
        print("⚠️ No embeddings were generated.")

    if skipped:
        print(f"⚠️ Skipped {len(skipped)} files not found.")


