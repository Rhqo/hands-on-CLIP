import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import os
import numpy as np

def calculate_and_visualize_similarity(image_path, texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Processing image and texts...")
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy().flatten()
        
        # For direct comparison, we also show the raw cosine similarity.
        # We get the embeddings and calculate it manually.
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        cosine_similarities = (image_features @ text_features.T).squeeze().cpu().numpy()

    print("Calculating scores...")

    # Visualize the results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Image-Text Similarity Analysis using CLIP', fontsize=16)

    # Plot 1: Cosine Similarity
    axes[0].bar(np.arange(len(texts)), cosine_similarities, color='skyblue')
    axes[0].set_ylabel("Cosine Similarity", fontsize=12)
    axes[0].set_title("Raw Cosine Similarity Scores", fontsize=14)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, score in enumerate(cosine_similarities):
        axes[0].text(i, score, f'{score:.4f}', ha='center', va='bottom' if score >= 0 else 'top')

    # Plot 2: Softmax Probabilities
    axes[1].bar(np.arange(len(texts)), probs, color='lightgreen')
    axes[1].set_ylabel("Probability", fontsize=12)
    axes[1].set_title("Softmax Probabilities", fontsize=14)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    for i, prob in enumerate(probs):
        axes[1].text(i, prob, f'{prob:.4f}', ha='center', va='bottom')

    plt.xticks(np.arange(len(texts)), texts, rotation=45, ha='right', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print(f"Displaying similarity scores for '{image_path}'")
    plt.show()

def main():
    image_path = "./image.jpg"

    texts = [
        "a photo of a cat",
        "a photo of a cute cat",
        "a photo of a ugly cat",
        "a photo of a kitty",
        "a photo of a bird",
        "a photo of a ship",
        "a photo of a truck",
        "a photo of a horse",
        "a photo of a plane",
        "a photo of a deer"
    ]

    calculate_and_visualize_similarity(image_path, texts)

if __name__ == '__main__':
    main()
