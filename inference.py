import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import argparse
from torchmetrics.classification import MulticlassAveragePrecision
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import os
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_and_plot_grid(model, processor, dataset, classes, grid_size=3):
    model.eval()
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    for i in range(grid_size):
        for j in range(grid_size):
            idx = random.randint(0, len(dataset) - 1)
            img, true_label = dataset[idx]
            
            # Preprocess the image and text labels
            inputs = processor(text=[f"a photo of a {c}" for c in classes], images=img, return_tensors="pt", padding=True).to(device)

            with torch.inference_mode():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                predicted_label = probs.argmax(-1).item()

            img_display = np.array(img)
            axes[i, j].imshow(img_display)
            
            truth = classes[true_label] == classes[predicted_label]
            color = "g" if truth else "r"

            axes[i, j].set_title(f"Truth: {classes[true_label]} Predicted: {classes[predicted_label]}", fontsize=10, c=color)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()

def evaluate_model(model, processor, test_dataloader, classes, device):
    print("Running evaluation...")
    num_classes = len(classes)
    metric = MulticlassAveragePrecision(num_classes=num_classes, average=None, thresholds=None).to(device)
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Evaluating"):
            labels = labels.to(device)

            # Preprocess the images and text labels
            inputs = processor(text=[f"a photo of a {c}" for c in classes], images=images, return_tensors="pt", padding=True).to(device)
            
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            
            all_preds.append(logits_per_image)
            all_targets.append(labels)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metric.update(all_preds, all_targets)
    ap_per_class = metric.compute()

    print("Class Wise Average Precisions")
    for i, ap in enumerate(ap_per_class):
        print(f"AP for class {classes[i]} = {ap:.4f}")
    
    mAP = ap_per_class.mean()
    print(f"Mean Average Precision : {mAP:.4f}")

if __name__ == '__main__':
    # Load the dataset
    test_dataset = CIFAR10(root=os.path.expanduser("./data"), download=True, train=False)
    classes = test_dataset.classes

    # Load CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    parser = argparse.ArgumentParser(description="CLIP Inference and Evaluation")
    parser.add_argument("--infer", action="store_true", help="Run inference and plot predictions")
    parser.add_argument("--eval", action="store_true", help="Evaluate model performance (mAP)")
    args = parser.parse_args()

    if args.infer:
        predict_and_plot_grid(model, processor, test_dataset, classes, grid_size=3)
    elif args.eval:
        def collate_fn(batch):
            images = [item[0] for item in batch]
            labels = torch.tensor([item[1] for item in batch])
            return images, labels
        test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
        evaluate_model(model, processor, test_dataloader, classes, device)
