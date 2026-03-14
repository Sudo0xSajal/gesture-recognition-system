"""
evaluate.py — Flexible model evaluation for debugging and visualization
========================================================================
Use this during development to:
  - Test single images
  - Visualize predictions
  - Analyze model mistakes
  - Compare different models

Usage:
    python evaluate.py --image path/to/image.jpg
    python evaluate.py --folder path/to/test/images/
    python evaluate.py --interactive
    python evaluate.py --analyze-mistakes
"""

import sys
import random
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import GestureConfig
from dataset import _val_transforms
from cnn_model import build_model

cfg = GestureConfig(mode="infer")

class GestureEvaluator:
    def __init__(self, model_path=None, backbone=None):
        self.device = torch.device(cfg.inference_device)
        self.backbone = backbone or cfg.backbone
        
        # Load model
        self.model = build_model(
            backbone=self.backbone,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout_rate,
            pretrained=False
        ).to(self.device)
        
        model_path = model_path or cfg.inference_model_path
        if Path(model_path).exists():
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print(f"⚠️  Model not found: {model_path}")
        
        self.model.eval()
        self.transform = _val_transforms(cfg.image_size)
        
    def predict_image(self, image_path):
        """Predict single image"""
        # Load and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.array(Image.open(image_path).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform
        img_tensor = self.transform(image=img)['image']
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = probs.max(dim=1)
        
        pred_class = cfg.class_names.get(pred.item(), f"Class_{pred.item()}")
        
        return {
            'class_id': pred.item(),
            'class_name': pred_class,
            'confidence': conf.item(),
            'probabilities': probs[0].cpu().numpy(),
            'image': img
        }
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction on image"""
        result = self.predict_image(image_path)
        
        plt.figure(figsize=(10, 5))
        
        # Show image
        plt.subplot(1, 2, 1)
        plt.imshow(result['image'])
        plt.title(f"Prediction: {result['class_name']}\nConfidence: {result['confidence']:.3f}")
        plt.axis('off')
        
        # Show probability distribution
        plt.subplot(1, 2, 2)
        classes = [cfg.class_names.get(i, f"{i}") for i in range(cfg.num_classes)]
        colors = ['green' if i == result['class_id'] else 'blue' 
                  for i in range(cfg.num_classes)]
        plt.bar(range(cfg.num_classes), result['probabilities'], color=colors)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.xticks(range(cfg.num_classes), classes, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        plt.show()
    
    def evaluate_folder(self, folder_path, num_samples=10):
        """Evaluate random images from a folder"""
        folder = Path(folder_path)
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        images = random.sample(images, min(num_samples, len(images)))
        
        correct = 0
        for img_path in images:
            # Get true label from folder name
            true_class = int(img_path.parent.name)
            true_name = cfg.class_names.get(true_class, f"Class_{true_class}")
            
            result = self.predict_image(img_path)
            
            is_correct = (result['class_id'] == true_class)
            correct += int(is_correct)
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {img_path.name}: "
                  f"True={true_name}({true_class}), "
                  f"Pred={result['class_name']}({result['class_id']}), "
                  f"Conf={result['confidence']:.3f}")
        
        accuracy = correct / len(images)
        print(f"\nAccuracy on {len(images)} samples: {accuracy:.3f}")
    
    def analyze_mistakes(self, test_loader, num_examples=10):
        """Find and display model mistakes"""
        self.model.eval()
        mistakes = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = outputs.max(1)
                
                # Find mistakes in this batch
                for i in range(len(labels)):
                    if preds[i] != labels[i]:
                        mistakes.append({
                            'image': images[i].cpu(),
                            'true': labels[i].item(),
                            'pred': preds[i].item(),
                            'confidence': F.softmax(outputs[i], dim=0).max().item()
                        })
                        
                        if len(mistakes) >= num_examples:
                            break
                if len(mistakes) >= num_examples:
                    break
        
        # Visualize mistakes
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, mistake in enumerate(mistakes[:10]):
            img = mistake['image'].permute(1, 2, 0).numpy()
            # Denormalize
            img = img * np.array(cfg.normalize_std) + np.array(cfg.normalize_mean)
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            true_name = cfg.class_names.get(mistake['true'], f"Class_{mistake['true']}")
            pred_name = cfg.class_names.get(mistake['pred'], f"Class_{mistake['pred']}")
            axes[i].set_title(f"True: {true_name}\nPred: {pred_name}\nConf: {mistake['confidence']:.2f}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('mistakes_analysis.png', dpi=120)
        plt.show()
        print(f"Found {len(mistakes)} mistakes. Analysis saved to mistakes_analysis.png")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate gesture recognition model")
    parser.add_argument('--image', type=str, help='Single image to evaluate')
    parser.add_argument('--folder', type=str, help='Folder of images to evaluate')
    parser.add_argument('--model-path', type=str, default=cfg.inference_model_path)
    parser.add_argument('--backbone', type=str, default=cfg.backbone)
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--analyze-mistakes', action='store_true', 
                       help='Analyze model mistakes on test set')
    parser.add_argument('--num-samples', type=int, default=10, 
                       help='Number of samples for folder evaluation')
    
    args = parser.parse_args()
    
    evaluator = GestureEvaluator(args.model_path, args.backbone)
    
    if args.image:
        evaluator.visualize_prediction(args.image)
    
    elif args.folder:
        evaluator.evaluate_folder(args.folder, args.num_samples)
    
    elif args.analyze_mistakes:
        from dataset import create_dataloaders
        loaders = create_dataloaders(batch_size=32)
        evaluator.analyze_mistakes(loaders['test'], args.num_samples)
    
    elif args.interactive:
        print("\nInteractive Evaluation Mode")
        print("Enter image paths (or 'quit' to exit):")
        while True:
            path = input("\nImage path: ").strip()
            if path.lower() in ['quit', 'exit', 'q']:
                break
            if Path(path).exists():
                evaluator.visualize_prediction(path)
            else:
                print(f"File not found: {path}")

if __name__ == '__main__':
    main()