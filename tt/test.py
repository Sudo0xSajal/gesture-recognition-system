"""
test.py — Final model evaluation on test set
=============================================
Run this AFTER training to get final performance metrics.

Usage:
    python test.py                          # test with best model from training
    python test.py --model-path /path/to/model.pth
    python test.py --backbone gesturecnn     # test different architecture
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from config import GestureConfig
from dataset import create_dataloaders
from cnn_model import build_model

cfg = GestureConfig(mode="eval")

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to: {save_path}")

def test_model(model, test_loader, device, class_names):
    """Evaluate model on test set and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class metrics
    target_names = [class_names.get(i, f"Class_{i}") for i in range(cfg.num_classes)]
    report = classification_report(all_labels, all_preds, 
                                  target_names=target_names,
                                  output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': float(accuracy),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': np.array(all_probs).tolist(),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

def main():
    parser = argparse.ArgumentParser(description="Test trained gesture recognition model")
    parser.add_argument('--model-path', type=str, default=cfg.best_model_path,
                        help='Path to trained model checkpoint')
    parser.add_argument('--backbone', type=str, default=cfg.backbone,
                        choices=['mobilenetv2', 'gesturecnn'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--save-dir', type=str, default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    args = parser.parse_args()
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Load test data
    print("\nLoading test dataset...")
    loaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=2,
        use_weighted_sampler=False  # No sampler needed for testing
    )
    test_loader = loaders['test']
    
    # Build model
    print(f"\nBuilding model: {args.backbone}")
    model = build_model(
        backbone=args.backbone,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout_rate,
        pretrained=False  # Don't need pretrained for testing
    ).to(device)
    
    # Load trained weights
    if Path(args.model_path).exists():
        print(f"Loading weights from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"⚠️  Model not found: {args.model_path}")
        return
    
    # Run test evaluation
    results = test_model(model, test_loader, device, cfg.class_names)
    
    # Save results
    print("\n" + "="*60)
    print("  TEST RESULTS")
    print("="*60)
    print(f"  Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Macro F1: {results['classification_report']['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1: {results['classification_report']['weighted avg']['f1-score']:.4f}")
    print("\n  Per-Class Performance:")
    print("-"*60)
    
    for i in range(cfg.num_classes):
        class_name = cfg.class_names.get(i, f"Class_{i}")
        precision = results['classification_report'][class_name]['precision']
        recall = results['classification_report'][class_name]['recall']
        f1 = results['classification_report'][class_name]['f1-score']
        support = results['classification_report'][class_name]['support']
        print(f"    {class_name:<12} | prec: {precision:.3f} | recall: {recall:.3f} | "
              f"f1: {f1:.3f} | support: {int(support):3d}")
    
    # Save JSON results
    results_path = save_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        # Remove large arrays for JSON
        json_results = {
            'accuracy': results['accuracy'],
            'classification_report': results['classification_report'],
            'num_classes': cfg.num_classes,
            'class_names': cfg.class_names
        }
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    # Plot and save confusion matrix
    cm_path = save_dir / 'confusion_matrix.png'
    plot_confusion_matrix(
        np.array(results['confusion_matrix']),
        [cfg.class_names.get(i, f"Class_{i}") for i in range(cfg.num_classes)],
        cm_path
    )
    
    print(f"\n✅ Test evaluation complete!")

if __name__ == '__main__':
    main()