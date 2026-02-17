"""
Evaluate visual grounding under different degradation conditions
Answers RQ1: How does visual grounding degrade under smoke, fog, and thermal?
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Import our modules
from degradation_pipeline import RefCOCODegradationDataset
from grounding_model import GroundingModel, calculate_iou


class GroundingEvaluator:
    """Evaluate grounding model on degraded images"""
    
    def __init__(self, dataset, model):
        """
        Args:
            dataset: RefCOCODegradationDataset instance
            model: GroundingModel instance
        """
        self.dataset = dataset
        self.model = model
        self.results = None
    
    def evaluate(self, num_samples=None, conditions=None):
        """
        Run evaluation on all conditions
        
        Args:
            num_samples: number of samples to evaluate (None = all)
            conditions: list of conditions to test (None = all)
            
        Returns:
            dict with results for each condition
        """
        if conditions is None:
            conditions = ['clean', 'fog', 'smoke', 'thermal']
        
        if num_samples is None:
            num_samples = len(self.dataset)
        else:
            num_samples = min(num_samples, len(self.dataset))
        
        # Initialize results storage
        results = {
            condition: {
                'ious': [],
                'confidences': [],
                'detections': [],  # True if box was detected
                'samples': []
            }
            for condition in conditions
        }
        
        print(f"\nEvaluating {num_samples} samples across {len(conditions)} conditions...")
        print("=" * 60)
        
        for idx in tqdm(range(num_samples), desc="Evaluating"):
            sample = self.dataset[idx]
            expression = sample['expression']
            gt_bbox = sample['bbox']
            
            for condition in conditions:
                image = sample['images'][condition]
                
                # Get prediction
                pred = self.model.predict(image, expression)
                pred_bbox = pred['bbox']
                confidence = pred['confidence']
                
                # Calculate IoU
                iou = calculate_iou(pred_bbox, gt_bbox)
                
                # Store results
                results[condition]['ious'].append(iou)
                results[condition]['confidences'].append(confidence)
                results[condition]['detections'].append(pred_bbox is not None)
                results[condition]['samples'].append({
                    'idx': idx,
                    'expression': expression,
                    'gt_bbox': gt_bbox,
                    'pred_bbox': pred_bbox,
                    'iou': iou,
                    'confidence': confidence
                })
        
        # Calculate summary statistics
        for condition in conditions:
            ious = results[condition]['ious']
            confs = results[condition]['confidences']
            dets = results[condition]['detections']
            
            results[condition]['summary'] = {
                'mean_iou': np.mean(ious),
                'std_iou': np.std(ious),
                'median_iou': np.median(ious),
                'mean_confidence': np.mean(confs),
                'detection_rate': np.mean(dets),
                'accuracy_0.5': np.mean([iou >= 0.5 for iou in ious]),  # IoU >= 0.5
                'accuracy_0.7': np.mean([iou >= 0.7 for iou in ious]),  # IoU >= 0.7
                'num_samples': len(ious)
            }
        
        self.results = results
        return results
    
    def print_summary(self):
        """Print evaluation summary"""
        if self.results is None:
            print("No results yet. Run evaluate() first.")
            return
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS: Visual Grounding Under Degradation")
        print("=" * 70)
        
        # Header
        print(f"\n{'Condition':<12} {'Mean IoU':<12} {'Acc@0.5':<12} {'Acc@0.7':<12} {'Det Rate':<12}")
        print("-" * 60)
        
        # Get clean baseline for comparison
        clean_iou = self.results['clean']['summary']['mean_iou']
        
        for condition in self.results:
            summary = self.results[condition]['summary']
            
            # Calculate drop from clean
            iou_drop = ((clean_iou - summary['mean_iou']) / clean_iou * 100) if condition != 'clean' else 0
            
            print(f"{condition:<12} "
                  f"{summary['mean_iou']:.4f}      "
                  f"{summary['accuracy_0.5']:.4f}      "
                  f"{summary['accuracy_0.7']:.4f}      "
                  f"{summary['detection_rate']:.4f}")
            
            if condition != 'clean':
                print(f"{'':12} (↓{iou_drop:.1f}% from clean)")
        
        print("-" * 60)
        print(f"Total samples per condition: {self.results['clean']['summary']['num_samples']}")
    
    def plot_results(self, save_path=None):
        """Create visualization of results"""
        if self.results is None:
            print("No results yet. Run evaluate() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        conditions = list(self.results.keys())
        colors = {'clean': '#2ecc71', 'fog': '#3498db', 'smoke': '#9b59b6', 'thermal': '#e74c3c'}
        
        # Plot 1: Mean IoU comparison
        ax1 = axes[0, 0]
        means = [self.results[c]['summary']['mean_iou'] for c in conditions]
        stds = [self.results[c]['summary']['std_iou'] for c in conditions]
        bars = ax1.bar(conditions, means, yerr=stds, capsize=5,
                       color=[colors[c] for c in conditions], alpha=0.8)
        ax1.set_ylabel('Mean IoU')
        ax1.set_title('Grounding Accuracy by Condition')
        ax1.set_ylim(0, 1)
        # Add value labels
        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: IoU Distribution (Box plot)
        ax2 = axes[0, 1]
        iou_data = [self.results[c]['ious'] for c in conditions]
        bp = ax2.boxplot(iou_data, labels=conditions, patch_artist=True)
        for patch, c in zip(bp['boxes'], conditions):
            patch.set_facecolor(colors[c])
            patch.set_alpha(0.8)
        ax2.set_ylabel('IoU')
        ax2.set_title('IoU Distribution by Condition')
        
        # Plot 3: Accuracy at different thresholds
        ax3 = axes[1, 0]
        thresholds = [0.3, 0.5, 0.7, 0.9]
        x = np.arange(len(thresholds))
        width = 0.2
        
        for i, condition in enumerate(conditions):
            ious = self.results[condition]['ious']
            accs = [np.mean([iou >= t for iou in ious]) for t in thresholds]
            ax3.bar(x + i*width, accs, width, label=condition, 
                   color=colors[condition], alpha=0.8)
        
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('IoU Threshold')
        ax3.set_title('Accuracy at Different IoU Thresholds')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels([f'≥{t}' for t in thresholds])
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # Plot 4: Degradation drop from clean
        ax4 = axes[1, 1]
        clean_iou = self.results['clean']['summary']['mean_iou']
        drops = []
        drop_conditions = []
        for c in conditions:
            if c != 'clean':
                drop = (clean_iou - self.results[c]['summary']['mean_iou']) / clean_iou * 100
                drops.append(drop)
                drop_conditions.append(c)
        
        bars = ax4.bar(drop_conditions, drops, 
                      color=[colors[c] for c in drop_conditions], alpha=0.8)
        ax4.set_ylabel('Performance Drop (%)')
        ax4.set_title('Degradation Impact (% Drop from Clean)')
        # Add value labels
        for bar, drop in zip(bars, drops):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{drop:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def save_results(self, save_path):
        """Save results to JSON file"""
        if self.results is None:
            print("No results yet. Run evaluate() first.")
            return
        
        # Prepare serializable results
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': self.results['clean']['summary']['num_samples'],
            'severity': self.dataset.severity,
            'conditions': {}
        }
        
        for condition in self.results:
            save_data['conditions'][condition] = {
                'summary': self.results[condition]['summary'],
                'ious': self.results[condition]['ious'],
                'confidences': self.results[condition]['confidences']
            }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to {save_path}")


# ==================== MAIN ====================
if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    REFCOCO_PATH = 'rq1_datasets/refcoco/'
    COCO_IMAGES_PATH = 'rq1_datasets/coco/images/train2014/'
    NUM_SAMPLES = 100  # Start small, increase later
    SEVERITY = 0.5
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # ===== SETUP =====
    print("Loading dataset...")
    dataset = RefCOCODegradationDataset(
        refcoco_path=REFCOCO_PATH,
        coco_images_path=COCO_IMAGES_PATH,
        split='val',  # Use validation set for evaluation
        severity=SEVERITY
    )
    
    print("Loading grounding model...")
    model = GroundingModel()
    
    # ===== EVALUATE =====
    evaluator = GroundingEvaluator(dataset, model)
    results = evaluator.evaluate(num_samples=NUM_SAMPLES)
    
    # ===== RESULTS =====
    evaluator.print_summary()
    evaluator.plot_results(save_path='results/grounding_evaluation.png')
    evaluator.save_results(save_path='results/grounding_results.json')
    
    print("\n✓ Evaluation complete! Check the 'results' folder.")