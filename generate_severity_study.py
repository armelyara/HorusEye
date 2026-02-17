"""
Generate samples at different severity levels
For analyzing how degradation intensity affects grounding
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from degradation_pipeline import RefCOCOLoader, DegradationPipeline


def generate_severity_comparison(loader, output_dir, num_samples=10, severities=None):
    """
    Generate images at multiple severity levels
    
    Args:
        loader: RefCOCOLoader instance
        output_dir: where to save outputs
        num_samples: how many samples to generate
        severities: list of severity values to test
    """
    if severities is None:
        severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    degradation_types = ['fog', 'smoke', 'thermal']
    
    # Create directories
    for deg_type in degradation_types:
        for sev in severities:
            path = os.path.join(output_dir, deg_type, f'severity_{sev:.1f}')
            os.makedirs(path, exist_ok=True)
    
    # Also save clean images
    os.makedirs(os.path.join(output_dir, 'clean'), exist_ok=True)
    
    print(f"Generating {num_samples} samples at {len(severities)} severity levels...")
    
    for idx in tqdm(range(min(num_samples, len(loader)))):
        sample = loader[idx]
        image = sample['image']
        
        # Save clean image
        clean_path = os.path.join(output_dir, 'clean', f'{idx:04d}.jpg')
        cv2.imwrite(clean_path, image)
        
        # Generate degraded versions
        for deg_type in degradation_types:
            for sev in severities:
                pipeline = DegradationPipeline(severity=sev)
                degraded = pipeline.apply_single(image, deg_type)
                
                save_path = os.path.join(output_dir, deg_type, f'severity_{sev:.1f}', f'{idx:04d}.jpg')
                cv2.imwrite(save_path, degraded)
    
    print(f"✓ Saved to {output_dir}")


def visualize_severity_grid(loader, sample_idx, save_path=None):
    """
    Create a grid showing all degradations at all severities
    
    Args:
        loader: RefCOCOLoader instance
        sample_idx: which sample to visualize
        save_path: where to save the figure
    """
    sample = loader[sample_idx]
    image = sample['image']
    expression = sample['expression']
    bbox = sample['bbox']
    
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    degradation_types = ['fog', 'smoke', 'thermal']
    
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    
    for row, deg_type in enumerate(degradation_types):
        # First column: clean image
        img_vis = image.copy()
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
        img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title('Clean' if row == 0 else '')
        axes[row, 0].set_ylabel(deg_type.capitalize(), fontsize=12)
        axes[row, 0].axis('off')
        
        # Other columns: different severities
        for col, sev in enumerate(severities):
            pipeline = DegradationPipeline(severity=sev)
            degraded = pipeline.apply_single(image, deg_type)
            
            img_vis = degraded.copy()
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
            img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            
            axes[row, col+1].imshow(img_rgb)
            axes[row, col+1].set_title(f'Sev: {sev}' if row == 0 else '')
            axes[row, col+1].axis('off')
    
    # Truncate expression for display
    display_expr = expression if len(expression) < 50 else expression[:47] + "..."
    plt.suptitle(f'Expression: "{display_expr}"', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def run_severity_evaluation(dataset_class, refcoco_path, coco_images_path, 
                            model, num_samples=50, severities=None):
    """
    Evaluate grounding at multiple severity levels
    
    Returns:
        dict with results for each severity level
    """
    from grounding_model import calculate_iou
    
    if severities is None:
        severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    degradation_types = ['fog', 'smoke', 'thermal']
    
    results = {
        deg_type: {sev: [] for sev in severities}
        for deg_type in degradation_types
    }
    
    # Also track clean performance
    results['clean'] = []
    
    print(f"\nRunning severity study: {num_samples} samples × {len(severities)} severities...")
    
    for sev in severities:
        print(f"\n  Severity {sev}:")
        
        dataset = dataset_class(
            refcoco_path=refcoco_path,
            coco_images_path=coco_images_path,
            split='val',
            severity=sev
        )
        
        for idx in tqdm(range(min(num_samples, len(dataset))), desc=f"    Evaluating"):
            sample = dataset[idx]
            expression = sample['expression']
            gt_bbox = sample['bbox']
            
            # Evaluate clean (only once, at sev=0)
            if sev == severities[0]:
                pred = model.predict(sample['images']['clean'], expression)
                iou = calculate_iou(pred['bbox'], gt_bbox)
                results['clean'].append(iou)
            
            # Evaluate each degradation type
            for deg_type in degradation_types:
                pred = model.predict(sample['images'][deg_type], expression)
                iou = calculate_iou(pred['bbox'], gt_bbox)
                results[deg_type][sev].append(iou)
    
    return results


def plot_severity_curves(results, save_path=None):
    """
    Plot how accuracy changes with severity
    
    Args:
        results: output from run_severity_evaluation
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'fog': '#3498db', 'smoke': '#9b59b6', 'thermal': '#e74c3c'}
    severities = sorted(list(results['fog'].keys()))
    
    # Plot 1: Mean IoU vs Severity
    ax1 = axes[0]
    
    # Clean baseline
    clean_mean = np.mean(results['clean'])
    ax1.axhline(y=clean_mean, color='#2ecc71', linestyle='--', 
                label=f'Clean ({clean_mean:.3f})', linewidth=2)
    
    for deg_type in ['fog', 'smoke', 'thermal']:
        means = [np.mean(results[deg_type][sev]) for sev in severities]
        stds = [np.std(results[deg_type][sev]) for sev in severities]
        
        ax1.plot(severities, means, 'o-', color=colors[deg_type], 
                label=deg_type.capitalize(), linewidth=2, markersize=8)
        ax1.fill_between(severities, 
                        [m-s for m,s in zip(means, stds)],
                        [m+s for m,s in zip(means, stds)],
                        color=colors[deg_type], alpha=0.2)
    
    ax1.set_xlabel('Degradation Severity', fontsize=12)
    ax1.set_ylabel('Mean IoU', fontsize=12)
    ax1.set_title('Grounding Performance vs Degradation Severity', fontsize=14)
    ax1.legend()
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance drop vs Severity
    ax2 = axes[1]
    
    for deg_type in ['fog', 'smoke', 'thermal']:
        drops = []
        for sev in severities:
            mean_iou = np.mean(results[deg_type][sev])
            drop = (clean_mean - mean_iou) / clean_mean * 100
            drops.append(drop)
        
        ax2.plot(severities, drops, 'o-', color=colors[deg_type],
                label=deg_type.capitalize(), linewidth=2, markersize=8)
    
    ax2.set_xlabel('Degradation Severity', fontsize=12)
    ax2.set_ylabel('Performance Drop (%)', fontsize=12)
    ax2.set_title('Performance Degradation vs Severity', fontsize=14)
    ax2.legend()
    ax2.set_xlim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


# ==================== MAIN ====================
if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    REFCOCO_PATH = 'rq1_datasets/refcoco/refer/data/refcoco'
    COCO_IMAGES_PATH = 'rq1_datasets/coco/images/train2014/'
    OUTPUT_DIR = 'results/severity_study'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ===== LOAD DATA =====
    print("Loading data...")
    from degradation_pipeline import RefCOCOLoader, RefCOCODegradationDataset
    
    loader = RefCOCOLoader(
        refcoco_path=REFCOCO_PATH,
        coco_images_path=COCO_IMAGES_PATH,
        split='val'
    )
    
    # ===== GENERATE VISUALIZATIONS =====
    print("\nGenerating severity grid visualizations...")
    for i in range(3):
        visualize_severity_grid(
            loader, 
            sample_idx=i,
            save_path=os.path.join(OUTPUT_DIR, f'severity_grid_{i}.png')
        )
    
    # ===== OPTIONAL: Full severity evaluation =====
    # Uncomment below to run full evaluation (takes longer)
    
    print("\nLoading grounding model...")
    from grounding_model import GroundingModel
    model = GroundingModel()
    
    print("\nRunning severity evaluation...")
    results = run_severity_evaluation(
    RefCOCODegradationDataset,
    REFCOCO_PATH,
    COCO_IMAGES_PATH,
    model,
    num_samples=200,
    severities=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    
    plot_severity_curves(results, save_path=os.path.join(OUTPUT_DIR, 'severity_curves.png'))
    
    print(f"\n✓ Done! Check {OUTPUT_DIR} for outputs.")