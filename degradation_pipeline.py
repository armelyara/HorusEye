"""
RefCOCO Degradation Pipeline
Applies fog, smoke, and thermal effects to clean images
No external refer package needed - standalone version
"""

import os
import cv2
import json
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Try to import imgaug, if not available use only physics-based fog
try:
    import imgaug.augmenters as iaa
    HAS_IMGAUG = True
except ImportError:
    HAS_IMGAUG = False
    print("Warning: imgaug not installed. Using physics-based fog only.")


# ==================== STANDALONE REFCOCO LOADER ====================
class RefCOCOLoader:
    """
    Simple RefCOCO loader without external dependencies
    Replaces the need for 'pip install refer'
    """
    
    def __init__(self, refcoco_path, coco_images_path, split='train'):
        """
        Args:
            refcoco_path: path to refcoco folder containing refs(unc).p and instances.json
            coco_images_path: path to COCO train2014 images folder
            split: 'train', 'val', 'testA', or 'testB'
        """
        self.coco_images_path = coco_images_path
        self.split = split
        
        # Load refs
        refs_file = os.path.join(refcoco_path, 'refs(unc).p')
        if not os.path.exists(refs_file):
            raise FileNotFoundError(f"refs(unc).p not found at {refs_file}")
        
        with open(refs_file, 'rb') as f:
            all_refs = pickle.load(f)
        
        # Load instances
        instances_file = os.path.join(refcoco_path, 'instances.json')
        if not os.path.exists(instances_file):
            raise FileNotFoundError(f"instances.json not found at {instances_file}")
        
        with open(instances_file, 'r') as f:
            instances = json.load(f)
        
        # Build image lookup
        self.images = {img['id']: img for img in instances['images']}
        
        # Build annotation lookup
        self.annotations = {ann['id']: ann for ann in instances['annotations']}
        
        # Filter refs by split
        self.refs = [r for r in all_refs if r['split'] == split]
        
        print(f"Loaded {len(self.refs)} referring expressions for split '{split}'")
        print(f"Total images available: {len(self.images)}")
    
    def __len__(self):
        return len(self.refs)
    
    def get_ref_ids(self):
        """Get all ref_ids"""
        return [r['ref_id'] for r in self.refs]
    
    def __getitem__(self, idx):
        """Get one sample by index"""
        ref = self.refs[idx]
        
        # Get image info
        image_id = ref['image_id']
        image_info = self.images[image_id]
        image_path = os.path.join(self.coco_images_path, image_info['file_name'])
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        
        # Get expression (first sentence)
        expression = ref['sentences'][0]['sent']
        
        # Get all sentences
        all_expressions = [s['sent'] for s in ref['sentences']]
        
        # Get bounding box
        ann_id = ref['ann_id']
        ann = self.annotations[ann_id]
        bbox = ann['bbox']  # [x, y, width, height]
        
        # Get segmentation if available
        segmentation = ann.get('segmentation', None)
        
        return {
            'image': image,
            'expression': expression,
            'all_expressions': all_expressions,
            'bbox': bbox,
            'segmentation': segmentation,
            'image_id': image_id,
            'ref_id': ref['ref_id'],
            'category_id': ann.get('category_id', None)
        }


# ==================== DEGRADATION PIPELINE ====================
class DegradationPipeline:
    """Apply various degradations to images"""
    
    def __init__(self, severity=0.5):
        """
        Args:
            severity: float 0-1, how strong the degradation is
        """
        self.severity = np.clip(severity, 0.0, 1.0)
    
    # ==================== FOG ====================
    def add_fog(self, image):
        """
        Apply synthetic fog using imgaug (if available) or physics model
        
        Args:
            image: numpy array (H, W, 3) BGR format
        Returns:
            foggy image
        """
        if HAS_IMGAUG:
            # Convert severity to imgaug format (1-5 scale)
            imgaug_severity = int(self.severity * 4) + 1
            fog_augmenter = iaa.Fog()
            foggy = fog_augmenter(image=image)
            return foggy
        else:
            return self.add_fog_physics(image)
    
    def add_fog_physics(self, image, depth_map=None):
        """
        Apply fog using atmospheric scattering model
        I_fog = I_clean * t + A * (1 - t)
        
        Args:
            image: numpy array (H, W, 3)
            depth_map: optional depth map, if None uses gradient
        Returns:
            foggy image
        """
        image_float = image.astype(np.float32) / 255.0
        h, w = image.shape[:2]
        
        # If no depth map, create synthetic one
        # Assumes distance increases from top to bottom (typical for outdoor scenes)
        if depth_map is None:
            depth_map = np.tile(
                np.linspace(0.3, 1.0, h).reshape(h, 1),
                (1, w)
            )
            # Add some random variation
            noise = cv2.GaussianBlur(
                np.random.rand(h, w).astype(np.float32) * 0.2,
                (21, 21), 0
            )
            depth_map = np.clip(depth_map + noise, 0, 1)
        
        # Atmospheric light (white/gray fog)
        A = 0.8
        
        # Transmission map: t = exp(-beta * d)
        beta = self.severity * 3.0  # fog density
        t = np.exp(-beta * depth_map)
        t = np.stack([t] * 3, axis=-1)  # expand to 3 channels
        
        # Apply atmospheric scattering model
        foggy = image_float * t + A * (1 - t)
        foggy = np.clip(foggy * 255, 0, 255).astype(np.uint8)
        
        return foggy
    
    # ==================== SMOKE ====================
    def add_smoke(self, image):
        """
        Apply synthetic smoke using procedural noise
        
        Args:
            image: numpy array (H, W, 3)
        Returns:
            smoky image
        """
        h, w = image.shape[:2]
        image_float = image.astype(np.float32) / 255.0
        
        # Generate smoke texture using multiple noise layers
        smoke = self._generate_smoke_texture(h, w)
        
        # Smoke color (gray-white with slight blue tint)
        smoke_color = np.array([0.75, 0.78, 0.82])  # BGR format
        smoke_rgb = np.stack([smoke] * 3, axis=-1) * smoke_color
        
        # Blend based on severity and smoke density
        alpha = smoke * self.severity * 0.9  # max 90% opacity
        alpha = np.stack([alpha] * 3, axis=-1)
        
        smoky = image_float * (1 - alpha) + smoke_rgb * alpha
        smoky = np.clip(smoky * 255, 0, 255).astype(np.uint8)
        
        return smoky
    
    def _generate_smoke_texture(self, h, w):
        """Generate procedural smoke using multi-scale noise"""
        smoke = np.zeros((h, w), dtype=np.float32)
        
        # Add multiple scales of noise for realistic smoke
        for scale in [4, 8, 16, 32, 64]:
            noise_h = max(h // scale + 1, 2)
            noise_w = max(w // scale + 1, 2)
            noise = np.random.rand(noise_h, noise_w).astype(np.float32)
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
            smoke += noise * (scale / 64.0)  # weight by scale
        
        # Normalize to 0-1
        smoke = (smoke - smoke.min()) / (smoke.max() - smoke.min() + 1e-8)
        
        # Apply non-linear transform for more realistic smoke patches
        smoke = np.clip(smoke * 1.5 - 0.3, 0, 1)
        
        # Smooth the result
        smoke = cv2.GaussianBlur(smoke, (15, 15), 0)
        
        return smoke
    
    # ==================== THERMAL ====================
    def add_thermal(self, image):
        """
        Simulate thermal/infrared appearance
        
        Args:
            image: numpy array (H, W, 3) BGR format
        Returns:
            thermal-style image
        """
        # Convert to grayscale (approximates heat signature)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Invert so brighter areas appear "hotter"
        # This is a simplification - real thermal depends on material properties
        gray_inverted = 255 - gray
        
        # Blend original and inverted based on intensity
        # Simulates that some objects emit more heat
        gray_thermal = cv2.addWeighted(gray, 0.3, gray_inverted, 0.7, 0)
        
        # Apply thermal colormap
        # Options: COLORMAP_INFERNO, COLORMAP_HOT, COLORMAP_JET
        thermal = cv2.applyColorMap(gray_thermal, cv2.COLORMAP_INFERNO)
        
        # Reduce detail (thermal cameras have lower resolution)
        blur_amount = int(3 * self.severity) * 2 + 1  # must be odd: 1, 3, 5, 7
        if blur_amount > 1:
            thermal = cv2.GaussianBlur(thermal, (blur_amount, blur_amount), 0)
        
        # Add sensor noise
        noise_level = 8 * self.severity
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, thermal.shape)
            thermal = np.clip(thermal.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return thermal
    
    # ==================== APPLY ALL ====================
    def apply_all(self, image):
        """
        Apply all degradations and return dictionary
        
        Args:
            image: numpy array (H, W, 3)
        Returns:
            dict with 'clean', 'fog', 'smoke', 'thermal' keys
        """
        return {
            'clean': image.copy(),
            'fog': self.add_fog_physics(image),
            'smoke': self.add_smoke(image),
            'thermal': self.add_thermal(image)
        }
    
    def apply_single(self, image, degradation_type):
        """
        Apply a single degradation type
        
        Args:
            image: numpy array (H, W, 3)
            degradation_type: 'fog', 'smoke', or 'thermal'
        Returns:
            degraded image
        """
        if degradation_type == 'fog':
            return self.add_fog_physics(image)
        elif degradation_type == 'smoke':
            return self.add_smoke(image)
        elif degradation_type == 'thermal':
            return self.add_thermal(image)
        elif degradation_type == 'clean':
            return image.copy()
        else:
            raise ValueError(f"Unknown degradation type: {degradation_type}")


# ==================== MAIN DATASET CLASS ====================
class RefCOCODegradationDataset:
    """Load RefCOCO and apply degradations"""
    
    def __init__(self, refcoco_path, coco_images_path, split='train', severity=0.5):
        """
        Args:
            refcoco_path: path to refcoco annotations folder
            coco_images_path: path to COCO train2014 images folder
            split: 'train', 'val', 'testA', or 'testB'
            severity: degradation severity 0-1
        """
        self.loader = RefCOCOLoader(refcoco_path, coco_images_path, split)
        self.degradation = DegradationPipeline(severity=severity)
        self.severity = severity
    
    def __len__(self):
        return len(self.loader)
    
    def __getitem__(self, idx):
        """
        Get one sample with all degradations
        
        Returns:
            dict with:
                - 'images': dict of clean/fog/smoke/thermal images
                - 'expression': referring expression text
                - 'all_expressions': list of all expressions for this region
                - 'bbox': ground truth bounding box [x, y, width, height]
                - 'ref_id': reference id
                - 'image_id': COCO image id
        """
        # Load from RefCOCO
        sample = self.loader[idx]
        
        # Apply all degradations
        images = self.degradation.apply_all(sample['image'])
        
        return {
            'images': images,
            'expression': sample['expression'],
            'all_expressions': sample['all_expressions'],
            'bbox': sample['bbox'],
            'ref_id': sample['ref_id'],
            'image_id': sample['image_id']
        }
    
    def get_degraded_image(self, idx, degradation_type):
        """
        Get a single degraded image
        
        Args:
            idx: sample index
            degradation_type: 'clean', 'fog', 'smoke', or 'thermal'
        """
        sample = self.loader[idx]
        degraded = self.degradation.apply_single(sample['image'], degradation_type)
        
        return {
            'image': degraded,
            'expression': sample['expression'],
            'bbox': sample['bbox']
        }
    
    def visualize_sample(self, idx, save_path=None, show=True):
        """
        Visualize one sample with all degradations
        
        Args:
            idx: sample index
            save_path: optional path to save the figure
            show: whether to display the figure
        """
        sample = self[idx]
        images = sample['images']
        expression = sample['expression']
        bbox = sample['bbox']
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        for ax, (name, img) in zip(axes, images.items()):
            # Draw bounding box
            img_vis = img.copy()
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Convert BGR to RGB for display
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img_vis)
            ax.set_title(name.capitalize(), fontsize=14)
            ax.axis('off')
        
        # Truncate long expressions
        display_expr = expression if len(expression) < 60 else expression[:57] + "..."
        plt.suptitle(f'Expression: "{display_expr}"', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_severity_comparison(self, idx, save_path=None, show=True):
        """
        Show how different severity levels affect the image
        
        Args:
            idx: sample index
        """
        sample = self.loader[idx]
        image = sample['image']
        expression = sample['expression']
        
        severities = [0.2, 0.4, 0.6, 0.8]
        degradation_types = ['fog', 'smoke', 'thermal']
        
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        
        for row, deg_type in enumerate(degradation_types):
            # Show clean image in first column
            axes[row, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[row, 0].set_title('Clean' if row == 0 else '')
            axes[row, 0].set_ylabel(deg_type.capitalize(), fontsize=12)
            axes[row, 0].axis('off')
            
            # Show different severities
            for col, sev in enumerate(severities):
                pipeline = DegradationPipeline(severity=sev)
                degraded = pipeline.apply_single(image, deg_type)
                
                axes[row, col+1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
                axes[row, col+1].set_title(f'Severity: {sev}' if row == 0 else '')
                axes[row, col+1].axis('off')
        
        plt.suptitle(f'Expression: "{expression}"', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


# ==================== MAIN USAGE ====================
if __name__ == "__main__":
    
    # ===== CONFIGURE THESE PATHS =====
    REFCOCO_PATH = 'rq1_datasets/refcoco/'          # folder with refs(unc).p and instances.json
    COCO_IMAGES_PATH = 'rq1_datasets/coco/images/train2014/'  # folder with COCO images
    
    # Check if paths exist
    if not os.path.exists(REFCOCO_PATH):
        print(f"ERROR: RefCOCO path not found: {REFCOCO_PATH}")
        print("Please update REFCOCO_PATH to point to your refcoco annotations folder")
        exit(1)
    
    if not os.path.exists(COCO_IMAGES_PATH):
        print(f"ERROR: COCO images path not found: {COCO_IMAGES_PATH}")
        print("Please update COCO_IMAGES_PATH to point to your COCO train2014 folder")
        exit(1)
    
    # Initialize dataset
    print("Loading dataset...")
    dataset = RefCOCODegradationDataset(
        refcoco_path=REFCOCO_PATH,
        coco_images_path=COCO_IMAGES_PATH,
        split='train',
        severity=0.5
    )
    
    print(f"\nDataset size: {len(dataset)} samples")
    
    # Visualize a few samples
    print("\nGenerating visualizations...")
    os.makedirs('outputs', exist_ok=True)
    
    for i in range(3):
        print(f"  Processing sample {i}...")
        dataset.visualize_sample(i, save_path=f'outputs/sample_{i}.png', show=False)
    
    # Show severity comparison for one sample
    print("\nGenerating severity comparison...")
    dataset.visualize_severity_comparison(0, save_path='outputs/severity_comparison.png', show=False)
    
    # Get one sample programmatically
    print("\n" + "="*50)
    print("Sample data structure:")
    print("="*50)
    sample = dataset[0]
    print(f"Expression: {sample['expression']}")
    print(f"All expressions: {sample['all_expressions']}")
    print(f"Bbox: {sample['bbox']}")
    print(f"Available images: {list(sample['images'].keys())}")
    print(f"Image shape: {sample['images']['clean'].shape}")
    
    print("\n✓ Done! Check the 'outputs' folder for visualizations.")