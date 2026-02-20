"""
RefCOCO Degradation Pipeline
Applies fog, smoke, and thermal effects to clean images
No external refer package needed - standalone version
No imgaug dependency - uses physics-based methods only
"""

import os
import cv2
import json
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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
    """Apply various degradations to images using physics-based methods"""
    
    def __init__(self, severity=0.5):
        """
        Args:
            severity: float 0-1, how strong the degradation is
        """
        self.severity = np.clip(severity, 0.0, 1.0)
    
    # ==================== FOG ====================
    def add_fog(self, image, depth_map=None):
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
        if depth_map is None:
            # Distance increases from top to bottom (typical outdoor scene)
            depth_map = np.tile(
                np.linspace(1.0, 0.0, h).reshape(h, 1),
                (1, w)
            )
            # Add random variation for realism
            noise = cv2.GaussianBlur(
                np.random.rand(h, w).astype(np.float32) * 0.15,
                (51, 51), 0
            )
            depth_map = np.clip(depth_map + noise, 0, 1)
        
        # Atmospheric light (white/gray fog)
        A = 0.95
        
        # Transmission map: t = exp(-beta * d)
        beta = self.severity * 2.0  # fog density
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
        
        # Generate smoke texture
        smoke = self._generate_smoke_texture(h, w)
        
        # Smoke color (gray-white with slight blue tint) in BGR
        smoke_color = np.array([0.9, 0.9, 0.92])
        smoke_rgb = np.stack([smoke] * 3, axis=-1) * smoke_color
        
        # Blend based on severity
        alpha = smoke * self.severity * 1.2
        alpha = np.stack([alpha]*3, axis=-1)

        smoke_layer = np.ones_like(image_float) * smoke_color
        
        smoky = image_float * (1 - alpha) + smoke_layer * alpha
        return np.clip(smoky * 255, 0, 255).astype(np.uint8)
        
        
    
    def _generate_smoke_texture(self, h, w):
        """Generate procedural smoke using multi-scale noise"""
        smoke = np.zeros((h, w), dtype=np.float32)
        
        # Add multiple scales of noise
        for scale in [4, 8, 16, 32, 64]:
            noise_h = max(h // scale + 1, 2)
            noise_w = max(w // scale + 1, 2)
            noise = np.random.rand(noise_h, noise_w).astype(np.float32)
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
            smoke += noise * (scale / 64.0)
        
        # Normalize
        smoke = (smoke - smoke.min()) / (smoke.max() - smoke.min() + 1e-8)
        
        # Create patches
        smoke = np.clip(smoke * 2.0, 0, 1)
        
        # Smooth
        smoke = np.power(smoke, 2.5) #increase contrast
        smoke = cv2.GaussianBlur(smoke, (41, 41), 0) #increase blur
        
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
        # Get dimensions
        h, w = image.shape[:2]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Simulate heat signature (invert so bright = hot)
        gray_thermal = cv2.equalizeHist(gray)
        
        # Apply thermal colormap
        thermal = cv2.applyColorMap(gray_thermal, cv2.COLORMAP_JET)
        
        # Reduce detail (thermal has lower resolution)
        blur_amount = int(3 * self.severity) * 2 + 1
        if blur_amount > 1:
            thermal = cv2.GaussianBlur(thermal, (blur_amount, blur_amount), 0)
        
        # pixelisation effect
        if self.severity > 0.2:
            f = 4 # Facteur de réduction
            small = cv2.resize(thermal, (w//f, h//f), interpolation=cv2.INTER_NEAREST)
            thermal = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Add sensor noise
        noise_level = 15 * self.severity
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, thermal.shape).astype(np.float32)
            thermal = np.clip(thermal.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return thermal
    
    # ==================== APPLY ALL ====================
    def apply_all(self, image):
        """Apply all degradations"""
        return {
            'clean': image.copy(),
            'fog': self.add_fog(image),
            'smoke': self.add_smoke(image),
            'thermal': self.add_thermal(image)
        }
    
    def apply_single(self, image, degradation_type):
        """Apply a single degradation type"""
        if degradation_type == 'fog':
            return self.add_fog(image)
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
        """Get one sample with all degradations"""
        sample = self.loader[idx]
        images = self.degradation.apply_all(sample['image'])
        
        return {
            'images': images,
            'expression': sample['expression'],
            'all_expressions': sample['all_expressions'],
            'bbox': sample['bbox'],
            'ref_id': sample['ref_id'],
            'image_id': sample['image_id']
        }
    
    def visualize_sample(self, idx, save_path=None, show=True):
        """Visualize one sample with all degradations"""
        sample = self[idx]
        images = sample['images']
        expression = sample['expression']
        bbox = sample['bbox']
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        for ax, (name, img) in zip(axes, images.items()):
            img_vis = img.copy()
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img_vis)
            ax.set_title(name.capitalize(), fontsize=14)
            ax.axis('off')
        
        display_expr = expression if len(expression) < 60 else expression[:57] + "..."
        plt.suptitle(f'Expression: "{display_expr}"', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_degraded_dataset(self, output_root, max_samples=None):
        """
        Save all degraded images to disk
        
        Creates:
        output_root/
        ├── clean/
        ├── fog/
        ├── smoke/
        ├── thermal/
        └── annotations.json
        """
        degradation_types = ['clean', 'fog', 'smoke', 'thermal']
        
        for deg_type in degradation_types:
            os.makedirs(os.path.join(output_root, deg_type), exist_ok=True)
        
        n_samples = len(self) if max_samples is None else min(max_samples, len(self))
        annotations = []
        
        print(f"Saving {n_samples} samples to {output_root}")
        
        for idx in range(n_samples):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{n_samples}")
            
            sample = self[idx]
            images = sample['images']
            filename = f"{idx:06d}.jpg"
            
            for deg_type in degradation_types:
                save_path = os.path.join(output_root, deg_type, filename)
                cv2.imwrite(save_path, images[deg_type])
            
            annotations.append({
                'index': idx,
                'filename': filename,
                'expression': sample['expression'],
                'all_expressions': sample['all_expressions'],
                'bbox': sample['bbox'],
                'ref_id': sample['ref_id'],
                'image_id': sample['image_id']
            })
        
        annotations_path = os.path.join(output_root, 'annotations.json')
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✓ Done! Saved {n_samples} samples")


# ==================== MAIN ====================
if __name__ == "__main__":
    
    # ===== UPDATE THESE PATHS =====
    REFCOCO_PATH = 'rq1_datasets/refcoco/refer/data/refcoco'
    COCO_IMAGES_PATH = 'rq1_datasets/coco/images/train2014/'
    
    # Check paths
    if not os.path.exists(REFCOCO_PATH):
        print(f"ERROR: RefCOCO not found at {REFCOCO_PATH}")
        print("Update REFCOCO_PATH in the script")
        exit(1)
    
    if not os.path.exists(COCO_IMAGES_PATH):
        print(f"ERROR: COCO images not found at {COCO_IMAGES_PATH}")
        print("Update COCO_IMAGES_PATH in the script")
        exit(1)
    
    # Initialize
    print("Loading dataset...")
    dataset = RefCOCODegradationDataset(
        refcoco_path=REFCOCO_PATH,
        coco_images_path=COCO_IMAGES_PATH,
        split='train',
        severity=0.5
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create output folder
    os.makedirs('outputs', exist_ok=True)
    
    # Visualize samples
    print("\nGenerating visualizations...")
    for i in range(3):
        dataset.visualize_sample(i, save_path=f'outputs/sample_{i}.png', show=False)
    
    # Print sample info
    sample = dataset[0]
    print(f"\nSample info:")
    print(f"  Expression: {sample['expression']}")
    print(f"  Bbox: {sample['bbox']}")
    print(f"  Image keys: {list(sample['images'].keys())}")
    
    print("\n✓ Done! Check 'outputs' folder")