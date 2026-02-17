"""
RefCOCO Degradation Pipeline
Applies fog, smoke, and thermal effects to clean images
"""

import os
import cv2
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from refer import REFER


class DegradationPipeline:
    """Apply various degradations to images"""
    
    def __init__(self, severity=0.5):
        """
        Args:
            severity: float 0-1, how strong the degradation is
        """
        self.severity = severity
    
    # ==================== FOG ====================
    def add_fog(self, image):
        """
        Apply synthetic fog using imgaug
        
        Args:
            image: numpy array (H, W, 3) BGR format
        Returns:
            foggy image
        """
        # imgaug fog augmenter
        fog_augmenter = iaa.Fog(
            severity=(self.severity, self.severity)
        )
        foggy = fog_augmenter(image=image)
        return foggy
    
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
        
        # If no depth map, create synthetic one (assumes distance increases downward)
        if depth_map is None:
            depth_map = np.tile(
                np.linspace(0, 1, h).reshape(h, 1),
                (1, w)
            )
        
        # Atmospheric light (white fog)
        A = 0.8
        
        # Transmission map: t = exp(-beta * d)
        beta = self.severity * 3  # fog density
        t = np.exp(-beta * depth_map)
        t = np.stack([t] * 3, axis=-1)  # expand to 3 channels
        
        # Apply atmospheric scattering
        foggy = image_float * t + A * (1 - t)
        foggy = np.clip(foggy * 255, 0, 255).astype(np.uint8)
        
        return foggy
    
    # ==================== SMOKE ====================
    def add_smoke(self, image):
        """
        Apply synthetic smoke using Perlin noise
        
        Args:
            image: numpy array (H, W, 3)
        Returns:
            smoky image
        """
        h, w = image.shape[:2]
        image_float = image.astype(np.float32) / 255.0
        
        # Generate smoke texture using multiple noise layers
        smoke = self._generate_smoke_texture(h, w)
        
        # Smoke color (gray-white)
        smoke_color = np.array([0.8, 0.8, 0.85])
        smoke_rgb = np.stack([smoke] * 3, axis=-1) * smoke_color
        
        # Blend based on severity
        alpha = smoke * self.severity
        alpha = np.stack([alpha] * 3, axis=-1)
        
        smoky = image_float * (1 - alpha) + smoke_rgb * alpha
        smoky = np.clip(smoky * 255, 0, 255).astype(np.uint8)
        
        return smoky
    
    def _generate_smoke_texture(self, h, w):
        """Generate procedural smoke using multi-scale noise"""
        smoke = np.zeros((h, w))
        
        # Add multiple scales of noise
        for scale in [4, 8, 16, 32]:
            noise = np.random.rand(h // scale + 1, w // scale + 1)
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
            smoke += noise / scale
        
        # Normalize
        smoke = (smoke - smoke.min()) / (smoke.max() - smoke.min())
        
        # Apply threshold for more realistic smoke patches
        smoke = np.clip(smoke * 1.5 - 0.25, 0, 1)
        
        return smoke
    
    # ==================== THERMAL ====================
    def add_thermal(self, image):
        """
        Simulate thermal/infrared appearance
        
        Args:
            image: numpy array (H, W, 3)
        Returns:
            thermal-style image
        """
        # Convert to grayscale (approximates heat signature)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thermal colormap
        # COLORMAP_INFERNO or COLORMAP_HOT simulate thermal cameras
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        
        # Reduce detail (thermal has lower resolution)
        blur_amount = int(5 * self.severity) * 2 + 1  # must be odd
        thermal = cv2.GaussianBlur(thermal, (blur_amount, blur_amount), 0)
        
        # Add slight noise (sensor noise)
        noise = np.random.normal(0, 10 * self.severity, thermal.shape)
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
            'clean': image,
            'fog': self.add_fog_physics(image),
            'smoke': self.add_smoke(image),
            'thermal': self.add_thermal(image)
        }


class RefCOCODegradationDataset:
    """Load RefCOCO and apply degradations"""
    
    def __init__(self, data_root, dataset='refcoco', split='train', severity=0.5):
        """
        Args:
            data_root: path to data folder
            dataset: 'refcoco', 'refcoco+', or 'refcocog'
            split: 'train', 'val', or 'test'
            severity: degradation severity 0-1
        """
        self.refer = REFER(data_root, dataset, splitBy='unc')
        self.ref_ids = self.refer.getRefIds(split=split)
        self.degradation = DegradationPipeline(severity=severity)
        self.image_dir = os.path.join(data_root, 'coco/images/train2014')
    
    def __len__(self):
        return len(self.ref_ids)
    
    def __getitem__(self, idx):
        """
        Get one sample with all degradations
        
        Returns:
            dict with:
                - 'images': dict of clean/fog/smoke/thermal images
                - 'expression': referring expression text
                - 'bbox': ground truth bounding box
                - 'ref_id': reference id
        """
        ref_id = self.ref_ids[idx]
        ref = self.refer.Refs[ref_id]
        
        # Load image
        image_id = ref['image_id']
        image_info = self.refer.Imgs[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        
        # Get referring expression
        expression = ref['sentences'][0]['sent']
        
        # Get bounding box
        ann = self.refer.refToAnn[ref_id]
        bbox = ann['bbox']  # [x, y, width, height]
        
        # Apply all degradations
        images = self.degradation.apply_all(image)
        
        return {
            'images': images,
            'expression': expression,
            'bbox': bbox,
            'ref_id': ref_id
        }
    
    def visualize_sample(self, idx, save_path=None):
        """Visualize one sample with all degradations"""
        sample = self[idx]
        images = sample['images']
        expression = sample['expression']
        bbox = sample['bbox']
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ['Clean', 'Fog', 'Smoke', 'Thermal']
        
        for ax, (name, img) in zip(axes, images.items()):
            # Draw bounding box
            img_vis = img.copy()
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Convert BGR to RGB for display
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img_vis)
            ax.set_title(name)
            ax.axis('off')
        
        plt.suptitle(f'Expression: "{expression}"', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


# ==================== MAIN USAGE ====================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Initialize dataset
    dataset = RefCOCODegradationDataset(
        data_root='datasets/',
        dataset='refcoco',
        split='train',
        severity=0.5
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Visualize a few samples
    for i in range(3):
        dataset.visualize_sample(i, save_path=f'sample_{i}.png')
    
    # Get one sample programmatically
    sample = dataset[0]
    print(f"Expression: {sample['expression']}")
    print(f"Bbox: {sample['bbox']}")
    print(f"Available images: {list(sample['images'].keys())}")
