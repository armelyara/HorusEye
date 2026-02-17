"""
Visual Grounding Model using OWL-ViT
Google's Open-Vocabulary Object Detection model
Optimized for RefCOCO expressions
"""

import torch
import numpy as np
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import re


class GroundingModel:
    """OWL-ViT based visual grounding model"""
    
    def __init__(self, device=None):
        """
        Initialize the grounding model
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load OWL-ViT model
        print("Loading OWL-ViT model...")
        self.processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def simplify_expression(self, expression):
        """
        Simplify RefCOCO expression for better OWL-ViT matching
        
        OWL-ViT works better with simple noun phrases.
        
        Args:
            expression: full referring expression
            
        Returns:
            list of simplified expressions to try
        """
        expression = expression.lower().strip()
        
        # Common object words to extract
        object_words = [
            'person', 'man', 'woman', 'boy', 'girl', 'child', 'baby', 'lady', 'guy',
            'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe',
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'airplane', 'boat',
            'chair', 'couch', 'bed', 'table', 'desk', 'bench',
            'bottle', 'cup', 'bowl', 'plate', 'fork', 'knife', 'spoon',
            'tv', 'laptop', 'phone', 'keyboard', 'mouse',
            'book', 'clock', 'vase', 'plant', 'flower',
            'pizza', 'cake', 'sandwich', 'apple', 'banana', 'orange',
            'ball', 'frisbee', 'skateboard', 'surfboard', 'tennis racket',
            'umbrella', 'handbag', 'backpack', 'suitcase'
        ]
        
        simplified = []
        
        # Try to find object words in expression
        for word in object_words:
            if word in expression:
                simplified.append(word)
        
        # Also try the original expression
        simplified.append(expression)
        
        # Try first few words (often contains the main object)
        words = expression.split()
        if len(words) >= 2:
            # "the lady" -> "lady"
            if words[0] in ['the', 'a', 'an']:
                simplified.insert(0, words[1])
            else:
                simplified.insert(0, words[0])
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for item in simplified:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        
        return unique[:3]  # Return top 3 candidates
    
    def predict(self, image, expression, threshold=0.01, use_simplification=True):
        """
        Predict bounding box for a text expression
        
        Args:
            image: numpy array (H, W, 3) in BGR format (OpenCV) or RGB
            expression: text query like "the woman in blue"
            threshold: confidence threshold
            use_simplification: whether to try simplified expressions
            
        Returns:
            dict with:
                - 'bbox': [x, y, width, height] or None if no detection
                - 'confidence': float
                - 'all_boxes': list of all detected boxes
                - 'matched_expression': which expression variant matched
        """
        # Convert BGR to RGB if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image[:, :, ::-1].copy()
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(image)
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        img_width, img_height = pil_image.size
        
        # Get expressions to try
        if use_simplification:
            expressions_to_try = self.simplify_expression(expression)
        else:
            expressions_to_try = [expression]
        
        best_result = {
            'bbox': None,
            'confidence': 0.0,
            'all_boxes': [],
            'matched_expression': None
        }
        
        # Try each expression variant
        for expr in expressions_to_try:
            # OWL-ViT expects text as a list of lists
            texts = [[expr]]
            
            try:
                inputs = self.processor(
                    text=texts,
                    images=pil_image,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=16,  # OWL-ViT's max token length
                    truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception as e:
                print(f"Warning: Failed to process expression '{expr}': {e}")
                continue
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([[img_height, img_width]]).to(self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )[0]
            
            boxes = results['boxes'].cpu().numpy()
            scores = results['scores'].cpu().numpy()
            
            if len(boxes) > 0:
                best_idx = np.argmax(scores)
                if scores[best_idx] > best_result['confidence']:
                    x1, y1, x2, y2 = boxes[best_idx]
                    best_result = {
                        'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        'confidence': float(scores[best_idx]),
                        'all_boxes': [
                            {
                                'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                                'confidence': float(s)
                            }
                            for b, s in zip(boxes, scores)
                        ],
                        'matched_expression': expr
                    }
        
        return best_result
    
    def predict_batch(self, images, expressions, threshold=0.01):
        """Predict bounding boxes for multiple image-expression pairs"""
        results = []
        for image, expression in zip(images, expressions):
            result = self.predict(image, expression, threshold)
            results.append(result)
        return results


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes
    
    Args:
        box1: [x, y, width, height]
        box2: [x, y, width, height]
        
    Returns:
        IoU score (0 to 1)
    """
    if box1 is None or box2 is None:
        return 0.0
    
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
    
    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


# ==================== TEST ====================
if __name__ == "__main__":
    import cv2
    import os
    import sys
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    model = GroundingModel()
    
    print("\n" + "="*60)
    print("Testing OWL-ViT Grounding Model (with expression simplification)")
    print("="*60)
    
    try:
        from degradation_pipeline import RefCOCODegradationDataset
        
        REFCOCO_PATH = 'rq1_datasets/refcoco/refer/data/refcoco'
        COCO_IMAGES_PATH = 'rq1_datasets/coco/images/train2014/'
        
        print("\nLoading RefCOCO dataset...")
        dataset = RefCOCODegradationDataset(
            refcoco_path=REFCOCO_PATH,
            coco_images_path=COCO_IMAGES_PATH,
            split='train',
            severity=0.5
        )
        
        print("\n" + "-"*60)
        print("Testing on RefCOCO samples (Clean vs Degraded):")
        print("-"*60)
        
        for i in range(5):
            sample = dataset[i]
            expression = sample['expression']
            gt_bbox = sample['bbox']
            
            print(f"\nSample {i}: '{expression}'")
            print(f"  GT Bbox: {[round(x, 1) for x in gt_bbox]}")
            
            # Test on each condition
            for condition in ['clean', 'fog', 'smoke', 'thermal']:
                image = sample['images'][condition]
                result = model.predict(image, expression, threshold=0.01)
                
                if result['bbox']:
                    iou = calculate_iou(result['bbox'], gt_bbox)
                    matched = result['matched_expression']
                    print(f"  {condition:8s}: IoU={iou:.3f}, Conf={result['confidence']:.3f}, Matched='{matched}'")
                else:
                    print(f"  {condition:8s}: No detection")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✓ Testing complete!")
    print("="*60)