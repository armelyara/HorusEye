"""
Visual Grounding Model using MDETR
Predicts bounding boxes from text expressions
"""

import torch
import numpy as np
from PIL import Image
from transformers import MdetrForObjectDetection, MdetrImageProcessor


class GroundingModel:
    """MDETR-based visual grounding model"""
    
    def __init__(self, device=None):
        """
        Initialize the grounding model
        
        Args:
            device: 'cuda', 'mps', or 'cpu' (auto-detected if None)
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'  # Mac M1/M2
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load MDETR model (pre-trained on RefCOCO)
        print("Loading MDETR model...")
        self.processor = MdetrImageProcessor.from_pretrained(
            "kamath-shreyas/mdetr_finetuned_refcoco"
        )
        self.model = MdetrForObjectDetection.from_pretrained(
            "kamath-shreyas/mdetr_finetuned_refcoco"
        )
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict(self, image, expression, threshold=0.5):
        """
        Predict bounding box for a text expression
        
        Args:
            image: numpy array (H, W, 3) in BGR format (OpenCV) or RGB
            expression: text query like "the woman in blue"
            threshold: confidence threshold
            
        Returns:
            dict with:
                - 'bbox': [x, y, width, height] or None if no detection
                - 'confidence': float
                - 'all_boxes': list of all detected boxes
        """
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR from OpenCV, convert to RGB
                image_rgb = image[:, :, ::-1]
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        # Process inputs
        inputs = self.processor(
            images=pil_image,
            text=expression,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes
        )[0]
        
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        
        if len(boxes) == 0:
            return {
                'bbox': None,
                'confidence': 0.0,
                'all_boxes': []
            }
        
        # Get the best box (highest confidence)
        best_idx = np.argmax(scores)
        best_box = boxes[best_idx]
        
        # Convert from [x1, y1, x2, y2] to [x, y, width, height]
        x1, y1, x2, y2 = best_box
        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        
        # All boxes in same format
        all_boxes = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            all_boxes.append({
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'confidence': float(score)
            })
        
        return {
            'bbox': bbox,
            'confidence': float(scores[best_idx]),
            'all_boxes': all_boxes
        }
    
    def predict_batch(self, images, expressions, threshold=0.5):
        """
        Predict bounding boxes for multiple image-expression pairs
        
        Args:
            images: list of numpy arrays
            expressions: list of text queries
            threshold: confidence threshold
            
        Returns:
            list of prediction dicts
        """
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
    
    # Convert to [x1, y1, x2, y2]
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
    
    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


# ==================== TEST ====================
if __name__ == "__main__":
    import cv2
    
    # Initialize model
    model = GroundingModel()
    
    # Test with a sample image
    print("\nTesting with a sample...")
    
    # Create a simple test (or use your own image)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_expression = "a person"
    
    result = model.predict(test_image, test_expression)
    print(f"Prediction: {result}")
    
    print("\n✓ Model is working!")