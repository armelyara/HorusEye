"""
Evaluate RefCOCO-Degraded with Gemini
"""

import os
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# Import Gemini
try:
    from google import genai
    USE_NEW_API = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_API = False


def calculate_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
    
    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1, area2 = box1[2] * box1[3], box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def extract_bbox_normalized(text, img_w, img_h):
    """
    Converts Gemini's [ymin, xmin, ymax, xmax] (0-1000) 
    back to RefCOCO's [x, y, w, h] (pixels).
    """
    if not text: return None
    # Look for Gemini's typical bracketed output: [ymin, xmin, ymax, xmax]
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(pattern, text)
    
    if match:
        try:
            ymin, xmin, ymax, xmax = [int(v) for v in match.groups()]
            
            # 1. Scale normalized 1000-scale back to pixels
            px_xmin = (xmin / 1000.0) * img_w
            px_ymin = (ymin / 1000.0) * img_h
            px_xmax = (xmax / 1000.0) * img_w
            px_ymax = (ymax / 1000.0) * img_h
            
            # 2. Convert to RefCOCO format [x, y, width, height]
            return [px_xmin, px_ymin, px_xmax - px_xmin, px_ymax - px_ymin]
        except:
            return None
    return None

class GeminiEvaluator:
    """Gemini evaluator"""
    
    def __init__(self, api_key=None, requests_per_minute=15):
        """
        Args:
            api_key: Google API key
            requests_per_minute: Rate limit (free tier = ~60, use 15 to be safe)
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Set GOOGLE_API_KEY environment variable")
        
        # Rate limiting (thread-safe)
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute  # Seconds between requests
        self.last_request_time = 0
        self._rate_lock = threading.Lock()  # Protects last_request_time across threads
        
        # Initialize API
        if USE_NEW_API:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = 'gemini-2.0-flash'
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Stats
        self.total_requests = 0
        self.total_cost = 0.0
        
        print(f"Gemini initialized (rate limit: {requests_per_minute} req/min)")
    
    def _wait_for_rate_limit(self):
        """Thread-safe rate limiter: serialises all threads through the delay."""
        with self._rate_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            self.last_request_time = time.time()
    
    def evaluate(self, image_path, expression, max_retries=3):
        """Evaluate grounding with rate limiting"""
        
        prompt = (f"Identify the object described as '{expression}'. "
          f"Even if the image is degraded (fog, smoke, or thermal), "
          f"provide your best estimate of its bounding box in [ymin, xmin, ymax, xmax] format. "
          f"Use a normalized scale of 0-1000. Output ONLY the list.")

        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                img = Image.open(image_path)
                w, h = img.size
                
                if USE_NEW_API:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[prompt, img]
                    )
                    text = response.text
                else:
                    response = self.model.generate_content([prompt, img])
                    text = response.text.strip()
                
                return {
                    'response': text,
                    'bbox': extract_bbox_normalized(text, w, h),
                    'success': True
                }
                
            except Exception as e:
                error = str(e).lower()
                
                if 'quota' in error or 'rate' in error or '429' in error:
                    wait = 60 * (attempt + 1)
                    print(f"\n  Rate limited. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        return {'response': str(e), 'bbox': None, 'success': False}
        
        return {'response': 'Max retries', 'bbox': None, 'success': False}


def _load_checkpoint(results_dir, conditions):
    """Resume from the latest checkpoint if it exists. Returns (results, start_idx)."""
    checkpoint_path = os.path.join(results_dir, 'checkpoint_latest.json')
    if not os.path.exists(checkpoint_path):
        return {c: {'ious': [], 'detected': []} for c in conditions}, 0
    with open(checkpoint_path) as f:
        ckpt = json.load(f)
    print(f"  Resuming from checkpoint at sample {ckpt['n_done']}")
    return ckpt['results'], ckpt['n_done']


def run_evaluation(dataset_path, num_samples=None, conditions=None,
                  requests_per_minute=15, max_workers=None):
    """
    Run evaluation with rate limiting and parallel condition evaluation.

    Args:
        dataset_path: Path to dataset
        num_samples: Number of samples (None = all)
        conditions: Conditions to test (None = main conditions only)
        requests_per_minute: API rate limit
        max_workers: Threads for parallel conditions (default = nb of conditions)
    """

    # Load data
    print("Loading dataset...")
    with open(os.path.join(dataset_path, 'annotations', 'annotations.json')) as f:
        data = json.load(f)

    annotations = data['annotations']
    all_conditions = data['info']['conditions']

    # Set defaults
    if num_samples is None:
        num_samples = len(annotations)
    annotations = annotations[:num_samples]

    if conditions is None:
        conditions = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
        conditions = [c for c in conditions if c in all_conditions]

    if max_workers is None:
        max_workers = len(conditions)  # One thread per condition

    total_calls = num_samples * len(conditions)
    # With parallel conditions each sample only occupies 1 rate-limit slot per
    # condition, but they overlap in time → effective throughput ≈ RPM / 1
    estimated_time = total_calls / requests_per_minute  # minutes
    estimated_cost = total_calls * 0.00025

    print(f"\nConfiguration:")
    print(f"  Samples        : {num_samples}")
    print(f"  Conditions     : {conditions}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Rate limit     : {requests_per_minute} req/min")
    print(f"  Parallel workers: {max_workers}")
    print(f"  Estimated time : {estimated_time:.0f} minutes")
    print(f"  Estimated cost : ${estimated_cost:.2f}")

    input("\nPress Enter to start (or Ctrl+C to cancel)...")

    # Initialize
    evaluator = GeminiEvaluator(requests_per_minute=requests_per_minute)
    results_dir = os.path.join(dataset_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Resume from checkpoint if available
    results, start_idx = _load_checkpoint(results_dir, conditions)
    annotations = annotations[start_idx:]   # Skip already-done samples

    start_time = time.time()
    errors_lock = threading.Lock()
    errors = 0

    print(f"\n{'='*60}")
    print(f"EVALUATION STARTED (from sample {start_idx})")
    print(f"{'='*60}\n")

    pbar = tqdm(total=len(annotations) * len(conditions), desc="Evaluating")

    def eval_condition(ann, condition):
        """Evaluate one (sample, condition) pair — runs in a thread."""
        image_path = os.path.join(dataset_path, 'images', condition, ann['filename'])
        if not os.path.exists(image_path):
            return condition, 0.0, False, False   # iou, detected, success
        result = evaluator.evaluate(image_path, ann['expression'])
        iou = calculate_iou(result['bbox'], ann['bbox'])
        return condition, iou, result['bbox'] is not None, result['success']

    for idx, ann in enumerate(annotations):
        global_idx = start_idx + idx

        # Submit all conditions for this sample in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(eval_condition, ann, cond): cond
                       for cond in conditions}
            for fut in as_completed(futures):
                cond, iou, detected, success = fut.result()
                results[cond]['ious'].append(iou)
                results[cond]['detected'].append(detected)
                if not success:
                    with errors_lock:
                        errors += 1
                pbar.update(1)
                pbar.set_postfix({'errors': errors})

        # Checkpoint every 10 samples (more frequent since runs are faster)
        if (global_idx + 1) % 10 == 0:
            save_checkpoint(results, conditions, results_dir,
                            global_idx + 1, evaluator.total_cost)
    
    pbar.close()
    
    # Calculate summary
    elapsed = time.time() - start_time
    summary = calculate_summary(results, conditions)
    
    # Print results
    print_results(summary, conditions, evaluator.total_cost, elapsed, num_samples, errors)
    
    # Save final results
    save_results(results, summary, conditions, results_dir, evaluator.total_cost, elapsed, num_samples)
    
    # Plot
    plot_results(summary, conditions, results_dir)
    print("\n--- Quick Performance Summary (Prec@0.5) ---")
    for cond in conditions:
        if cond in results and results[cond]['ious']:
            ious = np.array(results[cond]['ious'])
            prec_05 = np.mean(ious >= 0.5) * 100
            print(f"Condition {cond:12}: Prec@0.5 = {prec_05:>6.2f}%")
    
    return results, summary


def calculate_summary(results, conditions):
    """Calculate statistics"""
    summary = {}
    
    clean_mean = np.mean(results['clean']['ious']) if results['clean']['ious'] else 0
    
    for cond in conditions:
        ious = results[cond]['ious']
        if not ious:
            continue
        
        mean_iou = np.mean(ious)
        drop = 0 if cond == 'clean' else (clean_mean - mean_iou) / clean_mean * 100 if clean_mean > 0 else 0
        
        summary[cond] = {
            'mean_iou': float(mean_iou),
            'std_iou': float(np.std(ious)),
            'acc_0.3': float(np.mean([i >= 0.3 for i in ious])),
            'acc_0.5': float(np.mean([i >= 0.5 for i in ious])),
            'acc_0.7': float(np.mean([i >= 0.7 for i in ious])),
            'detection_rate': float(np.mean(results[cond]['detected'])),
            'drop': float(drop),
            'n': len(ious)
        }
    
    return summary


def print_results(summary, conditions, cost, elapsed, n_samples, errors):
    """Print results table"""
    
    print("\n" + "=" * 70)
    print("GEMINI EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n{'Condition':<15} {'Mean IoU':<10} {'Acc@0.5':<10} {'Det Rate':<10} {'Drop':<10}")
    print("-" * 60)
    
    for cond in conditions:
        if cond not in summary:
            continue
        s = summary[cond]
        drop_str = "—" if cond == 'clean' else f"-{s['drop']:.1f}%"
        print(f"{cond:<15} {s['mean_iou']:<10.3f} {s['acc_0.5']:<10.3f} {s['detection_rate']:<10.3f} {drop_str:<10}")
    
    print("-" * 60)
    print(f"\nStatistics:")
    print(f"  Samples: {n_samples}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Cost: ${cost:.2f}")
    print(f"  Errors: {errors}")
    print("=" * 70)


def save_checkpoint(results, conditions, results_dir, n, cost):
    """Save intermediate results and update the resume checkpoint."""
    summary = calculate_summary(results, conditions)
    # Numbered snapshot (for history)
    path = os.path.join(results_dir, f'checkpoint_{n}.json')
    with open(path, 'w') as f:
        json.dump({'n': n, 'cost': cost, 'summary': summary}, f, indent=2)
    # Latest checkpoint for resume (includes full raw results)
    latest_path = os.path.join(results_dir, 'checkpoint_latest.json')
    with open(latest_path, 'w') as f:
        json.dump({'n_done': n, 'cost': cost, 'results': results}, f, indent=2)


def save_results(results, summary, conditions, results_dir, cost, elapsed, n_samples):
    """Save final results"""
    path = os.path.join(results_dir, 'gemini_results.json')
    with open(path, 'w') as f:
        json.dump({
            'metadata': {
                'model': 'gemini-2.0-flash',
                'timestamp': datetime.now().isoformat(),
                'samples': n_samples,
                'conditions': conditions,
                'cost': cost,
                'time_seconds': elapsed
            },
            'summary': summary,
            'raw_ious': {c: results[c]['ious'] for c in conditions}
        }, f, indent=2)
    print(f"\nResults saved to {path}")


def plot_results(summary, conditions, results_dir):
    """Create visualization"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'clean': '#27ae60', 'fog_0.5': '#3498db', 'smoke_0.5': '#9b59b6', 'thermal_0.5': '#e74c3c'}

    # Plot 1: Mean IoU
    ax1 = axes[0]
    conds = [c for c in conditions if c in summary]
    color = [colors.get(c, '#95a5a6') for c in conds]  # Utilise un gris moyen si la clé n'existe pas
    means = [summary[c]['mean_iou'] for c in conds]
    stds = [summary[c]['std_iou'] for c in conds]
    
    bars = ax1.bar(range(len(conds)), means, yerr=stds, capsize=5,
                   color=color, alpha=0.8)
    ax1.set_xticks(range(len(conds)))
    ax1.set_xticklabels([c.replace('_0.5', '') for c in conds])
    ax1.set_ylabel('Mean IoU')
    ax1.set_title('Grounding Performance (Gemini)')
    ax1.set_ylim(0, 1)
    
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', fontsize=10)
    
    # Plot 2: Performance drop
    ax2 = axes[1]
    degraded = [c for c in conds if c != 'clean']
    drops = [summary[c]['drop'] for c in degraded]
    
    color_degraded = [colors.get(c, '#95a5a6') for c in degraded]
    bars = ax2.bar(range(len(degraded)), drops,
                   color=color_degraded, alpha=0.8)
    ax2.set_xticks(range(len(degraded)))
    ax2.set_xticklabels([c.replace('_0.5', '') for c in degraded])
    ax2.set_ylabel('Performance Drop (%)')
    ax2.set_title('Degradation Impact')
    
    for bar, drop in zip(bars, drops):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{drop:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    path = os.path.join(results_dir, 'gemini_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {path}")
    plt.show()


# ==================== MAIN ====================

if __name__ == "__main__":
    
    DATASET_PATH = 'datasets/refcoco_degraded_benchmark/'
    
    # Start with small test
    results, summary = run_evaluation(
        dataset_path=DATASET_PATH,
        num_samples=None,          # Start with 50 samples for testing
        conditions=['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5'],  # Main conditions
        requests_per_minute=15   # Conservative rate limit
    )
    
    print("\n✓ Done!")