# SETUP & CONFIGURATION

#!pip install google-genai pillow tqdm matplotlib -q
#!pip install transformers accelerate bitsandbytes qwen-vl-utils -q

import os
import json
import time
import re
import string
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

#from google.colab import drive, userdata
#drive.mount('/content/drive')

# CONFIGURATION
DATASET_PATH = '/HorusEye/horuseye_VLM/refcoco_degraded_benchmark'
RQ1_RESULTS_PATH = os.path.join(DATASET_PATH, 'results', 'rq1_with_bboxes.json')

CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
NUM_SAMPLES = 100
CHECKPOINT_EVERY = 10 

print("✓ Setup complete")
print(f"  DATASET_PATH: {DATASET_PATH}")
print(f"  NUM_SAMPLES: {NUM_SAMPLES}")
print(f"  CHECKPOINT_EVERY: {CHECKPOINT_EVERY}")

# LOAD MODELS

#from google.colab import userdata

# --- Gemini API ---
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

from google import genai
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
print("✓ Gemini API ready")

# --- Qwen2-VL ---
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

print("Loading Qwen2-VL-2B...")
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
qwen_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    trust_remote_code=True
)
print("✓ Qwen2-VL loaded")

# DEFINE HALLUCINATION DETECTION FUNCTIONS

# Objects that are commonly hallucinated
COMMON_HALLUCINATED_OBJECTS = [
    'phone', 'bag', 'hat', 'umbrella', 'book', 'cup', 'bottle',
    'glasses', 'watch', 'backpack', 'purse', 'wallet', 'keys',
    'dog', 'cat', 'bird', 'car', 'bicycle', 'tree', 'flower'
]

# Hallucination indicator phrases
UNCERTAINTY_PHRASES = ['might be', 'could be', 'possibly', 'appears to', 'seems like',
                       'i think', 'maybe', 'probably', 'unclear', 'hard to tell',
                       'cannot determine', 'not sure', 'difficult to see']

OVERCONFIDENCE_PHRASES = ['clearly', 'definitely', 'certainly', 'obviously',
                          'without doubt', 'absolutely', 'undoubtedly']


def save_checkpoint(checkpoint_path, data):
    """
    Save checkpoint with forced disk write.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Write with forced sync to disk
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())

    # Verify
    return os.path.exists(checkpoint_path)


def load_checkpoint(checkpoint_path):
    """
    Load checkpoint if exists.
    """
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            return json.load(f)
    return None


def load_dataset(dataset_path, num_samples=100):
    """Load dataset from RQ1 results."""
    with open(RQ1_RESULTS_PATH) as f:
        rq1_data = json.load(f)

    samples = rq1_data['samples']

    # Filter for person-related expressions
    person_keywords = ['person', 'man', 'woman', 'boy', 'girl', 'guy', 'lady',
                       'people', 'kid', 'child', 'player', 'worker']
    person_samples = [s for s in samples if any(kw in s['expression'].lower() for kw in person_keywords)]

    if len(person_samples) < num_samples:
        remaining = num_samples - len(person_samples)
        other_samples = [s for s in samples if s not in person_samples][:remaining]
        person_samples.extend(other_samples)

    samples = person_samples[:num_samples]

    print(f"✓ Loaded {len(samples)} samples")
    return samples


def analyze_hallucination_text(response_text, expression):
    """
    Analyze text response for hallucination indicators.

    Returns:
        dict with hallucination metrics
    """
    response_lower = response_text.lower()
    expression_lower = expression.lower()

    # Strip punctuation for better word matching
    response_clean = response_lower.translate(str.maketrans('', '', string.punctuation))
    expression_clean = expression_lower.translate(str.maketrans('', '', string.punctuation))

    # Convert to word set for accurate matching
    response_words = set(response_clean.split())
    expression_words = set(expression_clean.split())

    # Count uncertainty phrases (use original for phrase matching)
    uncertainty_count = sum(1 for phrase in UNCERTAINTY_PHRASES if phrase in response_lower)

    # Count fabricated objects (mentioned but not in expression)
    fabrication_count = 0
    fabricated_items = []
    for obj in COMMON_HALLUCINATED_OBJECTS:
        if obj in response_words and obj not in expression_words:
            fabrication_count += 1
            fabricated_items.append(obj)

    # Count overconfidence phrases
    overconfidence_count = sum(1 for phrase in OVERCONFIDENCE_PHRASES if phrase in response_lower)

    # Response length
    word_count = len(response_text.split())

    # Color mentions (potential hallucinations if image is thermal)
    colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'orange', 'purple', 'pink', 'brown']
    color_mentions = sum(1 for color in colors if color in response_words)

    # Hallucination score: higher = more hallucination risk
    hallucination_score = fabrication_count + (overconfidence_count * 0.5) - (uncertainty_count * 0.3)

    return {
        'uncertainty_count': uncertainty_count,
        'fabrication_count': fabrication_count,
        'fabricated_items': fabricated_items,
        'overconfidence_count': overconfidence_count,
        'word_count': word_count,
        'color_mentions': color_mentions,
        'hallucination_score': hallucination_score
    }


# Gemini Tester
class GeminiHallucinationTester:
    def __init__(self, client):
        self.client = client
        self.model_name = 'gemini-2.0-flash'
        self.request_delay = 4.0
        self.last_request = 0

    def _wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request = time.time()

    def get_detailed_description(self, image_path, expression):
        """Ask for detailed description to detect hallucinations."""
        self._wait()

        prompt = f"""Look at this image and describe what you see in detail.

Focus on the object described as: "{expression}"

Describe:
1. The object's appearance (color, size, shape)
2. The object's position in the scene
3. What the object is doing
4. Any other objects or people nearby

Be specific and detailed."""

        image = Image.open(image_path)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            return response.text
        except Exception as e:
            return f"ERROR: {str(e)}"


# Qwen2-VL Tester
class QwenHallucinationTester:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def get_detailed_description(self, image_path, expression):
        """Ask for detailed description to detect hallucinations."""

        prompt = f"""Look at this image and describe what you see in detail.

Focus on the object described as: "{expression}"

Describe:
1. The object's appearance (color, size, shape)
2. The object's position in the scene
3. What the object is doing
4. Any other objects or people nearby

Be specific and detailed."""

        image = Image.open(image_path).convert('RGB')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)

        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        return response


print("✓ Functions defined")

# GEMINI HALLUCINATION TEST

def run_gemini_hallucination_test(dataset_path, num_samples=100, checkpoint_every=10):
    """
    Test Gemini for hallucinations across degraded conditions.
    """

    print("RQ4: Gemini Hallucination Test")

    results_dir = os.path.join(dataset_path, 'results')
    checkpoint_path = os.path.join(results_dir, 'rq4_gemini_checkpoint.json')

    print(f"Checkpoint path: {checkpoint_path}")

    # Load checkpoint if exists
    start_idx = 0
    processed_samples = []

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        start_idx = checkpoint.get('last_index', 0) + 1
        processed_samples = checkpoint.get('samples', [])
        print(f"✓ Resuming from checkpoint at sample {start_idx}")

    # Load samples
    samples = load_dataset(dataset_path, num_samples)

    # Initialize tester
    tester = GeminiHallucinationTester(gemini_client)

    # Progress
    total = len(samples) * len(CONDITIONS)
    completed = start_idx * len(CONDITIONS)
    pbar = tqdm(total=total, initial=completed, desc="Gemini Hallucination")

    for idx, sample in enumerate(samples):
        if idx < start_idx:
            continue

        filename = sample['filename']
        expression = sample['expression']

        sample_result = {
            'sample_id': sample['sample_id'],
            'filename': filename,
            'expression': expression,
            'conditions': {}
        }

        for condition in CONDITIONS:
            image_path = os.path.join(dataset_path, 'images', condition, filename)

            if not os.path.exists(image_path):
                pbar.update(1)
                continue

            try:
                # Get detailed description
                description = tester.get_detailed_description(image_path, expression)

                # Analyze for hallucinations
                analysis = analyze_hallucination_text(description, expression)

                sample_result['conditions'][condition] = {
                    'description': description[:500],
                    'hallucination_analysis': analysis
                }

            except Exception as e:
                sample_result['conditions'][condition] = {'error': str(e)}

            pbar.update(1)

        processed_samples.append(sample_result)

        # Save checkpoint
        if (idx + 1) % checkpoint_every == 0:
            checkpoint_data = {
                'last_index': idx,
                'samples': processed_samples,
                'timestamp': datetime.now().isoformat()
            }

            if save_checkpoint(checkpoint_path, checkpoint_data):
                tqdm.write(f"  ✓ Checkpoint VERIFIED at sample {idx + 1}")
            else:
                tqdm.write(f"  ⚠️ Checkpoint FAILED at sample {idx + 1}")

    pbar.close()

    # Save final results
    results = {
        'metadata': {
            'experiment': 'RQ4 - Gemini Hallucination Test',
            'model': 'Gemini 2.0 Flash',
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(processed_samples)
        },
        'samples': processed_samples
    }

    save_path = os.path.join(results_dir, 'rq4_gemini_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"\n✓ Results saved to {save_path}")

    # Delete checkpoint after successful save
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("✓ Checkpoint cleaned up")

    return results


# Run Gemini test
gemini_results = run_gemini_hallucination_test(DATASET_PATH, NUM_SAMPLES, CHECKPOINT_EVERY)

# QWEN2-VL HALLUCINATION TEST

def run_qwen_hallucination_test(dataset_path, num_samples=100, checkpoint_every=10):
    """
    Test Qwen2-VL for hallucinations across degraded conditions.
    """

    print("RQ4: Qwen2-VL Hallucination Test")

    results_dir = os.path.join(dataset_path, 'results')
    checkpoint_path = os.path.join(results_dir, 'rq4_qwen_checkpoint.json')

    print(f"Checkpoint path: {checkpoint_path}")

    # Load checkpoint if exists
    start_idx = 0
    processed_samples = []

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        start_idx = checkpoint.get('last_index', 0) + 1
        processed_samples = checkpoint.get('samples', [])
        print(f"✓ Resuming from checkpoint at sample {start_idx}")

    # Load samples
    samples = load_dataset(dataset_path, num_samples)

    # Initialize tester
    tester = QwenHallucinationTester(qwen_model, qwen_processor)

    # Progress
    total = len(samples) * len(CONDITIONS)
    completed = start_idx * len(CONDITIONS)
    pbar = tqdm(total=total, initial=completed, desc="Qwen Hallucination")

    for idx, sample in enumerate(samples):
        if idx < start_idx:
            continue

        filename = sample['filename']
        expression = sample['expression']

        sample_result = {
            'sample_id': sample['sample_id'],
            'filename': filename,
            'expression': expression,
            'conditions': {}
        }

        for condition in CONDITIONS:
            image_path = os.path.join(dataset_path, 'images', condition, filename)

            if not os.path.exists(image_path):
                pbar.update(1)
                continue

            try:
                # Get detailed description
                description = tester.get_detailed_description(image_path, expression)

                # Analyze for hallucinations
                analysis = analyze_hallucination_text(description, expression)

                sample_result['conditions'][condition] = {
                    'description': description[:500],
                    'hallucination_analysis': analysis
                }

            except Exception as e:
                sample_result['conditions'][condition] = {'error': str(e)}

            pbar.update(1)

        processed_samples.append(sample_result)

        # Save checkpoint with FORCED DISK WRITE
        if (idx + 1) % checkpoint_every == 0:
            checkpoint_data = {
                'last_index': idx,
                'samples': processed_samples,
                'timestamp': datetime.now().isoformat()
            }

            if save_checkpoint(checkpoint_path, checkpoint_data):
                tqdm.write(f"  ✓ Checkpoint VERIFIED at sample {idx + 1}")
            else:
                tqdm.write(f"  ⚠️ Checkpoint FAILED at sample {idx + 1}")

    pbar.close()

    # Save final results
    results = {
        'metadata': {
            'experiment': 'RQ4 - Qwen2-VL Hallucination Test',
            'model': 'Qwen2-VL-2B-Instruct',
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(processed_samples)
        },
        'samples': processed_samples
    }

    save_path = os.path.join(results_dir, 'rq4_qwen_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"\n✓ Results saved to {save_path}")

    # Delete checkpoint after successful save
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("✓ Checkpoint cleaned up")

    return results


# Run Qwen test
qwen_results = run_qwen_hallucination_test(DATASET_PATH, NUM_SAMPLES, CHECKPOINT_EVERY)

# CROSS-MODEL COMPARISON & ANALYSIS

def analyze_hallucination_results(dataset_path):
    """
    Analyze and compare hallucination results across models.
    """

    print("RQ4: CROSS-MODEL HALLUCINATION ANALYSIS")

    results_dir = os.path.join(dataset_path, 'results')

    # Load results
    gemini_path = os.path.join(results_dir, 'rq4_gemini_results.json')
    qwen_path = os.path.join(results_dir, 'rq4_qwen_results.json')

    with open(gemini_path) as f:
        gemini_data = json.load(f)

    with open(qwen_path) as f:
        qwen_data = json.load(f)

    # Initialize analysis storage
    analysis = {
        'gemini': {cond: {'fabrication': [], 'uncertainty': [], 'overconfidence': [],
                          'word_count': [], 'color_mentions': [], 'hallucination_score': []}
                   for cond in CONDITIONS},
        'qwen': {cond: {'fabrication': [], 'uncertainty': [], 'overconfidence': [],
                        'word_count': [], 'color_mentions': [], 'hallucination_score': []}
                 for cond in CONDITIONS}
    }

    # Process Gemini results
    for sample in gemini_data['samples']:
        for cond in CONDITIONS:
            if cond in sample.get('conditions', {}):
                cond_data = sample['conditions'][cond]
                if 'hallucination_analysis' in cond_data:
                    ha = cond_data['hallucination_analysis']
                    analysis['gemini'][cond]['fabrication'].append(ha['fabrication_count'])
                    analysis['gemini'][cond]['uncertainty'].append(ha['uncertainty_count'])
                    analysis['gemini'][cond]['overconfidence'].append(ha['overconfidence_count'])
                    analysis['gemini'][cond]['word_count'].append(ha['word_count'])
                    analysis['gemini'][cond]['color_mentions'].append(ha['color_mentions'])
                    analysis['gemini'][cond]['hallucination_score'].append(ha['hallucination_score'])

    # Process Qwen results
    for sample in qwen_data['samples']:
        for cond in CONDITIONS:
            if cond in sample.get('conditions', {}):
                cond_data = sample['conditions'][cond]
                if 'hallucination_analysis' in cond_data:
                    ha = cond_data['hallucination_analysis']
                    analysis['qwen'][cond]['fabrication'].append(ha['fabrication_count'])
                    analysis['qwen'][cond]['uncertainty'].append(ha['uncertainty_count'])
                    analysis['qwen'][cond]['overconfidence'].append(ha['overconfidence_count'])
                    analysis['qwen'][cond]['word_count'].append(ha['word_count'])
                    analysis['qwen'][cond]['color_mentions'].append(ha['color_mentions'])
                    analysis['qwen'][cond]['hallucination_score'].append(ha['hallucination_score'])

    # Calculate summaries
    gemini_summary = {}
    qwen_summary = {}

    print("GEMINI 2.0 FLASH - Hallucination Metrics")
    print(f"\n{'Condition':<15} {'Fabrication':<12} {'Uncertainty':<12} {'Overconf':<12} {'Colors':<10} {'H-Score':<10}")


    for cond in CONDITIONS:
        fab = np.mean(analysis['gemini'][cond]['fabrication']) if analysis['gemini'][cond]['fabrication'] else 0
        unc = np.mean(analysis['gemini'][cond]['uncertainty']) if analysis['gemini'][cond]['uncertainty'] else 0
        ovc = np.mean(analysis['gemini'][cond]['overconfidence']) if analysis['gemini'][cond]['overconfidence'] else 0
        col = np.mean(analysis['gemini'][cond]['color_mentions']) if analysis['gemini'][cond]['color_mentions'] else 0
        hsc = np.mean(analysis['gemini'][cond]['hallucination_score']) if analysis['gemini'][cond]['hallucination_score'] else 0

        gemini_summary[cond] = {
            'fabrication': float(fab),
            'uncertainty': float(unc),
            'overconfidence': float(ovc),
            'color_mentions': float(col),
            'h_score': float(hsc)
        }
        print(f"{cond:<15} {fab:<12.2f} {unc:<12.2f} {ovc:<12.2f} {col:<10.2f} {hsc:<10.2f}")

    print("QWEN2-VL-2B - Hallucination Metrics")
    print(f"\n{'Condition':<15} {'Fabrication':<12} {'Uncertainty':<12} {'Overconf':<12} {'Colors':<10} {'H-Score':<10}")

    for cond in CONDITIONS:
        fab = np.mean(analysis['qwen'][cond]['fabrication']) if analysis['qwen'][cond]['fabrication'] else 0
        unc = np.mean(analysis['qwen'][cond]['uncertainty']) if analysis['qwen'][cond]['uncertainty'] else 0
        ovc = np.mean(analysis['qwen'][cond]['overconfidence']) if analysis['qwen'][cond]['overconfidence'] else 0
        col = np.mean(analysis['qwen'][cond]['color_mentions']) if analysis['qwen'][cond]['color_mentions'] else 0
        hsc = np.mean(analysis['qwen'][cond]['hallucination_score']) if analysis['qwen'][cond]['hallucination_score'] else 0

        qwen_summary[cond] = {
            'fabrication': float(fab),
            'uncertainty': float(unc),
            'overconfidence': float(ovc),
            'color_mentions': float(col),
            'h_score': float(hsc)
        }
        print(f"{cond:<15} {fab:<12.2f} {unc:<12.2f} {ovc:<12.2f} {col:<10.2f} {hsc:<10.2f}")

    # Plot comparison
    plot_hallucination_comparison(gemini_summary, qwen_summary, results_dir)

    # Key findings
    print("KEY FINDINGS")

    # Compare clean vs thermal
    gemini_clean_h = gemini_summary['clean']['h_score']
    gemini_thermal_h = gemini_summary['thermal_0.5']['h_score']
    qwen_clean_h = qwen_summary['clean']['h_score']
    qwen_thermal_h = qwen_summary['thermal_0.5']['h_score']

    print(f"\n1. HALLUCINATION SCORE CHANGE (clean → thermal):")
    print(f"   Gemini: {gemini_clean_h:.2f} → {gemini_thermal_h:.2f} ({gemini_thermal_h - gemini_clean_h:+.2f})")
    print(f"   Qwen:   {qwen_clean_h:.2f} → {qwen_thermal_h:.2f} ({qwen_thermal_h - qwen_clean_h:+.2f})")

    # Color hallucinations in thermal
    gemini_clean_col = gemini_summary['clean']['color_mentions']
    gemini_thermal_col = gemini_summary['thermal_0.5']['color_mentions']
    qwen_clean_col = qwen_summary['clean']['color_mentions']
    qwen_thermal_col = qwen_summary['thermal_0.5']['color_mentions']

    print(f"\n2. COLOR MENTIONS (thermal images are purple/orange, not natural colors):")
    print(f"   Gemini: {gemini_clean_col:.2f} (clean) → {gemini_thermal_col:.2f} (thermal)")
    print(f"   Qwen:   {qwen_clean_col:.2f} (clean) → {qwen_thermal_col:.2f} (thermal)")

    if gemini_thermal_col > 1 or qwen_thermal_col > 1:
        print(f"\n   ⚠️ Models mention colors in thermal images = COLOR HALLUCINATION")

    # Conclusion
    print(f"\n3. CONCLUSION:")
    if gemini_thermal_h > gemini_clean_h and qwen_thermal_h > qwen_clean_h:
        print(f"   ✗ CONFIRMED: Both models hallucinate MORE in degraded conditions")
    elif gemini_thermal_h > gemini_clean_h:
        print(f"   ⚠️ Gemini hallucinates more in degraded conditions")
    elif qwen_thermal_h > qwen_clean_h:
        print(f"   ⚠️ Qwen hallucinates more in degraded conditions")
    else:
        print(f"   ✓ Neither model shows increased hallucination in degraded conditions")

    # Which model is more robust?
    avg_gemini_h = np.mean([gemini_summary[c]['h_score'] for c in CONDITIONS])
    avg_qwen_h = np.mean([qwen_summary[c]['h_score'] for c in CONDITIONS])

    print(f"\n4. OVERALL HALLUCINATION RISK:")
    print(f"   Gemini average H-score: {avg_gemini_h:.2f}")
    print(f"   Qwen average H-score:   {avg_qwen_h:.2f}")

    if avg_gemini_h < avg_qwen_h:
        print(f"\n   → Gemini is MORE ROBUST to hallucination")
    else:
        print(f"\n   → Qwen is MORE ROBUST to hallucination")

    # Save summary
    summary = {
        'metadata': {
            'experiment': 'RQ4 - Hallucination Risk Analysis',
            'timestamp': datetime.now().isoformat()
        },
        'gemini': gemini_summary,
        'qwen': qwen_summary,
        'findings': {
            'gemini_clean_to_thermal': float(gemini_thermal_h - gemini_clean_h),
            'qwen_clean_to_thermal': float(qwen_thermal_h - qwen_clean_h),
            'gemini_avg_h_score': float(avg_gemini_h),
            'qwen_avg_h_score': float(avg_qwen_h)
        }
    }

    summary_path = os.path.join(results_dir, 'rq4_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"\n✓ Summary saved to {summary_path}")

    return summary


def plot_hallucination_comparison(gemini_summary, qwen_summary, results_dir):
    """Plot hallucination comparison between models."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RQ4: Hallucination Risk - When Rich Language Meets Degraded Vision',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(CONDITIONS))
    width = 0.35

    # Plot 1: Fabrication Count
    ax1 = axes[0, 0]
    gemini_fab = [gemini_summary[c]['fabrication'] for c in CONDITIONS]
    qwen_fab = [qwen_summary[c]['fabrication'] for c in CONDITIONS]

    ax1.bar(x - width/2, gemini_fab, width, label='Gemini', color='#4285F4')
    ax1.bar(x + width/2, qwen_fab, width, label='Qwen2-VL', color='#9C27B0')
    ax1.set_ylabel('Avg Fabricated Objects')
    ax1.set_title('Object Fabrication')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('_0.5', '') for c in CONDITIONS])
    ax1.legend()

    # Plot 2: Color Mentions (key for thermal)
    ax2 = axes[0, 1]
    gemini_col = [gemini_summary[c]['color_mentions'] for c in CONDITIONS]
    qwen_col = [qwen_summary[c]['color_mentions'] for c in CONDITIONS]

    ax2.bar(x - width/2, gemini_col, width, label='Gemini', color='#4285F4')
    ax2.bar(x + width/2, qwen_col, width, label='Qwen2-VL', color='#9C27B0')
    ax2.set_ylabel('Avg Color Mentions')
    ax2.set_title('Color Mentions (Hallucination in Thermal)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace('_0.5', '') for c in CONDITIONS])
    ax2.legend()

    # Plot 3: Overconfidence
    ax3 = axes[1, 0]
    gemini_ovc = [gemini_summary[c]['overconfidence'] for c in CONDITIONS]
    qwen_ovc = [qwen_summary[c]['overconfidence'] for c in CONDITIONS]

    ax3.bar(x - width/2, gemini_ovc, width, label='Gemini', color='#4285F4')
    ax3.bar(x + width/2, qwen_ovc, width, label='Qwen2-VL', color='#9C27B0')
    ax3.set_ylabel('Avg Overconfidence Phrases')
    ax3.set_title('Overconfidence')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace('_0.5', '') for c in CONDITIONS])
    ax3.legend()

    # Plot 4: Hallucination Score
    ax4 = axes[1, 1]
    gemini_h = [gemini_summary[c]['h_score'] for c in CONDITIONS]
    qwen_h = [qwen_summary[c]['h_score'] for c in CONDITIONS]

    ax4.bar(x - width/2, gemini_h, width, label='Gemini', color='#4285F4')
    ax4.bar(x + width/2, qwen_h, width, label='Qwen2-VL', color='#9C27B0')
    ax4.set_ylabel('Hallucination Score')
    ax4.set_title('Overall Hallucination Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels([c.replace('_0.5', '') for c in CONDITIONS])
    ax4.legend()
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    save_path = os.path.join(results_dir, 'rq4_hallucination_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {save_path}")
    plt.show()


# Run analysis
summary = analyze_hallucination_results(DATASET_PATH)

print("RQ4 COMPLETE!")