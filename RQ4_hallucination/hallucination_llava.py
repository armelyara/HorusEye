"""
RQ4: Hallucination Risk with LLaVA-1.5
Proper prompting and response decoding.

Research Question: Do VLMs hallucinate more when vision is degraded?
"""

# SETUP & INSTALL

#!pip install transformers accelerate pillow tqdm bitsandbytes matplotlib -q

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

#from google.colab import drive
#drive.mount('/content/drive')

print("✓ Packages installed")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# CONFIGURATION

DATASET_PATH = '/HorusEye/horuseye_VLM/refcoco_degraded_benchmark'
RQ1_RESULTS_PATH = os.path.join(DATASET_PATH, 'results', 'rq1_with_bboxes.json')

CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
NUM_SAMPLES = 100
CHECKPOINT_EVERY = 25
MODEL_NAME = "llava"

print(f"✓ Configuration set")
print(f"  Dataset: {DATASET_PATH}")
print(f"  Samples: {NUM_SAMPLES}")


# LOAD MODEL
print("Loading LLaVA-1.5 model...")
print("  This may take 3-5 minutes...")

# 4-bit quantization for T4 GPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config
)

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

model.eval()
print("✓ LLaVA-1.5 loaded!")


# DEFINE HALLUCINATION DETECTION FUNCTIONS

# Objects that are commonly hallucinated
COMMON_HALLUCINATED_OBJECTS = [
    'phone', 'bag', 'hat', 'umbrella', 'book', 'cup', 'bottle', 
    'glasses', 'watch', 'backpack', 'purse', 'wallet', 'keys',
    'dog', 'cat', 'bird', 'car', 'bicycle', 'tree', 'flower'
]

# Hallucination indicator phrases
UNCERTAINTY_PHRASES = [
    'might be', 'could be', 'possibly', 'appears to', 'seems like', 
    'i think', 'maybe', 'probably', 'unclear', 'hard to tell', 
    'cannot determine', 'not sure', 'difficult to see'
]

OVERCONFIDENCE_PHRASES = [
    'clearly', 'definitely', 'certainly', 'obviously', 
    'without doubt', 'absolutely', 'undoubtedly'
]

# Colors (hallucination indicator in thermal images)
COLORS = [
    'red', 'blue', 'green', 'yellow', 'white', 'black', 
    'orange', 'purple', 'pink', 'brown'
]


def load_dataset(num_samples=100):
    """Load dataset from RQ1 results, filtering for person-related expressions."""
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
    
    print(f"✓ Loaded {len(samples)} samples (person-related filter applied)")
    return samples


def analyze_hallucination_text(response_text, expression):
    """
    Analyze text response for hallucination indicators.
    Matches the exact methodology used for Gemini, Qwen, and BLIP-2.
    """
    response_lower = response_text.lower()
    expression_lower = expression.lower()
    
    # Count uncertainty phrases
    uncertainty_count = sum(1 for phrase in UNCERTAINTY_PHRASES if phrase in response_lower)
    
    # Count fabricated objects
    fabrication_count = 0
    fabricated_items = []
    for obj in COMMON_HALLUCINATED_OBJECTS:
        if obj in response_lower and obj not in expression_lower:
            fabrication_count += 1
            fabricated_items.append(obj)
    
    # Count overconfidence phrases
    overconfidence_count = sum(1 for phrase in OVERCONFIDENCE_PHRASES if phrase in response_lower)
    
    # Response length
    word_count = len(response_text.split())
    
    # Color mentions (hallucination in thermal images)
    color_mentions = sum(1 for color in COLORS if color in response_lower)
    
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


def get_detailed_description_llava(image_path, expression, model, processor):
    """
    Ask LLaVA for detailed description.
    Uses proper chat template and decodes only NEW tokens.
    """
    image = Image.open(image_path).convert('RGB')
    
    # LLaVA prompt
    prompt_text = f"Describe the {expression} in this image in detail. What are they wearing, what color clothes, what are they doing, and what objects are nearby?"
    
    # Build conversation format for LLaVA
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    ).to(model.device, torch.float16)
    
    # Store input length for later
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )
    
    # Decode only the new tokens, not the input prompt
    generated_ids = outputs[0][input_length:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    return response


print("✓ Functions defined")

# RUN HALLUCINATION TEST

def run_llava_hallucination_test(dataset_path, num_samples=100, checkpoint_every=25):
    """
    Test LLaVA for hallucinations across degraded conditions.
    """
    
    print("RQ4: LLaVA Hallucination Test")
    
    results_dir = os.path.join(dataset_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(results_dir, f'rq4_{MODEL_NAME}_checkpoint.json')
    
    # Load checkpoint if exists
    start_idx = 0
    processed_samples = []
    
    if os.path.exists(checkpoint_path):
        print("✓ Found checkpoint, resuming...")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        start_idx = checkpoint.get('last_index', 0) + 1
        processed_samples = checkpoint.get('samples', [])
        print(f"  Resuming from sample {start_idx}")
    
    # Load samples
    samples = load_dataset(num_samples)
    
    # Progress
    total = len(samples) * len(CONDITIONS)
    completed = start_idx * len(CONDITIONS)
    pbar = tqdm(total=total, initial=completed, desc="LLaVA Hallucination")
    
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
                description = get_detailed_description_llava(
                    image_path, expression, model, processor
                )
                
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
            checkpoint = {
                'last_index': idx,
                'samples': processed_samples,
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)
                f.flush()
                os.fsync(f.fileno())
            tqdm.write(f"  ✓ Checkpoint saved at sample {idx + 1}")
    
    pbar.close()
    
    # Save final results
    results = {
        'metadata': {
            'experiment': 'RQ4 - LLaVA Hallucination Test',
            'model': 'LLaVA-1.5-7B',
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(processed_samples)
        },
        'samples': processed_samples
    }
    
    save_path = os.path.join(results_dir, f'rq4_{MODEL_NAME}_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {save_path}")
    
    # Delete checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return results


# Run the test
llava_results = run_llava_hallucination_test(DATASET_PATH, NUM_SAMPLES, CHECKPOINT_EVERY)


# ANALYZE RESULTS

def analyze_llava_results(dataset_path):
    """Analyze LLaVA hallucination results."""
    
    print("RQ4: LLaVA Hallucination Analysis")
    
    results_dir = os.path.join(dataset_path, 'results')
    llava_path = os.path.join(results_dir, f'rq4_{MODEL_NAME}_results.json')
    
    with open(llava_path) as f:
        llava_data = json.load(f)
    
    # Initialize analysis storage
    analysis = {cond: {
        'fabrication': [], 
        'uncertainty': [], 
        'overconfidence': [], 
        'word_count': [], 
        'color_mentions': [], 
        'hallucination_score': []
    } for cond in CONDITIONS}
    
    # Process results
    for sample in llava_data['samples']:
        for cond in CONDITIONS:
            if cond in sample.get('conditions', {}):
                cond_data = sample['conditions'][cond]
                if 'hallucination_analysis' in cond_data:
                    ha = cond_data['hallucination_analysis']
                    analysis[cond]['fabrication'].append(ha['fabrication_count'])
                    analysis[cond]['uncertainty'].append(ha['uncertainty_count'])
                    analysis[cond]['overconfidence'].append(ha['overconfidence_count'])
                    analysis[cond]['word_count'].append(ha['word_count'])
                    analysis[cond]['color_mentions'].append(ha['color_mentions'])
                    analysis[cond]['hallucination_score'].append(ha['hallucination_score'])
    
    # Print summary
    print("LLaVA Hallucination Metrics")
    print(f"\n{'Condition':<15} {'Fabrication':<12} {'Uncertainty':<12} {'Overconf':<12} {'Colors':<10} {'H-Score':<10}")
    
    summary = {}
    for cond in CONDITIONS:
        fab = np.mean(analysis[cond]['fabrication']) if analysis[cond]['fabrication'] else 0
        unc = np.mean(analysis[cond]['uncertainty']) if analysis[cond]['uncertainty'] else 0
        ovc = np.mean(analysis[cond]['overconfidence']) if analysis[cond]['overconfidence'] else 0
        col = np.mean(analysis[cond]['color_mentions']) if analysis[cond]['color_mentions'] else 0
        hsc = np.mean(analysis[cond]['hallucination_score']) if analysis[cond]['hallucination_score'] else 0
        wc = np.mean(analysis[cond]['word_count']) if analysis[cond]['word_count'] else 0
        
        summary[cond] = {
            'fabrication': float(fab),
            'uncertainty': float(unc),
            'overconfidence': float(ovc),
            'color_mentions': float(col),
            'h_score': float(hsc),
            'avg_word_count': float(wc)
        }
        print(f"{cond:<15} {fab:<12.2f} {unc:<12.2f} {ovc:<12.2f} {col:<10.2f} {hsc:<10.2f}")
    
    # Key findings
    clean_h = summary['clean']['h_score']
    thermal_h = summary['thermal_0.5']['h_score']
    clean_col = summary['clean']['color_mentions']
    thermal_col = summary['thermal_0.5']['color_mentions']
    avg_wc = np.mean([summary[c]['avg_word_count'] for c in CONDITIONS])
    
    print(f"\n1. Average response length: {avg_wc:.1f} words")
    print(f"\n2. H-Score change (clean → thermal): {clean_h:.2f} → {thermal_h:.2f} ({thermal_h - clean_h:+.2f})")
    print(f"\n3. Color mentions (clean → thermal): {clean_col:.2f} → {thermal_col:.2f}")
    
    if thermal_col > clean_col:
        print(f"\n   ⚠️ LLaVA mentions MORE colors in thermal = hallucination!")
    elif thermal_col < clean_col:
        print(f"\n   ✓ LLaVA mentions FEWER colors in thermal = appropriate caution")
    
    if thermal_h > clean_h:
        print(f"\n   ✗ LLaVA hallucinates MORE in degraded conditions")
    else:
        print(f"\n   ✓ LLaVA does NOT show increased hallucination in degraded conditions")
    
    # Save summary
    summary_data = {
        'model': 'LLaVA-1.5-7B',
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'findings': {
            'clean_to_thermal_h_change': float(thermal_h - clean_h),
            'clean_to_thermal_color_change': float(thermal_col - clean_col),
            'avg_word_count': float(avg_wc)
        }
    }
    
    summary_path = os.path.join(results_dir, f'rq4_{MODEL_NAME}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_path}")
    
    return summary


# Run analysis
summary = analyze_llava_results(DATASET_PATH)


# CHECK SAMPLE RESPONSES

def debug_check_responses(dataset_path, num_to_check=3):
    """Check what LLaVA actually outputs."""
    
    print("DEBUG: Checking LLaVA Responses")
    
    results_dir = os.path.join(dataset_path, 'results')
    llava_path = os.path.join(results_dir, f'rq4_{MODEL_NAME}_results.json')
    
    with open(llava_path) as f:
        llava_data = json.load(f)
    
    samples = llava_data['samples'][:num_to_check]
    
    for i, sample in enumerate(samples):
        print(f"\n[Sample {i+1}] {sample['filename']}")
        print(f"  Expression: {sample['expression']}")
        
        for cond in CONDITIONS:
            if cond in sample.get('conditions', {}):
                cond_data = sample['conditions'][cond]
                if 'description' in cond_data:
                    desc = cond_data['description']
                    print(f"\n  [{cond}]")
                    print(f"    Response ({len(desc.split())} words): {desc[:150]}...")
                    
                    if 'hallucination_analysis' in cond_data:
                        ha = cond_data['hallucination_analysis']
                        print(f"    → Colors: {ha['color_mentions']}, Fabrication: {ha['fabrication_count']}, H-Score: {ha['hallucination_score']:.2f}")

# Run debug
debug_check_responses(DATASET_PATH, num_to_check=3)


# COMPARE ALL RQ4 MODELS

def compare_all_rq4_models(dataset_path):
    """Compare all RQ4 model results."""
    
    print("RQ4: ALL MODELS COMPARISON")
    
    results_dir = os.path.join(dataset_path, 'results')
    
    models = {
        'Gemini': 'rq4_gemini_results.json',
        'Qwen2-VL': 'rq4_qwen_results.json',
        'BLIP-2': 'rq4_blip2_results.json',
        'LLaVA': 'rq4_llava_results.json'
    }
    
    all_summaries = {}
    
    for model_name, filename in models.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
            
            # Calculate summary
            analysis = {cond: {'h_score': [], 'color_mentions': []} for cond in CONDITIONS}
            
            for sample in data['samples']:
                for cond in CONDITIONS:
                    if cond in sample.get('conditions', {}):
                        cond_data = sample['conditions'][cond]
                        if 'hallucination_analysis' in cond_data:
                            ha = cond_data['hallucination_analysis']
                            analysis[cond]['h_score'].append(ha['hallucination_score'])
                            analysis[cond]['color_mentions'].append(ha['color_mentions'])
            
            summary = {}
            for cond in CONDITIONS:
                summary[cond] = {
                    'h_score': np.mean(analysis[cond]['h_score']) if analysis[cond]['h_score'] else 0,
                    'colors': np.mean(analysis[cond]['color_mentions']) if analysis[cond]['color_mentions'] else 0
                }
            
            all_summaries[model_name] = summary
            print(f"✓ Loaded {model_name}")
        else:
            print(f"✗ {model_name} not found")
    
    if not all_summaries:
        print("No model results found!")
        return
    
    # Print H-Score comparison
    print("H-SCORE COMPARISON (Higher = More Hallucination)")
    
    header = f"{'Condition':<15}" + "".join(f"{m:<15}" for m in all_summaries.keys())
    print(f"\n{header}")
    print("-"*len(header))
    
    for cond in CONDITIONS:
        row = f"{cond:<15}"
        for model_name in all_summaries.keys():
            h = all_summaries[model_name][cond]['h_score']
            row += f"{h:<15.2f}"
        print(row)
    
    # Print Color Mentions comparison
    print("COLOR MENTIONS (Higher in Thermal = Hallucination)")
    
    print(f"\n{header}")
    print("-"*len(header))
    
    for cond in CONDITIONS:
        row = f"{cond:<15}"
        for model_name in all_summaries.keys():
            c = all_summaries[model_name][cond]['colors']
            row += f"{c:<15.2f}"
        print(row)
    
    # Summary of findings
    print(f"\n{'Model':<15} {'Clean H-Score':<15} {'Thermal H-Score':<17} {'Change':<10}")
    
    for model_name, summary in all_summaries.items():
        clean_h = summary['clean']['h_score']
        thermal_h = summary['thermal_0.5']['h_score']
        change = thermal_h - clean_h
        print(f"{model_name:<15} {clean_h:<15.2f} {thermal_h:<17.2f} {change:+.2f}")
    
    # Save comparison
    comparison_path = os.path.join(results_dir, 'rq4_all_models_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(all_summaries, f, indent=2, default=float)
    
    print(f"\n✓ Comparison saved to {comparison_path}")


# Run comparison
compare_all_rq4_models(DATASET_PATH)
print("RQ4 LLaVA COMPLETE!")