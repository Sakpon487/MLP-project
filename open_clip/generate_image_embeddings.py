#!/usr/bin/env python3
"""
Inference pipeline to generate image embeddings using CLIP models.

Requirements:
- Uses original CLIP preprocessing pipeline
- Supports MPS (Apple Silicon GPU) for ViT models
- ResNet models automatically use CPU (MPS has fatal compatibility issues)
- Input: txt file with format: image_id class_id super_class_id path
- Output: N x D embedding array and N array of super_class_ids

Available models:
- ResNet: RN50, RN101, RN50x4, RN50x16, RN50x64 (CPU only - MPS causes fatal errors)
- Vision Transformer: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px (MPS supported)
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add CLIP to path before importing
clip_path = Path(__file__).parent.parent / 'CLIP'
if str(clip_path) not in sys.path:
    sys.path.insert(0, str(clip_path))

import clip


def get_model_info(model_name):
    """
    Get information about a CLIP model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model information
    """
    available = clip.available_models()
    
    info = {
        'name': model_name,
        'available': model_name in available,
        'available_models': available,
    }
    
    # Model type classification
    if model_name.startswith('RN'):
        info['type'] = 'ResNet'
        if model_name == 'RN50':
            info['size'] = 'small'
            info['embed_dim'] = 1024
        elif model_name == 'RN101':
            info['size'] = 'medium'
            info['embed_dim'] = 512
        else:
            info['size'] = 'large'
            info['embed_dim'] = 1024
    elif model_name.startswith('ViT'):
        info['type'] = 'Vision Transformer'
        if 'B' in model_name:
            info['size'] = 'base'
            info['embed_dim'] = 512
        elif 'L' in model_name:
            info['size'] = 'large'
            info['embed_dim'] = 768
        else:
            info['embed_dim'] = 512  # default
    else:
        info['type'] = 'unknown'
        info['embed_dim'] = None
    
    return info


def load_dataset(dataset_file, base_image_dir=None):
    """
    Load dataset from text file.
    
    Expected format (with header):
    image_id class_id super_class_id path
    1 1 1 bicycle_final/111085122871_0.JPG
    
    Args:
        dataset_file: Path to dataset file
        base_image_dir: Base directory for images (if paths are relative)
    
    Returns:
        List of tuples: (image_path, super_class_id)
    """
    dataset = []
    
    with open(dataset_file, 'r') as f:
        # Skip header
        header = f.readline().strip()
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
            
            image_id = parts[0]
            class_id = parts[1]
            super_class_id = parts[2]
            image_path = ' '.join(parts[3:])  # Handle paths with spaces
            
            # Construct full path
            if base_image_dir:
                full_path = Path(base_image_dir) / image_path
            else:
                full_path = Path(image_path)
            
            if full_path.exists():
                dataset.append((str(full_path), int(super_class_id)))
            else:
                print(f"Warning: Image not found: {full_path}")
    
    return dataset


def generate_embeddings(dataset_file, base_image_dir=None, batch_size=32, device='mps', model_name='ViT-B/32'):
    """
    Generate image embeddings for all images in the dataset.
    
    Args:
        dataset_file: Path to dataset file
        base_image_dir: Base directory for images
        batch_size: Batch size for processing
        device: Device to use ('mps', 'cuda', or 'cpu')
        model_name: CLIP model name (e.g., 'RN50', 'ViT-B/32', 'ViT-B/16')
    
    Returns:
        embeddings: N x D numpy array of embeddings
        super_class_ids: N numpy array of super_class_ids
        model_name: Name of the model used
        actual_device: Actual device used (may differ from requested if MPS fails)
    """
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_file, base_image_dir)
    print(f"Loaded {len(dataset)} images")
    
    if len(dataset) == 0:
        raise ValueError("No valid images found in dataset")
    
    # Get model info
    model_info = get_model_info(model_name)
    
    # Check available models
    if not model_info['available']:
        print(f"Warning: Model '{model_name}' not in available models.")
        print(f"Available models: {', '.join(model_info['available_models'])}")
        print(f"Using '{model_name}' anyway (might be a custom checkpoint path)")
    else:
        print(f"Model type: {model_info['type']}")
        if 'size' in model_info:
            print(f"Model size: {model_info['size']}")
    
    # CRITICAL: ResNet models cause fatal MPS errors that cannot be caught
    # Force CPU for ResNet models BEFORE calling clip.load()
    if model_name.startswith('RN') and device == 'mps':
        print("⚠️  ResNet models have fatal MPS compatibility issues.")
        print("   Automatically forcing CPU for ResNet models...")
        device = 'cpu'
    
    print(f"Loading CLIP model ({model_name}) on {device}...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    print(f"✓ Model loaded successfully on {device}")
    
    # Get embedding dimension
    with torch.no_grad():
        dummy_image = preprocess(Image.new('RGB', (224, 224))).unsqueeze(0).to(device)
        dummy_embedding = model.encode_image(dummy_image)
        embedding_dim = dummy_embedding.shape[1]
    
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Using device: {device}")
    
    # Initialize output arrays
    all_embeddings = []
    all_super_class_ids = []
    
    # Process in batches
    print("Generating embeddings...")
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch = dataset[i:i+batch_size]
        batch_images = []
        batch_super_class_ids = []
        
        # Load and preprocess images
        for image_path, super_class_id in batch:
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = preprocess(image)
                batch_images.append(image_tensor)
                batch_super_class_ids.append(super_class_id)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack into batch tensor
        image_batch = torch.stack(batch_images).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            image_features = model.encode_image(image_batch)
            # Normalize embeddings (CLIP standard)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Store embeddings and super_class_ids
        all_embeddings.append(image_features.cpu().numpy())
        all_super_class_ids.extend(batch_super_class_ids)
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    super_class_ids = np.array(all_super_class_ids)
    
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    print(f"Super class IDs shape: {super_class_ids.shape}")
    print(f"Number of unique super classes: {len(np.unique(super_class_ids))}")
    print(f"Final device used: {device}")
    
    return embeddings, super_class_ids, model_name, device


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate CLIP image embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  ResNet models:
    RN50        - ResNet-50 (smallest ResNet)
    RN101       - ResNet-101
    RN50x4      - ResNet-50 4x wider
    RN50x16     - ResNet-50 16x wider
    RN50x64     - ResNet-50 64x wider
  
  Vision Transformer models:
    ViT-B/32    - ViT Base, patch size 32 (smallest ViT)
    ViT-B/16    - ViT Base, patch size 16
    ViT-L/14    - ViT Large, patch size 14
    ViT-L/14@336px - ViT Large, 336x336 resolution

Examples:
  # Use ResNet-50
  python generate_image_embeddings.py --model RN50
  
  # Use ViT-B/16
  python generate_image_embeddings.py --model ViT-B/16
  
  # List available models
  python -c "import clip; print(clip.available_models())"
        """
    )
    parser.add_argument(
        '--dataset-file',
        type=str,
        default='SOP/.data/Ebay_train.txt',
        help='Path to dataset file (relative to working dir or absolute)'
    )
    parser.add_argument(
        '--base-image-dir',
        type=str,
        default="/Users/boud/mlpractical/final_project/open_clip/SOP/.data",
        help='Base directory for images (if paths in dataset are relative)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ViT-B/32',
        help='CLIP model name (default: ViT-B/32). Use --list-models to see all available models.'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available CLIP models and exit'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use (default: mps for Apple Silicon). Note: ResNet models automatically use CPU due to fatal MPS compatibility issues.'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        default=False,
        help='Force CPU usage (useful if MPS has compatibility issues)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./embeddings',
        help='Output directory for embeddings (default: ./embeddings)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default=None,
        help='Prefix for output files (default: auto-generated from model name)'
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("Available CLIP models:")
        available = clip.available_models()
        for model in available:
            print(f"  - {model}")
        return
    
    # Resolve dataset file path
    dataset_file = Path(args.dataset_file)
    if not dataset_file.is_absolute():
        # Assume relative to current working directory
        working_dir = Path('/Users/boud/mlpractical/final_project/open_clip')
        dataset_file = working_dir / args.dataset_file
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    print(f"Dataset file: {dataset_file}")
    print(f"Model: {args.model}")
    
    # Force CPU if requested
    if args.force_cpu:
        print("Force CPU mode enabled")
        args.device = 'cpu'
    
    # Check device availability
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # CRITICAL: Automatically force CPU for ResNet models
    # ResNet models cause fatal MPS errors (LLVM abort) that cannot be caught
    # This must happen BEFORE any model loading attempts
    if args.device == 'mps' and args.model.startswith('RN'):
        print("⚠️  ResNet models have fatal MPS compatibility issues (LLVM abort).")
        print("   Automatically forcing CPU for ResNet models...")
        args.device = 'cpu'
    
    # Generate embeddings
    embeddings, super_class_ids, model_name, actual_device = generate_embeddings(
        dataset_file,
        base_image_dir=args.base_image_dir,
        batch_size=args.batch_size,
        device=args.device,
        model_name=args.model
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output prefix if not provided
    if args.output_prefix is None:
        # Create safe filename from model name
        model_safe = model_name.replace('/', '_').replace('@', '_')
        args.output_prefix = f'{model_safe}_embeddings'
    
    # Save embeddings
    embeddings_file = output_dir / f'{args.output_prefix}_embeddings.npy'
    np.save(embeddings_file, embeddings)
    print(f"\nSaved embeddings to: {embeddings_file}")
    
    # Save super class IDs
    super_class_ids_file = output_dir / f'{args.output_prefix}_super_class_ids.npy'
    np.save(super_class_ids_file, super_class_ids)
    print(f"Saved super class IDs to: {super_class_ids_file}")
    
    # Save metadata
    metadata = {
        'num_images': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'num_super_classes': len(np.unique(super_class_ids)),
        'super_class_ids_unique': np.unique(super_class_ids).tolist(),
        'device_requested': args.device,
        'device_actual': actual_device,
        'model': model_name,
        'pretrained': 'openai'
    }
    
    import json
    metadata_file = output_dir / f'{args.output_prefix}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")
    
    print(f"\n✓ Done!")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Super class IDs shape: {super_class_ids.shape}")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
