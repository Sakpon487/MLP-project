#!/usr/bin/env python3
"""
Simple pipeline to generate CLIP embeddings for images.

This script can:
1. Generate image embeddings
2. Generate text embeddings (if text descriptions are provided)
3. Optionally generate text descriptions using CoCa models
4. Save embeddings to disk
"""

import argparse
import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import open_clip


def load_dataset(dataset_file, image_dir, delimiter=' '):
    """
    Load dataset from a text file.
    
    Expected format: category_id image_filename
    Example: 1 111085122871_0.JPG
    
    Args:
        dataset_file: Path to dataset file
        image_dir: Directory containing images
        delimiter: Delimiter in dataset file (default: space)
    
    Returns:
        List of tuples: (image_path, category_id, image_filename)
    """
    dataset = []
    image_dir = Path(image_dir)
    
    with open(dataset_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(delimiter, 1)  # Split only on first delimiter
            if len(parts) < 2:
                continue
            
            category_id = parts[0].strip()
            image_filename = parts[1].strip()
            image_path = image_dir / image_filename
            
            if image_path.exists():
                dataset.append((str(image_path), category_id, image_filename))
            else:
                print(f"Warning: Image not found: {image_path}")
    
    return dataset


def generate_image_embeddings(model, preprocess, dataset, batch_size=32, device='cuda'):
    """
    Generate image embeddings for all images in the dataset.
    
    Args:
        model: CLIP model
        preprocess: Image preprocessing function
        dataset: List of (image_path, category_id, image_filename) tuples
        batch_size: Batch size for processing
        device: Device to run on
    
    Returns:
        Dictionary mapping image_filename to embedding
    """
    model.eval()
    embeddings = {}
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating image embeddings"):
        batch = dataset[i:i+batch_size]
        batch_images = []
        batch_filenames = []
        
        # Load and preprocess images
        for image_path, category_id, image_filename in batch:
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = preprocess(image)
                batch_images.append(image_tensor)
                batch_filenames.append(image_filename)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack into batch tensor
        image_batch = torch.stack(batch_images).to(device)
        
        # Generate embeddings
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image_batch)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Store embeddings
        for j, filename in enumerate(batch_filenames):
            embeddings[filename] = image_features[j].cpu().numpy()
    
    return embeddings


def generate_text_embeddings(model, tokenizer, texts, batch_size=32, device='cuda'):
    """
    Generate text embeddings for given texts.
    
    Args:
        model: CLIP model
        tokenizer: Text tokenizer
        texts: List of text strings
        batch_size: Batch size for processing
        device: Device to run on
    
    Returns:
        Dictionary mapping text to embedding
    """
    model.eval()
    embeddings = {}
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating text embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        text_tokens = tokenizer(batch_texts).to(device)
        
        # Generate embeddings
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(text_tokens)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Store embeddings
        for j, text in enumerate(batch_texts):
            embeddings[text] = text_features[j].cpu().numpy()
    
    return embeddings


def generate_captions(model, preprocess, dataset, batch_size=8, device='cuda', 
                     seq_len=30, generation_type='beam_search'):
    """
    Generate text captions for images using CoCa model.
    
    Args:
        model: CoCa model (must support .generate() method)
        preprocess: Image preprocessing function
        dataset: List of (image_path, category_id, image_filename) tuples
        batch_size: Batch size for processing (smaller for generation)
        device: Device to run on
        seq_len: Maximum sequence length for generation
        generation_type: Generation type ('beam_search', 'top_p', 'top_k')
    
    Returns:
        Dictionary mapping image_filename to generated caption
    """
    model.eval()
    captions = {}
    
    # Process one at a time for generation (can be slow)
    for image_path, category_id, image_filename in tqdm(dataset, desc="Generating captions"):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            # Generate caption
            with torch.no_grad(), torch.cuda.amp.autocast():
                generated = model.generate(
                    image_tensor,
                    seq_len=seq_len,
                    generation_type=generation_type
                )
            
            # Decode caption
            caption = open_clip.decode(generated[0])
            # Clean up caption
            caption = caption.split("<end_of_text>")[0].replace("<start_of_text>", "").strip()
            captions[image_filename] = caption
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            captions[image_filename] = ""
    
    return captions


def save_embeddings(embeddings, output_file, format='numpy'):
    """
    Save embeddings to disk.
    
    Args:
        embeddings: Dictionary mapping keys to embeddings
        output_file: Output file path
        format: Save format ('numpy', 'json', 'npz')
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'numpy' or format == 'npz':
        # Save as compressed numpy file
        np.savez_compressed(output_file, **embeddings)
        print(f"Saved embeddings to {output_file} (NPZ format)")
    
    elif format == 'json':
        # Convert to lists for JSON serialization
        json_embeddings = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in embeddings.items()}
        with open(output_file, 'w') as f:
            json.dump(json_embeddings, f)
        print(f"Saved embeddings to {output_file} (JSON format)")


def main():
    parser = argparse.ArgumentParser(description='Generate CLIP embeddings for images')
    parser.add_argument('--dataset-file', type=str, required=True,
                       help='Path to dataset file (format: category_id image_filename)')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--model', type=str, default='ViT-B-32',
                       help='Model architecture (default: ViT-B-32)')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b79k',
                       help='Pretrained weights (default: laion2b_s34b_b79k)')
    parser.add_argument('--output-dir', type=str, default='./embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (default: cuda if available)')
    parser.add_argument('--generate-captions', action='store_true',
                       help='Generate text captions using CoCa (requires CoCa model)')
    parser.add_argument('--texts', type=str, nargs='+', default=None,
                       help='Optional: Generate text embeddings for these texts')
    parser.add_argument('--delimiter', type=str, default=' ',
                       help='Delimiter in dataset file (default: space)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_file, args.image_dir, delimiter=args.delimiter)
    print(f"Loaded {len(dataset)} images")
    
    # Load model
    print(f"Loading model: {args.model} with weights: {args.pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=args.device
    )
    model = model.to(args.device)
    model.eval()
    
    tokenizer = open_clip.get_tokenizer(args.model)
    
    # Generate image embeddings
    print("\nGenerating image embeddings...")
    image_embeddings = generate_image_embeddings(
        model, preprocess, dataset, 
        batch_size=args.batch_size, 
        device=args.device
    )
    
    # Save image embeddings
    image_emb_file = output_dir / 'image_embeddings.npz'
    save_embeddings(image_embeddings, image_emb_file, format='npz')
    
    # Generate text embeddings if texts provided
    if args.texts:
        print("\nGenerating text embeddings...")
        text_embeddings = generate_text_embeddings(
            model, tokenizer, args.texts,
            batch_size=args.batch_size,
            device=args.device
        )
        text_emb_file = output_dir / 'text_embeddings.npz'
        save_embeddings(text_embeddings, text_emb_file, format='npz')
    
    # Generate captions if requested
    if args.generate_captions:
        print("\nGenerating captions (this may take a while)...")
        # Check if model supports generation
        if not hasattr(model, 'generate'):
            print("Warning: Model does not support generation. Use a CoCa model (e.g., coca_ViT-L-14)")
        else:
            captions = generate_captions(
                model, preprocess, dataset,
                batch_size=1,  # Generation is done one at a time
                device=args.device
            )
            # Save captions
            captions_file = output_dir / 'generated_captions.json'
            with open(captions_file, 'w') as f:
                json.dump(captions, f, indent=2)
            print(f"Saved captions to {captions_file}")
    
    # Save metadata
    metadata = {
        'model': args.model,
        'pretrained': args.pretrained,
        'num_images': len(dataset),
        'embedding_dim': list(image_embeddings.values())[0].shape[0] if image_embeddings else None,
        'device': args.device
    }
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Done! Embeddings saved to {output_dir}")
    print(f"  - Image embeddings: {image_emb_file}")
    if args.texts:
        print(f"  - Text embeddings: {text_emb_file}")
    if args.generate_captions and hasattr(model, 'generate'):
        print(f"  - Generated captions: {output_dir / 'generated_captions.json'}")
    print(f"  - Metadata: {metadata_file}")


if __name__ == '__main__':
    main()
