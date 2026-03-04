#!/usr/bin/env python3
"""
Example script showing how to use the embeddings after generation.
"""

import numpy as np
import json
from pathlib import Path


def load_embeddings(embeddings_file):
    """Load embeddings from NPZ file."""
    return np.load(embeddings_file)


def find_similar_images(query_embedding, image_embeddings, top_k=5):
    """
    Find most similar images to a query embedding.
    
    Args:
        query_embedding: Query embedding vector
        image_embeddings: Dictionary of image_filename -> embedding
        top_k: Number of top results to return
    
    Returns:
        List of (filename, similarity_score) tuples
    """
    similarities = []
    
    for filename, embedding in image_embeddings.items():
        # Compute cosine similarity
        similarity = np.dot(query_embedding, embedding)
        similarities.append((filename, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def search_by_text(text_embedding, image_embeddings, top_k=5):
    """
    Search images by text embedding.
    
    Args:
        text_embedding: Text embedding vector
        image_embeddings: Dictionary of image_filename -> embedding
        top_k: Number of top results to return
    
    Returns:
        List of (filename, similarity_score) tuples
    """
    return find_similar_images(text_embedding, image_embeddings, top_k)


def example_usage():
    """Example of how to use the generated embeddings."""
    
    # Load image embeddings
    print("Loading image embeddings...")
    image_emb_file = Path('./embeddings/image_embeddings.npz')
    
    if not image_emb_file.exists():
        print(f"Error: {image_emb_file} not found. Run generate_embeddings.py first.")
        return
    
    image_embeddings = load_embeddings(image_emb_file)
    
    # Convert to dictionary for easier access
    image_dict = {k: image_embeddings[k] for k in image_embeddings.files}
    
    print(f"Loaded {len(image_dict)} image embeddings")
    print(f"Embedding dimension: {image_dict[list(image_dict.keys())[0]].shape}")
    
    # Example 1: Find similar images
    print("\n=== Example 1: Find similar images ===")
    query_image = list(image_dict.keys())[0]  # Use first image as query
    query_embedding = image_dict[query_image]
    
    similar = find_similar_images(query_embedding, image_dict, top_k=5)
    print(f"\nMost similar images to {query_image}:")
    for filename, similarity in similar:
        print(f"  {filename}: {similarity:.4f}")
    
    # Example 2: Load text embeddings if available
    text_emb_file = Path('./embeddings/text_embeddings.npz')
    if text_emb_file.exists():
        print("\n=== Example 2: Search by text ===")
        text_embeddings = load_embeddings(text_emb_file)
        text_dict = {k: text_embeddings[k] for k in text_embeddings.files}
        
        # Search for first text query
        query_text = list(text_dict.keys())[0]
        query_text_emb = text_dict[query_text]
        
        results = search_by_text(query_text_emb, image_dict, top_k=5)
        print(f"\nImages most similar to '{query_text}':")
        for filename, similarity in results:
            print(f"  {filename}: {similarity:.4f}")
    
    # Example 3: Load generated captions if available
    captions_file = Path('./embeddings/generated_captions.json')
    if captions_file.exists():
        print("\n=== Example 3: Generated captions ===")
        with open(captions_file, 'r') as f:
            captions = json.load(f)
        
        # Show first 5 captions
        print("\nFirst 5 generated captions:")
        for i, (filename, caption) in enumerate(list(captions.items())[:5]):
            print(f"  {filename}: {caption}")
    
    # Example 4: Load metadata
    metadata_file = Path('./embeddings/metadata.json')
    if metadata_file.exists():
        print("\n=== Example 4: Metadata ===")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print("\nRun metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    example_usage()
