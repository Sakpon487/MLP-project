# CLIP Inference Pipeline

Simple pipeline to generate CLIP embeddings and text descriptions for your dataset.

## Features

- ✅ Generate image embeddings
- ✅ Generate text embeddings
- ✅ Generate text captions using CoCa models
- ✅ Batch processing for efficiency
- ✅ Save embeddings in multiple formats (NPZ, JSON)

## Installation

```bash
pip install open-clip-torch torch torchvision pillow tqdm numpy
```

## Quick Start

### Basic Usage: Generate Image Embeddings

```bash
python generate_embeddings.py \
    --dataset-file /path/to/your/dataset.txt \
    --image-dir /path/to/images \
    --output-dir ./embeddings
```

### With Custom Model

```bash
python generate_embeddings.py \
    --dataset-file /path/to/your/dataset.txt \
    --image-dir /path/to/images \
    --model ViT-L-14 \
    --pretrained laion2b_s32b_b82k \
    --output-dir ./embeddings
```

### Generate Text Embeddings

```bash
python generate_embeddings.py \
    --dataset-file /path/to/your/dataset.txt \
    --image-dir /path/to/images \
    --texts "a photo of a cat" "a photo of a dog" "a photo of a bird" \
    --output-dir ./embeddings
```

### Generate Captions (CoCa Model)

```bash
python generate_embeddings.py \
    --dataset-file /path/to/your/dataset.txt \
    --image-dir /path/to/images \
    --model coca_ViT-L-14 \
    --pretrained mscoco_finetuned_laion2B-s13B-b90k \
    --generate-captions \
    --output-dir ./embeddings
```

## Inference evaluation: recall, distribution, t-SNE

After generating embeddings (and super-class labels), run evaluation and visualisations:

```bash
python run_inference.py \
    --embeddings /path/to/embeddings.npy \
    --labels /path/to/super_class_ids.npy \
    --output-dir ./inference_output
```

This script:

- **Rank-X recall**: Computes Recall@1, @5, @10, @100, @1000 (configurable via `--rank-k`).
- **Match vs non-match distribution**: Plots similarity histograms for same-class vs different-class pairs; saved as `match_vs_nonmatch_distribution.png`.
- **t-SNE**: 2D embedding plot coloured by super-class; saved as `tsne_embeddings.png`. By default uses 5000 points (`--tsne-subsample`); use `--tsne-subsample 0` to use all (slow).

Options: `--skip-tsne`, `--skip-distribution` to skip slow steps; `--block-size`, `--batch-size` for memory tuning.

## Dataset Format

The script expects a text file with the following format:

```
category_id image_filename
1 111085122871_0.JPG
1 111085122871_1.JPG
2 111265328556_0.JPG
```

- First column: Category ID (can be any identifier)
- Second column: Image filename
- Delimiter: Space (can be changed with `--delimiter`)

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-file` | Path to dataset file | Required |
| `--image-dir` | Directory containing images | Required |
| `--model` | Model architecture | `ViT-B-32` |
| `--pretrained` | Pretrained weights | `laion2b_s34b_b79k` |
| `--output-dir` | Output directory | `./embeddings` |
| `--batch-size` | Batch size | `32` |
| `--device` | Device (cuda/cpu) | Auto-detect |
| `--generate-captions` | Generate captions with CoCa | False |
| `--texts` | Generate text embeddings | None |
| `--delimiter` | Dataset file delimiter | Space |

## Output Files

The script generates:

1. **`image_embeddings.npz`** - Image embeddings (numpy compressed format)
2. **`text_embeddings.npz`** - Text embeddings (if `--texts` provided)
3. **`generated_captions.json`** - Generated captions (if `--generate-captions`)
4. **`metadata.json`** - Metadata about the run

## Loading Embeddings

### Python

```python
import numpy as np

# Load image embeddings
embeddings = np.load('embeddings/image_embeddings.npz')
image_emb = embeddings['111085122871_0.JPG']

# Load text embeddings
text_embeddings = np.load('embeddings/text_embeddings.npz')
text_emb = text_embeddings['a photo of a cat']

# Compute similarity
similarity = np.dot(image_emb, text_emb)
```

### JSON Format

```python
import json

with open('embeddings/generated_captions.json', 'r') as f:
    captions = json.load(f)
    
print(captions['111085122871_0.JPG'])
```

## Example: Using with Your Dataset

For the Ebay dataset format:

```bash
python generate_embeddings.py \
    --dataset-file /Users/boud/mlpractical/final_project/open_clip/SOP/.labels/Ebay_final_with_catid.txt \
    --image-dir /path/to/ebay/images \
    --model ViT-B-32 \
    --pretrained laion2b_s34b_b79k \
    --batch-size 64 \
    --output-dir ./ebay_embeddings
```

## Performance Tips

1. **Batch Size**: Increase `--batch-size` for faster processing (limited by GPU memory)
2. **Device**: Use `--device cuda` if you have a GPU
3. **Model Selection**: Smaller models (ViT-B-32) are faster, larger models (ViT-L-14) are more accurate
4. **Caption Generation**: This is slow - consider processing a subset first

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Use a smaller model (ViT-B-32 instead of ViT-L-14)

### Images Not Found
- Check that `--image-dir` points to the correct directory
- Verify image filenames match exactly (case-sensitive)

### Model Not Found
- Check available models: `python -c "import open_clip; print(open_clip.list_models())"`
- Check available pretrained weights: `python -c "import open_clip; print(open_clip.list_pretrained_tags_by_model('ViT-B-32'))"`
