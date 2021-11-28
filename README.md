# Leveraging information in spatial transcriptomics to predict super-resolution gene expression from histology images in tumors
### Minxing Pang, Kenong Su*, Mingyao Li*
HisToGene is a deep learning method that predicts super-resolution gene expression from histology images in tumors. Trained in a spatial transcriptomics dataset, HisToGene models the spatial dependency in gene expression and histological features among spots through a modified Vision Transformer model.
<img src="Workflow.PNG" width="800px"></img>
# Usage
```python
import torch
from vis_model import HisToGene

model = HisToGene(
    n_genes=1000, 
    patch_size=112, 
    n_layers=4, 
    dim=1024, 
    learning_rate=1e-5, 
    dropout=0.1, 
    n_pos=64
)

# flatten_patches: [N, 3*W*H]
# coordinates: [N, 2]

pred_expression = model(flatten_patches, coordinates)  # [N, n_genes]

```

## System environment
Required package:
- PyTorch >= 1.8
- pytorch-lightening >= 1.4
- scanpy >= 1.8

## Parameters
- `n_genes`: int.  
  Amount of genes.
- `patch_size`: int.  
  Width/diameter of the spots.
- `n_layers`: int, default `4`.  
  Number of Transformer blocks.
- `dim`: int.  
  Dimension of the embeddings.
- `learning_rate`: float between `[0, 1]`, default `1e-5`.  
  Learning rate.
- `dropout`: float between `[0, 1]`, default `0.1`.  
  Dropout rate in the Transformer.
- `n_pos`: int, default `64`.  
   Maximum number of the coordinates.

# HisToGene pipeline
See [tutorial.ipynb](tutorial.ipynb)

# References

