# VLM Efficiency Analysis

This repository details an efficiency analysis conducted using various CLIP model sizes under different processing functions and input types, specifically tailored for a Tesla V100 GPU with 16 GB VRAM.

## Analysis Overview

### Single Image Processing

- **Models Tested:** ViT-B and ViT-L
- **Input Dimensions:** 224x224 pixels (standard for CLIP)
- **Observations:** When processing a single image, there is negligible difference in the forward pass times between the ViT-B and ViT-L models.

### Batch Image Processing

- **Batch Size:** 500 images
- **Processing Time:** Approximately 7 seconds for both models without optimization.
- **Forward pass Time:** Notable differences emerge when doing a forward pass in batches. The ViT-L model demonstrates a significant reduction in forward pass time compared (~6.6segs) to ViT-B (~0.4segs).

### Custom Processor Function

- **Implementation:** A hand-crafted `processor` function was developed to enhance efficiency.
- **Impact:** This custom processor reduces the image processing time drastically—from approximately 7 seconds to about 0.3 seconds.
- **Model Performance:** The improvement is especially pronounced with the ViT-L model, where the ~6.6segs decrease up to 0.3 seconds.

### Numpy Array Inputs

- **Experiment:** Instead of standard images, inputs were created as numpy arrays.
- **Standard Processing Time:** Using numpy arrays reduces the naive processing time to 2.3 seconds (instead of the initial ~7seconds).
- **With Custom Processor:** Applying the custom processor function further reduces this time to 0.3 seconds, underscoring the effectiveness of the processing optimization across different input methods.

## Conclusion

The custom processor function significantly enhances processing efficiency for CLIP models, particularly evident when handling large batches and alternative input formats such as numpy arrays. 
