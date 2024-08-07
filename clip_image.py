from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
import numpy as np
import time
import cv2
# add the following to avoid ssl issues from the server
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load your model and processor
# clip-vit-base-patch32, openai/clip-vit-large-patch14
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Load image from internet
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess the image for a single forward pass
start_time_processing = time.time()
inputs_single = processor(images=image, return_tensors="pt")
inputs_single = {k: v.to(device) for k, v in inputs_single.items()}
elapsed_time_processing = time.time() - start_time_processing

# Measure single image forward pass time
start_time_single = time.time()
with torch.no_grad():
    image_features_single = model.get_image_features(**inputs_single)
elapsed_time_single = time.time() - start_time_single

# Prepare a batch of N identical images
batch_size = 100
images_batch = [image for _ in range(batch_size)]
start_time_batch_processing = time.time()
inputs_batch = processor(images=images_batch, return_tensors="pt")
inputs_batch = {k: v.to(device) for k, v in inputs_batch.items()}
elapsed_time_batch_processing = time.time() - start_time_batch_processing


# Measure batch forward pass time
start_time_batch = time.time()
with torch.no_grad():
    image_features_batch = model.get_image_features(**inputs_batch)
elapsed_time_batch = time.time() - start_time_batch

print(f"Forward pass time for single image: {elapsed_time_single:.4f} seconds")
print(f"Forward pass time for batch of {batch_size} images: {elapsed_time_batch:.4f} seconds")
print(f"Processing time for single image: {elapsed_time_processing:.4f} seconds")
print(f"Processing time for batch image: {elapsed_time_batch_processing:.4f} seconds")

print()
print('Input single image shape:', np.array(image).shape)
print('Preprocessed single image shape:', inputs_single['pixel_values'].shape)
print('Output single image features:', image_features_single.shape)
print('Preprocessed batch image shape:', inputs_batch['pixel_values'].shape)
print('Output batch image features:', image_features_batch.shape)
