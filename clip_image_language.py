from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
# add the following to avoid ssl issues from the server
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load your model and processor
# clip-vit-base-patch32, openai/clip-vit-large-patch14
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load image from internet
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

print('\n****Preprocessing...')
# Preprocess the image and text
text = "A photo of a cat"
inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device as the model

# Pass the image through the visual encoder of CLIP
print('\n****Forward pass...')
outputs = model(**inputs)
image_features = outputs.image_embeds
text_features = outputs.text_embeds

# Now image_features contains the latent representation of your image
print('Image features:',image_features.shape)
# Concatenate image and text features
concatenated_features = torch.cat((image_features, text_features), dim=1)

# Now concatenated_features contains the combined latent representation of your image and text
print('Concatenated features shape:', concatenated_features.shape)

