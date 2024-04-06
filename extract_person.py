import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms


def load_and_preprocess_image(image_path):
    """Load an image and apply preprocessing required for Deeplab."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(513),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return img, preprocess(img_rgb).unsqueeze(0)


def segment_person(deeplab_model, img_tensor, device):
    """Segment the person from the image using the deeplab model."""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = deeplab_model(img_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    person_mask = (output_predictions == 15)  # 15 is typically the class for person
    return person_mask


def apply_mask_to_image(original_img, mask):
    """Apply the segmentation mask to the original image, resizing the mask as needed."""
    # Resize mask to match original image dimensions
    resized_mask = cv2.resize(mask.astype(np.uint8), (original_img.shape[1], original_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # Convert the resized mask to a 3 channel image
    mask_3d = np.repeat(resized_mask[:, :, np.newaxis], 3, axis=2)

    # Use the mask to select the person, setting background to white or any desired color
    person_image = np.where(mask_3d, original_img, 255)  # 255 for white background
    return person_image


# Initialize Deeplab model
device = torch.device("mps")
deeplab = deeplabv3_resnet101(pretrained=True).to(device)
deeplab.eval()

# Load and preprocess image
image_path = "extracted_linus/corrected.png"  # Update with your image path
original_img, img_tensor = load_and_preprocess_image(image_path)

# Perform segmentation to get person mask
person_mask = segment_person(deeplab, img_tensor, device)

# Apply mask to the original image
person_image = apply_mask_to_image(original_img, person_mask)
# Optionally, save the result
output_filename = image_path.replace('.png', '_segmented.png')
cv2.imwrite(output_filename, person_image)
