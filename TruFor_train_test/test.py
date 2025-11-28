import argparse
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms

# Import necessary modules from the project
from lib.config import config, update_config
from lib.utils import get_model

def load_and_evaluate_image(input_path,threshold=0.5):
    """
    Load a single image and evaluate it using the TruFor model.

    Args:
        input_path (str): Path to the input image.
        experiment (str): Experiment name for configuration.
        ckpt_path (str): Path to the model checkpoint file.
        threshold (float, optional): Threshold for determining manipulation. Defaults to 0.5.
    """
    # Define the argument parser
    args = argparse.Namespace(
        input=input_path,
        experiment='trufor_ph3',
        opts=None,
        ckpt= "pretrained_models/trufor.pth.tar"
    )

    # Update configuration and set up device
    update_config(config, args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the model
    print(f'Loading model from {args.ckpt}')
    checkpoint = torch.load(args.ckpt, map_location=device)
    model = get_model(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Load the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    image = Image.open(args.input).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Process the single image and output whether it is manipulated
    with torch.no_grad():
        pred, _, det, _ = model(image)

        if det is not None:
            manipulation_score = torch.sigmoid(det).item()
        else:
            pred = F.softmax(pred, dim=1)
            manipulation_score = pred[:, 1].item()  # Assuming the second class is manipulation

        if manipulation_score > threshold:
            print(f"Image {args.input} is likely manipulated (score: {manipulation_score:.4f})")
        else:
            print(f"Image {args.input} is likely not manipulated (score: {manipulation_score:.4f})")

# Example usage
if __name__ == "__main__":
    input_path = '/home/chenhui/MMFakeBench/LLaVA_2/data/MMFakeBench_val/real/fakeddit_val_50/fakeddit_val_2.png'
    load_and_evaluate_image(input_path)