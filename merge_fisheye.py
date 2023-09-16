# Implemented by Chat-GPT3.5

import torch
import torchvision.transforms.functional as TF


def merge_fisheye_images(left_image, right_image):
    # Resize images to have the same height
    height = min(left_image.size(1), right_image.size(1))
    left_image = TF.resize(left_image, (height, height))
    right_image = TF.resize(right_image, (height, height))

    # Convert images to tensors
    left_tensor = TF.to_tensor(left_image)
    right_tensor = TF.to_tensor(right_image)

    # Scale images between -1 and 1
    left_tensor = left_tensor * 2 - 1
    right_tensor = right_tensor * 2 - 1

    # Create a canvas for the merged image
    canvas = torch.zeros(left_tensor.size())

    # Set the left half of the canvas
    canvas[:, : canvas.size(1) // 2, :] = left_tensor[:, : canvas.size(1) // 2, :]

    # Set the right half of the canvas
    right_offset = canvas.size(1) // 2
    canvas[:, right_offset:, :] = right_tensor[:, right_offset:, :]

    # Scale the canvas between 0 and 1
    canvas = (canvas + 1) / 2

    # Convert the canvas tensor to an image
    merged_image = TF.to_pil_image(canvas)

    return merged_image


if __name__ == "__main__":
    # Assuming you have two fisheye images in PIL format: left_image and right_image
    merged_image = merge_fisheye_images(left_image, right_image)

    # Save the merged image
    merged_image.save("merged_image.png")
