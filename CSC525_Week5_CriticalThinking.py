"""
CSC525 Week 5 Critical Thinking Assignment Option #2: Image Dataset Augmentation
This script augments image file and saves original and augmented images in a folder.
Augmentations applied: brightness adjustment, contrast adjustment,
rotation, horizontal flip, and cropping.

Steps for running the program
- Install libraries and packages
- Place original image in the same directory as this script
- Run the script
- Review images in the "augmented_dataset" folder
- Close the plot window.
- Program ends.
"""
from PIL import Image, ImageEnhance
import random
from pathlib import Path
import shutil

def augment_image(cat):
    """computer vision preprocessing dataset augmentation"""
    augmentations = {}
    # Adjust brightness to a random level between 0.5 and 1.4
    augmentations["bright"] = ImageEnhance.Brightness(cat).enhance(random.uniform(0.5, 1.4))
    # Adjust contrast to a random level between 0.5 and 1.4
    augmentations["contrast"] = ImageEnhance.Contrast(cat).enhance(random.uniform(0.5, 1.4))
    # Rotate the image by a random angle between -45 and 45 degrees
    augmentations[f"rotated"] = cat.rotate(random.uniform(-45, 45), expand=True)
    # Flip the image
    augmentations["flipped"] = cat.transpose(Image.FLIP_LEFT_RIGHT)

    # Crop 80% randomly and resume original size
    w, h = cat.size
    crop_size = int(0.8 * min(w, h))
    left = random.randint(0, w - crop_size)
    top = random.randint(0, h - crop_size)
    cropped_img = cat.crop((left, top, left + crop_size, top + crop_size)).resize((w, h))
    augmentations["cropped"] = cropped_img

    return augmentations

def plot_images(original_img, augmented_images):
    """Comparesoriginal and augmented cat images."""
    import matplotlib.pyplot as plt
    images = [original_img] + list(augmented_images.values())
    titles = [
        "Original",
        "Brightness Adjusted",
        "Contrast Adjusted",
        "Rotated",
        "Horizontally Flipped",
        "Randomly Cropped 80%"
    ]

    plt.figure(figsize=(15, 6))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_images(original_path, augmented_images, output_dir):
    """Saves the original and augmented images to the output directory."""
    base_name = original_path.stem
    suffix = original_path.suffix

    # Save original image
    shutil.copy2(original_path, output_dir / original_path.name)

    # Save augmented images with descriptive names
    for aug_name, aug_img in augmented_images.items():
        output_filename = f"{base_name}_{aug_name}{suffix}"
        aug_img.save(output_dir / output_filename)

def main():
    current_dir = Path(".")
    output_dir = current_dir / "augmented_dataset"
    output_dir.mkdir(exist_ok=True)

    image_extensions = (".png", ".jpg", ".jpeg")
    for img_path in current_dir.iterdir(): # Process all cats and save them
        if img_path.name.lower().endswith(image_extensions) and img_path.is_file():
            with Image.open(img_path) as img:
                augmented_imgs = augment_image(img)
                save_images(img_path, augmented_imgs, output_dir)
    #show all cats
    print('Showing cat and augmentations...')
    plot_images(img, augmented_imgs)
    print('End of Week5 Critical Thinking')
if __name__ == "__main__":
    main()