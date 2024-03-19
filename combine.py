import cv2
import numpy as np

def combine_images(image_paths, output_path):
    # Load images
    images = [cv2.imread(path) for path in image_paths]

    # Ensure all images have the same dimensions
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    images = [cv2.resize(image, (max_width, max_height)) for image in images]

    # Create blank canvas for the comic
    comic = 255 *  np.ones((2 * max_height, 2 * max_width, 3), dtype=np.uint8)

    # Arrange images in a four-panel layout
    for i, image in enumerate(images):
        row = i // 2
        col = i % 2
        comic[row * max_height:(row + 1) * max_height, col * max_width:(col + 1) * max_width] = image

    # Write combined image to file
    cv2.imwrite(output_path, comic)

# Example usage
if __name__ == "__main__":
    image_paths = ['./output_folder/frame_3.jpg', './output_folder/frame_13.jpg', './output_folder/frame_23.jpg', './output_folder/frame_33.jpg']
    # image_paths = ['./images/keyframe_3.jpg', './images/keyframe_13.jpg', './images/keyframe_23.jpg', './images/keyframe_33.jpg']
    output_path = 'four_panel_comic.jpg'
    combine_images(image_paths, output_path)
