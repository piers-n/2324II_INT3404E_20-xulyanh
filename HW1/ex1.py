import cv2 
import matplotlib.pyplot as plt
import numpy as np

# Load an image from file as function
def load_image(image_path):
    return cv2.imread(image_path)

# Display an image as function
def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
    pass


# grayscale an image as function
def grayscale_image(image):
    height, width = image.shape[:2]
    img_gray = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
    
            B = image[i, j, 0]
            G = image[i, j, 1]
            R = image[i, j, 2]
            
            gray_value = 0.299 * R + 0.587 * G + 0.114 * B

            img_gray[i, j] = gray_value

    return img_gray


# Save an image as function
def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    pass


# flip an image as function 
def flip_image(image):
    return cv2.flip(image, 1)


# rotate an image as function
def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


if __name__ == "__main__":
    # Load an image from file
    img = load_image("images/uet.png")

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "images/lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Save the flipped grayscale image
    save_image(img_gray_flipped, "images/lena_gray_flipped.jpg")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/lena_gray_rotated.jpg")

    # Show the images
    plt.show() 
