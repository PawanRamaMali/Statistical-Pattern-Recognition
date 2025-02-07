import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to extract 24-dimensional color histogram features from a single patch and plot the histograms
def extract_patch_histogram(patch, bins=(8, 8, 8), plot_histogram=False, patch_num=None):
    # Extract 8-bin color histograms for R, G, B channels
    hist_r = cv2.calcHist([patch], [0], None, [bins[0]], [0, 256]).flatten()  # Red channel
    hist_g = cv2.calcHist([patch], [1], None, [bins[1]], [0, 256]).flatten()  # Green channel
    hist_b = cv2.calcHist([patch], [2], None, [bins[2]], [0, 256]).flatten()  # Blue channel
    
    # Normalize histograms
    hist_r /= np.sum(hist_r)
    hist_g /= np.sum(hist_g)
    hist_b /= np.sum(hist_b)
    
    # Concatenate R, G, B histograms into a 24-dimensional vector
    feature_vector = np.concatenate([hist_r, hist_g, hist_b])
    
    # Plot the histogram for the patch if requested
    if plot_histogram:
        plot_color_histogram(hist_r, hist_g, hist_b, patch_num)
    
    return feature_vector

# Function to plot the color histograms for the patch
def plot_color_histogram(hist_r, hist_g, hist_b, patch_num):
    plt.figure(figsize=(8, 4))
    plt.suptitle(f'Patch {patch_num} - Color Histograms', fontsize=16)

    plt.subplot(1, 3, 1)
    plt.plot(hist_r, color='r')
    plt.title('Red Channel')
    plt.xlim([0, 8])

    plt.subplot(1, 3, 2)
    plt.plot(hist_g, color='g')
    plt.title('Green Channel')
    plt.xlim([0, 8])

    plt.subplot(1, 3, 3)
    plt.plot(hist_b, color='b')
    plt.title('Blue Channel')
    plt.xlim([0, 8])

    plt.tight_layout()
    plt.show()

# Function to divide image into 32x32 patches, extract features from each patch, and optionally plot histograms
def extract_image_features(image, patch_size=(32, 32), bins=(8, 8, 8), plot_histograms=False):
    img_h, img_w, _ = image.shape
    all_patches_features = []
    
    patch_num = 1  # To track the patch number for plotting
    for i in range(0, img_h, patch_size[0]):
        for j in range(0, img_w, patch_size[1]):
            # Extract a 32x32 patch
            patch = image[i:i+patch_size[0], j:j+patch_size[1]]
            if patch.shape[0] == patch_size[0] and patch.shape[1] == patch_size[1]:
                # Extract 24-dimensional feature vector for the patch
                patch_features = extract_patch_histogram(patch, bins, plot_histograms, patch_num)
                all_patches_features.append(patch_features)
                patch_num += 1  # Increment patch number

    # Stack all patch feature vectors for the image
    return np.array(all_patches_features)

# Function to save feature vectors to a file
def save_feature_vectors(feature_vectors, save_path):
    np.save(save_path, feature_vectors)  # Save as .npy file for efficient storage

# Main function to process a single image and extract features
def process_single_image(image_path, output_folder, plot_histograms=False):
    # Load the image
    image = cv2.imread(image_path)
    
    # Ensure image is valid
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Extract 24-dimensional feature vectors from 32x32 patches and optionally plot histograms
    features = extract_image_features(image, plot_histograms=plot_histograms)
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the feature vectors to a .npy file
    filename = os.path.basename(image_path).split('.')[0] + '_features.npy'
    save_path = os.path.join(output_folder, filename)
    save_feature_vectors(features, save_path)
    
    print(f"Feature vectors saved for {image_path} at {save_path}")
    
    return features

# Display image with patches for visualization
def display_image_with_patches(image, patch_size=(32, 32)):
    img_h, img_w, _ = image.shape
    patches = []
    for i in range(0, img_h, patch_size[0]):
        for j in range(0, img_w, patch_size[1]):
            patch = image[i:i+patch_size[0], j+j+patch_size[1]]
            if patch.shape[0] == patch_size[0] and patch.shape[1] == patch_size[1]:
                patches.append(patch)

    plt.figure(figsize=(8, 8))
    plt.suptitle("Image Patches")
    for i, patch in enumerate(patches[:16]):  # Display first 16 patches
        plt.subplot(4, 4, i+1)
        plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Example usage
def main():
    # Path to a single image
    image_path = r'group03-2/train/botanical_garden/sun_aamlzecjkxlnoedl.jpg'  # Update this with the actual image path
    output_folder = r'output_folder'  # Folder where feature files will be saved

    # Process the image and extract features, plot histograms of patches
    features = process_single_image(image_path, output_folder, plot_histograms=True)
    
    # Visualize the image and patches for verification
    image = cv2.imread(image_path)
    display_image_with_patches(image)

if __name__ == '__main__':
    main()
