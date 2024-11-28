# Görkem Kadir Solun 22003214
# CS464 Machine Learning - Homework 2

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


### --- PCA --- PCA --- PCA --- PCA --- PCA --- PCA --- PCA --- PCA --- PCA --- PCA --- PCA
def pca():
    ## get the images

    # NOTE: Please change the path to the dataset folder
    resized_images_folder_path = os.path.join(os.getcwd(), "resized_images")
    resized_images = []
    for filename in os.listdir(resized_images_folder_path):
        img = Image.open(os.path.join(resized_images_folder_path, filename))
        # Convert image to numpy array as float32
        img = np.array(img, dtype=np.float32)
        # Flatten the image preserving the channel information
        img = img.reshape(-1, 3)
        resized_images.append(img)

    # Convert the list to numpy array
    resized_images = np.array(resized_images)

    # Get the each channel of the images
    red_channel = resized_images[:, :, 0]
    green_channel = resized_images[:, :, 1]
    blue_channel = resized_images[:, :, 2]
    rgb_channels = np.array([red_channel, green_channel, blue_channel])

    def calculate_channel_pca(channel):
        """
        Calculate PCA for the given not centered channel

        Parameters:
            channel: numpy array
                The channel of the images

        Returns:
            eigenvalues: numpy array
                The eigenvalues of the PCA
            eigen_vectors: numpy array
                The eigen vectors of the PCA
        """

        # Center the given channel
        channel_mean = np.mean(channel, axis=0)
        centered_channel = channel - channel_mean
        # Covariance matrix of the centered channel
        covariance_matrix = np.cov(centered_channel.T)
        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues, eigenvectors, channel_mean

    # Calculate the PCA for each channel, also get the means of the channels
    all_eigenvalues, all_eigenvectors, means = [], [], []
    for channel in rgb_channels:
        eigenvalues, eigenvectors, mean = calculate_channel_pca(channel)
        all_eigenvalues.append(eigenvalues)
        all_eigenvectors.append(eigenvectors)
        means.append(mean)

    ## PCA Q1
    # Calculate the proportion of the variance explained by each principal component
    def pve(eigenvalues, threshold=0.7, component_count=10):
        """
        Calculate the proportion of the variance explained by each principal component

        Parameters:
            eigenvalues: numpy array
                The eigenvalues of the PCA in descending order
            threshold: float
                The threshold to stop the calculation
            component_count: int
                The number of components to calculate the PVE

        Returns:
            first_n_pves: numpy array
                The PVE values of the given component count
            pve_sum: float
                The sum of the PVE values
            min_component_count: int
                The minimum component count to reach the threshold
        """

        # Calculate the PVE values
        pves = eigenvalues / np.sum(eigenvalues)
        # Consider the given component count
        first_n_pves = pves[:component_count]
        # Calculate the sum of the PVE values
        pve_sum = np.sum(first_n_pves)
        # Calculate the minimum component count to reach the threshold
        min_component_count = 0
        pvs_sum_for_threshold = 0
        for i in range(len(pves)):
            if pvs_sum_for_threshold < threshold:
                min_component_count += 1
                pvs_sum_for_threshold += pves[i]
            else:
                break

        return first_n_pves, pve_sum, min_component_count

    # Calculate the PVE values for each channel
    colors = ["Red", "Green", "Blue"]
    threshold = 0.7
    component_count = 10
    for index, eigenvalues in enumerate(all_eigenvalues):
        pve_values, pve_sum, min_component_count = pve(
            eigenvalues, threshold, component_count
        )
        print(f"{colors[index]} Channel")
        print(f"For the first {component_count} components:")
        print(f"PVE values: {pve_values}")
        print(f"Sum of PVE values: {pve_sum}")
        print(
            f"Minimum component count to reach the threshold {threshold}: {min_component_count}"
        )

    ## PCA Q2
    # Visualize top given number of principal components
    all_channel_images = []
    component_count = 10
    for i in range(component_count):
        channel_images = []
        for j, color in enumerate(colors):
            # Get the eigenvector of the given component
            eigenvector = all_eigenvectors[j][:, i]
            # Reshape the eigenvector to the image size
            eigenvector = eigenvector.reshape(64, 64)
            # Normalize it with min-max scaling
            eigenvector = (eigenvector - np.min(eigenvector)) / (
                np.max(eigenvector) - np.min(eigenvector)
            )
            channel_images.append(eigenvector)
        # Stack the channel images as RGB
        all_channel_image = np.stack(channel_images, axis=-1)
        all_channel_images.append(all_channel_image)

    # Plot the top given number of principal components
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(all_channel_images[i])
        ax.axis("off")
        ax.set_title(f"Principal Component {i + 1}")
    plt.tight_layout()
    plt.show()

    ## PCA Q3
    # Reconstruct the images with the given number of components
    first_image = resized_images[0].reshape(64, 64, 3) / 255
    component_counts = [1, 50, 250, 500, 1000, 4096]
    recreated_images_from_given_component_counts = (
        []
    )  # First image is the original image
    for component_count in component_counts:
        recreated_images = []
        for j, color in enumerate(colors):
            # Get the eigenvectors of the given component count
            eigenvectors = all_eigenvectors[j][:, :component_count]
            # Get the centered channel
            centered_channel = resized_images[0][:, j] - means[j]
            # Calculate the weights of the principal components
            weights = np.dot(centered_channel, eigenvectors)
            # Reconstruct the channel with the given component count
            reconstructed_channel = np.dot(weights, eigenvectors.T) + means[j]
            # Reshape the channel to the image size
            reconstructed_channel = reconstructed_channel.reshape(64, 64)
            recreated_images.append(reconstructed_channel)
        # Stack the channel images as RGB
        recreated_image = np.stack(recreated_images, axis=-1) / 255
        recreated_images_from_given_component_counts.append(recreated_image)

    # Plot the original image
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(first_image)
    ax.axis("off")
    ax.set_title("Original Image")

    # Plot the reconstructed images with the given number of components
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(recreated_images_from_given_component_counts[i])
        ax.axis("off")
        ax.set_title(f"Component Count: {component_counts[i]}")
    plt.tight_layout()
    plt.show()


### --- PCA END --- PCA END --- PCA END --- PCA END --- PCA END --- PCA END --- PCA END ---

### --- Logistic Regression --- Logistic Regression --- Logistic Regression --- Logistic Regression

### --- Logistic Regression END --- Logistic Regression END --- Logistic Regression END ---

### To run PCA press 1 and to run Logistic Regression press 2
is_pca = 1  # int(input("Enter 1 for PCA, 2 for Logistic Regression: "))

if is_pca == 1:
    print("Running PCA...")
    pca()
else:
    print("Running Logistic Regression...")
