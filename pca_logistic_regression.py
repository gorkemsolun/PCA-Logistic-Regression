# Görkem Kadir Solun 22003214
# CS464 Machine Learning - Homework 2

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# NOTE: Change the random seed for reproducibility
random_seed = 14


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
def logistic_regression():
    def one_hot_encoding(label_data):
        row_number = label_data.shape[0]
        num_labels = 10  # MNIST dataset (0-9)
        encoded_labels = np.zeros((row_number, num_labels), dtype="int")
        encoded_labels[list(range(row_number)), label_data] = 1
        return encoded_labels

    def read_pixels(data_path):
        with open(data_path, "rb") as f:
            pixel_data = np.frombuffer(f.read(), "B", offset=16).astype("float32")
        flattened_pixels = pixel_data.reshape(-1, 784)
        normalized_pixels = flattened_pixels / 255
        return normalized_pixels

    def read_labels(data_path):
        with open(data_path, "rb") as f:
            label_data = np.frombuffer(f.read(), "B", offset=8)
        one_hot_encoding_labels = one_hot_encoding(label_data)
        return one_hot_encoding_labels

    def read_dataset(path):
        X_train = read_pixels(path + "/train-images-idx3-ubyte")
        y_train = read_labels(path + "/train-labels-idx1-ubyte")
        X_test = read_pixels(path + "/t10k-images-idx3-ubyte")
        y_test = read_labels(path + "/t10k-labels-idx1-ubyte")
        return X_train, y_train, X_test, y_test

    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def prepare_confusion_matrix(y_true, y_pred, number_of_classes):
        confusion_matrix = np.zeros((number_of_classes, number_of_classes))
        for true_label, predicted_label in zip(y_true, y_pred):
            confusion_matrix[true_label][predicted_label] += 1

        # Display the confusion matrix with pandas
        print(pd.DataFrame(confusion_matrix))

        # Plot the confusion matrix
        plt.imshow(confusion_matrix, cmap=plt.cm.Blues, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(range(number_of_classes))
        plt.yticks(range(number_of_classes))

        # Add the values to the cells
        for i in range(number_of_classes):
            for j in range(number_of_classes):
                plt.text(
                    j,
                    i,
                    confusion_matrix[i, j],
                    ha="center",
                    va="center",
                    color="black",
                )

        plt.show()

        return confusion_matrix

    # NOTE: Please change the path to the dataset folder
    dataset_path = os.path.join(os.getcwd(), "mnist_dataset")
    X_train, y_train, X_test, y_test = read_dataset(dataset_path)
    # Now separate train as validation and train, 50000 for train and 10000 for validation
    X_validation = X_train[50000:]
    y_validation = y_train[50000:]
    X_train = X_train[:50000]
    y_train = y_train[:50000]

    ## Logistic Regression Q1
    print("Logistic Regression Q1")

    # Define the initial parameters for the logistic regression
    learning_rate = 5e-4
    l2_regularization_coefficient = 1e-4
    batch_size = 200
    epochs = 100
    number_of_classes = 10

    def multinomial_logistic_regression_train(weights):
        np.random.seed(random_seed)

        # Accuracy and loss values for each epoch
        validation_accuracy_values = []
        # validation_accuracy_losses = []

        for epoch in range(epochs):
            # Shuffle the training data
            """permutation = np.random.permutation(X_train.shape[0])
            X_train = X_train[permutation]
            y_train = y_train[permutation]"""

            # Training the model with mini-batch gradient descent
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                # Calculate the softmax
                y_predicted = softmax(np.dot(X_batch, weights))

                """ # Calculate the loss
                loss = -np.sum(y_batch * np.log(y_predicted)) / batch_size
                # Add the L2 regularization term to the loss
                loss += l2_regularization_coefficient * np.sum(weights**2) """

                # Calculate the gradient
                gradient = np.dot(X_batch.T, y_predicted - y_batch) / batch_size
                # Add the L2 regularization term to the gradient
                gradient += l2_regularization_coefficient * weights

                # Update the weights
                weights -= learning_rate * gradient

            # Calculate the validation accuracy
            y_validation_pred = softmax(np.dot(X_validation, weights))
            validation_accuracy = np.mean(
                np.argmax(y_validation, axis=1) == np.argmax(y_validation_pred, axis=1)
            )
            validation_accuracy_values.append(validation_accuracy)

            print(f"Epoch: {epoch + 1}, Validation Accuracy: {validation_accuracy}")

        return weights, validation_accuracy_values

    def multinomial_logistic_regression_predict(weights, X):
        return np.argmax(softmax(np.dot(X, weights)), axis=1)

    weights, validation_accuracy_values = multinomial_logistic_regression_train(
        np.random.normal(0, 1, (X_train.shape[1], number_of_classes))
    )

    # Calculate the test accuracy
    y_test_pred = multinomial_logistic_regression_predict(weights, X_test)
    y_test_true = np.argmax(y_test, axis=1)
    test_accuracy = np.mean(y_test_true == y_test_pred)
    print(f"Test Accuracy: {test_accuracy}")
    # Display the confusion matrix with pandas
    confusion_matrix_test = prepare_confusion_matrix(
        y_test_true, y_test_pred, number_of_classes
    )

    ## Logistic Regression Q2
    print("Logistic Regression Q2")

    default_learning_rate = learning_rate
    default_l2_regularization_coefficient = l2_regularization_coefficient
    default_batch_size = batch_size
    # default_initialization = "normal"

    parameters_2_test = {
        "batch_size": [1, 64, 3000],
        "learning_rate": [1e-2, 1e-3, 1e-4, 1e-5],
        "l2_regularization_coefficient": [1e-2, 1e-4, 1e-9],
        "initialization": ["normal", "uniform", "zeros"],
    }

    best_parameters = {}
    results_4_each_parameter = {}
    for parameter_name, parameter_values in parameters_2_test.items():
        best_accuracy = 0
        epoch_validation_accuracies = []

        for parameter_value in parameter_values:
            # Reset the parameters to the default values
            batch_size = default_batch_size
            learning_rate = default_learning_rate
            l2_regularization_coefficient = default_l2_regularization_coefficient
            weights = np.random.normal(0, 1, (X_train.shape[1], number_of_classes))

            # Update the parameter value
            if parameter_name == "batch_size":
                batch_size = parameter_value
            elif parameter_name == "learning_rate":
                learning_rate = parameter_value
            elif parameter_name == "l2_regularization_coefficient":
                l2_regularization_coefficient = parameter_value
            elif parameter_name == "initialization":
                # Default initialization is normal skip it
                if parameter_value == "uniform":
                    weights = np.random.uniform(
                        0, 1, (X_train.shape[1], number_of_classes)
                    )
                elif parameter_value == "zeros":
                    weights = np.zeros((X_train.shape[1], number_of_classes))

            weights, validation_accuracy_values = multinomial_logistic_regression_train(
                weights
            )

            # Update the best parameters if the accuracy is better
            if validation_accuracy_values[-1] > best_accuracy:
                best_accuracy = validation_accuracy_values[-1]
                best_parameters[parameter_name] = parameter_value

            epoch_validation_accuracies.append(validation_accuracy_values)

        results_4_each_parameter[parameter_name] = (
            parameter_values,
            epoch_validation_accuracies,
        )

    # Print the best parameters
    print("Best Parameters:")
    for parameter_name, parameter_value in best_parameters.items():
        print(f"{parameter_name}: {parameter_value}")

    # Plot the validation accuracies for each parameter
    fig, axs = plt.subplots(
        2,
        2,
    )
    for i, (parameter_name, parameter_accuracies) in enumerate(
        results_4_each_parameter.items()
    ):
        parameter_values, epoch_validation_accuracies = parameter_accuracies
        ax = axs.flat[i]
        for j, validation_accuracies in enumerate(epoch_validation_accuracies):
            ax.plot(validation_accuracies, label=f"{parameter_values[j]}")
        ax.set_title(f"{parameter_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.legend()
    plt.tight_layout()
    plt.show()

    ## Logistic Regression Q3
    print("Logistic Regression Q3")

    # Train the model with the best parameters
    batch_size = best_parameters["batch_size"]
    learning_rate = best_parameters["learning_rate"]
    l2_regularization_coefficient = best_parameters["l2_regularization_coefficient"]
    weights = np.random.normal(0, 1, (X_train.shape[1], number_of_classes))
    if best_parameters["initialization"] == "uniform":
        weights = np.random.uniform(0, 1, (X_train.shape[1], number_of_classes))
    elif best_parameters["initialization"] == "zeros":
        weights = np.zeros((X_train.shape[1], number_of_classes))

    weights, validation_accuracy_values = multinomial_logistic_regression_train(weights)
    # Calculate the test accuracy
    y_test_pred = multinomial_logistic_regression_predict(weights, X_test)
    y_test_true = np.argmax(y_test, axis=1)
    test_accuracy = np.mean(y_test_true == y_test_pred)
    print(f"Test Accuracy: {test_accuracy}")
    # Display the confusion matrix with pandas
    confusion_matrix_test = prepare_confusion_matrix(
        y_test_true, y_test_pred, number_of_classes
    )

    ## Logistic Regression Q4
    print("Logistic Regression Q4")

    # Display the weights as images
    for i in range(number_of_classes):
        image_weight = weights[:, i].reshape(28, 28)
        plt.matshow(
            image_weight,
            cmap=plt.cm.gray,
            vmin=0.5 * image_weight.min(),
            vmax=0.5 * image_weight.max(),
        )
        plt.title(f"Weight for Class {i}")
        plt.colorbar()
        plt.show()

    ## Logistic Regression Q5
    print("Logistic Regression Q5")

    # Display the metrics for the best model
    # Calculate the precision, recall, F1 score, and F2 score for each class
    precision = np.zeros(number_of_classes)
    recall = np.zeros(number_of_classes)
    f1 = np.zeros(number_of_classes)
    f2 = np.zeros(number_of_classes)

    for i in range(number_of_classes):
        true_positive = confusion_matrix_test[i, i]
        false_positive = np.sum(confusion_matrix_test[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix_test[i, :]) - true_positive

        precision[i] = true_positive / (true_positive + false_positive)
        recall[i] = true_positive / (true_positive + false_negative)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        f2[i] = 5 * precision[i] * recall[i] / (4 * precision[i] + recall[i])

    # Display the metrics
    metrics = pd.DataFrame(
        {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "F2 Score": f2,
        }
    )
    print("Metrics for the Best Model:")
    print(metrics)


### --- Logistic Regression END --- Logistic Regression END --- Logistic Regression END ---

### To run PCA press 1 and to run Logistic Regression press 2
is_pca = int(input("Enter 1 for PCA, 2 for Logistic Regression: "))

if is_pca == 1:
    print("Running PCA...")
    pca()
else:
    print("Running Logistic Regression...")
    logistic_regression()
