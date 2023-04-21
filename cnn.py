from ckplus_loader import load_ckplus_data
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_curve, auc


class ConvLayer:
    # A Convolution layer using 3x3 filters.

    def __init__(self, total_filters):
        """
        Initializes a ConvLayer object with the specified number of filters.

        Args:
        total_filters: The number of filters to use in the convolutional layer.

        Returns:
        None
        """
        # Store the number of filters as an instance variable.
        self.total_filters = total_filters

        # Initialize the filters as a 3D array of random values with shape (total_filters, 3, 3).
        # Divide the values by 9 to reduce their magnitude and prevent numerical instability.
        self.filters = np.random.randn(total_filters, 3, 3) / 9

    def iterate_over_image(self, image):
        """
        Iterates over all 3x3 regions in the input image, yielding each region along with its coordinates in the image.

        Args:
        image (numpy.ndarray): A 2D array representing the input image.

        Yields:
        A tuple containing:
            - iterate_region: A 3x3 subregion of the input image, centered at the current iteration position.
            - i (int): The row index of the top-left corner of the subregion in the input image.
            - j (int): The column index of the top-left corner of the subregion in the input image.
        """
        # Get the height and width of the input image.
        height, width = image.shape

        # Iterate over all possible 3x3 regions in the image, skipping the last two rows and columns.
        for i in range(height - 2):
            for j in range(width - 2):
                # Extract the 3x3 region centered at the current iteration position.
                iterate_region = image[i:(i + 3), j:(j + 3)]
                # Yield the region along with its coordinates in the image.
                yield iterate_region, i, j

    def forward(self, input):

        self.previous_input = input

        height, weight = input.shape
        output = np.zeros((height - 2, weight - 2, self.total_filters))

        for iterate_region, x, y in self.iterate_over_image(input):
            output_sum = np.sum(iterate_region * self.filters, axis=(1, 2))
            output[x, y] = output_sum

        return output

    def backpropagation(self, loss_gradient, learn_rate):

        """
            Computes the gradient of the loss function with respect to the filters in the convolutional layer,
            and updates the filters using gradient descent.

            Args:
            loss_gradient: The gradient of the loss function with respect to the output of this layer.
            learn_rate: The learning rate used for updating the filters.

            Returns:
            None, because we use Convolutional layer as the first layer
            """

        # Initialize a matrix to store the gradient of the loss function with respect to each filter in the layer.
        loss_gradient_filters = np.zeros(self.filters.shape)

        # Iterate over the input image, computing the contribution of each pixel to the gradient of the loss function
        # with respect to the filters.
        for iterate_region, i, j in self.iterate_over_image(self.previous_input):
            # Iterate over each filter in the layer.
            for k in range(self.total_filters):
                # Compute the contribution of this pixel to the gradient of the loss function with respect to this
                # filter.
                loss_gradient_filters[k] += loss_gradient[i, j, k] * iterate_region

        # Update the filters using gradient descent.
        self.filters -= learn_rate * loss_gradient_filters

        return None


class Max_pool:
    # A Max Pooling layer using a pool size of 2.

    def iterate_over_image(self, image):
        """
        Iterates over all 2x2 regions in the input image, yielding each region along with its coordinates in the image.

        Args:
        image: A 3D array representing the input image.

        Yields:
        A tuple containing:
            - iterate_region (numpy.ndarray): A 2x2 subregion of the input image, starting at the current iteration position.
            - h: The row index of the top-left corner of the subregion in the input image.
            - w: The column index of the top-left corner of the subregion in the input image.
        """
        # Get the height, width, and number of channels of the input image.
        height, width, _ = image.shape

        # Compute the new height and width of the downsampled image (ignoring the last row and column if they are
        # incomplete).
        new_height = height // 2
        new_width = width // 2

        # Iterate over all possible 2x2 regions in the image.
        for h in range(new_height):
            for w in range(new_width):
                # Extract the 2x2 region starting at the current iteration position.
                iterate_region = image[(h * 2):(h * 2 + 2), (w * 2):(w * 2 + 2)]
                # Yield the region along with its coordinates in the image.
                yield iterate_region, h, w

    def forward(self, input):
        """
        Performs forward propagation for the MaxPoolLayer, downsampling the input image by a factor of 2.

        Args:
        input: A 3D array representing the input image, with dimensions (height, width, total_filters).

        Returns:
        output: A 3D array representing the downsampled output image, with dimensions (height/2, width/2, total_filters).
        """
        # Store the input as an instance variable.
        self.previous_input = input

        # Get the height, width, and number of filters of the input image.
        height, width, total_filters = input.shape

        # Initialize the output array as a 3D array of zeros with dimensions (height/2, width/2, total_filters).
        output = np.zeros((height // 2, width // 2, total_filters))

        # Iterate over all possible 2x2 regions in the input image.
        for iterate_region, i, j in self.iterate_over_image(input):
            # Compute the maximum value over each channel in the 2x2 region, and store it in the corresponding position
            # in the output array.
            output[i, j] = np.amax(iterate_region, axis=(0, 1))

        # Return the downsampled output image.
        return output

    def backpropagation(self, loss_gradient):
        """
        Computes the gradient of the loss function with respect to the inputs of the MaxPoolLayer, using the stored input
        and output from the forward pass, and the given loss gradient.

        Args:
        loss_gradient: A 3D array representing the gradient of the loss function with respect to the output
        of this layer, with dimensions (height/2, width/2, total_filters).

        Returns:
        loss_gradient_input: A 3D array representing the gradient of the loss function with respect to the
        inputs of this layer, with dimensions (height, width, total_filters).
        """
        # Initialize an array to store the gradient of the loss function with respect to the inputs of this layer.
        loss_gradient_input = np.zeros(self.previous_input.shape)

        # Iterate over all possible 2x2 regions in the input image.
        for iterate_region, i, j in self.iterate_over_image(self.previous_input):
            # Get the height, width, and number of filters of the iterate_region.
            height, weight, f = iterate_region.shape

            # Compute the maximum value over each channel in the 2x2 region.
            amax = np.amax(iterate_region, axis=(0, 1))

            # Iterate over each pixel in the 2x2 region.
            for h in range(height):
                for w in range(weight):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the corresponding gradient from the loss gradient to it.
                        if iterate_region[h, w, f2] == amax[f2]:
                            loss_gradient_input[i * 2 + h, j * 2 + w, f2] = loss_gradient[i, j, f2]

        # Return the gradient of the loss function with respect to the inputs of this layer.
        return loss_gradient_input


class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_size, num_nodes):
        """
        Initializes a DenseLayer object with the specified input size and number of nodes.

        Args:
        input_size: The number of input features to the layer.
        num_nodes: The number of nodes in the layer.

        Returns:
        None
        """
        # Initialize the weights matrix with random values with shape (input_size, num_nodes).
        # Divide the values by input_size to reduce their variance and prevent numerical instability.
        self.weights = np.random.randn(input_size, num_nodes) / input_size

        # Initialize the biases vector as a vector of zeros with length num_nodes.
        self.biases = np.zeros(num_nodes)

    def forward(self, input):
        """
        Performs forward propagation for the DenseLayer, computing the output of the layer given the input.

        Args:
        input: A 1D array representing the input to the layer, with length equal to the input size of the layer.

        Returns:
        output: A 1D array representing the output of the layer, with length equal to the number of nodes in the layer.
        """
        # Store the shape of the input as an instance variable.
        self.previous_input_shape = input.shape

        # Flatten the input into a 1D array.
        input = input.flatten()
        # Store the flattened input as an instance variable.
        self.previous_input = input

        # Get the shape of the weights' matrix.
        input_size, num_nodes = self.weights.shape

        # Compute the weighted sums for each node, plus the biases.
        dot_product = np.dot(input, self.weights) + self.biases
        # Store the weighted sums as an instance variable.
        self.last_totals = dot_product

        # Apply the softmax activation function to the weighted sums, and return the result.
        softmax_fn = np.exp(dot_product)
        return softmax_fn / np.sum(softmax_fn, axis=0)

    def backpropagation(self, loss_gradient, learn_rate):  # The backpropagation method for the Softmax layer
        # We know only 1 element of loss_gradient will be nonzero
        for i, gradient in enumerate(loss_gradient):
            if gradient == 0:
                continue

            # e^totals
            e_totals = np.exp(self.last_totals)

            # Sum of all e^totals
            sum_e_totals = np.sum(e_totals)

            # Gradients of out[i] against totals
            gradient_output_ws = -e_totals[i] * e_totals / (sum_e_totals ** 2)  # gradient_output_ws is output of
            # denseLayer wrt weighted sums of the nodes
            gradient_output_ws[i] = e_totals[i] * (sum_e_totals - e_totals[i]) / (sum_e_totals ** 2)

            # Gradients of totals against weights/biases/input
            ws_weight = self.previous_input  # ws_weight is weighted sums wrt to weights of layer
            ws_bias = 1  # ws_bias is weighted sums wrt to bias of layer
            ws_inputs = self.weights  # ws_inputs is weighted sums wrt to input

            # Gradients of loss against totals
            lf_weights = gradient * gradient_output_ws  # lf_weights is loss function wrt to weighted sums

            # Gradients of loss against weights/biases/input
            loss_weights = ws_weight[np.newaxis].T @ lf_weights[np.newaxis]
            loss_bias = lf_weights * ws_bias
            loss_inputs = ws_inputs @ lf_weights

            # Update weights / biases
            self.weights -= learn_rate * loss_weights
            self.biases -= learn_rate * loss_bias

            return loss_inputs.reshape(self.previous_input_shape)


# Function to create batches for training
def create_batches(images, labels, batch_size):
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        yield images[start:end], labels[start:end]


def forward(image, label):
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


from averagepool import AveragePool2


# Function to train a batch of images
def train_batch(images, labels, lr=0.0001):
    batch_loss = 0
    num_correct = 0
    batch_size = len(images)

    for im, label in zip(images, labels):
        out, loss, acc = forward(im, label)

        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        gradient = softmax.backpropagation(gradient, lr)
        gradient = pool.backpropagation(gradient)
        gradient = conv.backpropagation(gradient, lr)

        batch_loss += loss
        num_correct += acc

    return batch_loss / batch_size, num_correct / batch_size


# Load the CK+ dataset
train_images, train_labels, test_images, test_labels = load_ckplus_data()

# Initialize the layers of the CNN
conv = ConvLayer(16)
pool = Max_pool()
# pool = AveragePool2
softmax = Softmax(23 * 23 * 16, 7)

# Set batch size and number of epochs
batch_size = 16
epochs = 50
epoch_list = []
train_loss_list = []
train_accuracy_list = []

# Iterate through the epochs
for epoch in range(epochs):
    print('\n--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    shuffle = np.random.permutation(len(train_images))
    train_images = train_images[shuffle]
    train_labels = train_labels[shuffle]

    loss = 0
    num_correct = 0
    num_batches = math.ceil(len(train_images) / batch_size)

    # Train the model on each batch
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(train_images))
        batch_images = train_images[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]

        batch_loss, batch_acc = train_batch(batch_images, batch_labels)
        loss += batch_loss * len(batch_images)
        num_correct += batch_acc * len(batch_images)

    # Calculate average loss and accuracy for the epoch
    avg_loss = loss / len(train_images)
    avg_accuracy = num_correct / len(train_images)

    epoch_list.append(epoch + 1)
    train_loss_list.append(avg_loss)
    train_accuracy_list.append(avg_accuracy)

    print('Train Loss:', avg_loss)
    print('Train Accuracy:', avg_accuracy)

# Plot loss vs epoch
plt.plot(epoch_list, train_loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.show()

# Calculate test probabilities and true labels for ROC curve
test_probs = []
test_true_labels = []

for image, label in zip(test_images, test_labels):
    out, _, _ = forward(image, label)
    test_probs.append(out)
    test_true_labels.append(label)

test_probs = np.array(test_probs)
test_true_labels = np.array(test_true_labels)

# Plot ROC curve
fpr, tpr, _ = roc_curve(test_true_labels, test_probs[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot epoch vs accuracy
plt.plot(epoch_list, train_accuracy_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Epoch vs Accuracy')
plt.grid()
plt.show()