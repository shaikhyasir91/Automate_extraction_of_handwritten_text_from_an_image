# Automate Extraction of Handwritten Text From an Image

# 1. Imports and Setup:
The code starts by importing essential libraries such as NumPy, OpenCV for image processing, pandas for data handling, and matplotlib for visualization.
It also imports various components from Keras and TensorFlow to build, train, and evaluate a deep learning model.
sklearn is used for data preprocessing tasks like scaling and splitting the dataset.

# 2. Reading and Preparing the Dataset:

The code reads a text file (words.txt) that contains metadata about handwritten words and their associated image file paths.
Lines from the file are extracted (starting after the first 18 lines) to process the actual image data and corresponding
1. Imports and Setup:

The code starts by importing essential libraries such as NumPy, OpenCV for image processing, pandas for data handling, and matplotlib for visualization.

It also imports various components from Keras and TensorFlow to build, train, and evaluate a deep learning model.

sklearn is used for data preprocessing tasks like scaling and splitting the dataset.


2. Reading and Preparing the Dataset:

The code reads a text file (words.txt) that contains metadata about handwritten words and their associated image file paths.

Lines from the file are extracted (starting after the first 18 lines) to process the actual image data and corresponding labels.


# 3. Character Set Definition and Label Encoding:

A character list (char_list) is defined, consisting of all possible characters (digits, uppercase, lowercase letters, and punctuation) that the model needs to recognize.

A function, encode_to_labels(), is created to convert each word (string) into a sequence of integers, with each character being mapped to its index in the char_list.



# 4. Image Processing Function:

The process_image() function takes an image, resizes it to a fixed dimension (32, 128), normalizes pixel values, and adds padding if necessary to ensure all images have the same size.

The images are converted to grayscale, and their pixel values are normalized to fall between 0 and 1 for better model performance.



# 5. Splitting the Dataset:

The dataset is split into training and validation sets. For every 10th image, the data is assigned to the validation set; the rest goes to the training set.

Images and their corresponding labels are processed and stored in separate lists for both training and validation.


# 6. Padding Labels:

The labels (sequences of integers) are padded to ensure all labels have the same length. This is done using pad_sequences() from Keras. Padding ensures that each batch fed into the model is of uniform size, which simplifies training.


# 7. CNN-BiLSTM Model Architecture:

The model is built with a combination of CNN and BiLSTM layers to handle both the spatial and sequential nature of the handwriting recognition task:

1. Convolutional layers (CNN): These layers extract features from the images, such as edges, shapes, and textures. Several convolutional layers are stacked, and the image dimensions are reduced using MaxPooling layers.


2. Batch Normalization is applied to stabilize and speed up training.


3. Squeezing: A Lambda layer is used to remove unnecessary dimensions from the image tensors, preparing the output for sequential data processing.


4. Bidirectional LSTMs: These layers capture sequential dependencies in the data. Since handwriting involves a sequence of characters, LSTMs are well-suited for this task. The model uses two Bidirectional LSTM layers, allowing it to capture dependencies in both forward and backward directions.


5. Dense Layer with Softmax Activation: This final layer produces a probability distribution over the possible characters for each time step, allowing the model to predict a sequence of characters.




# 8. Connectionist Temporal Classification (CTC) Loss:

Since the input (image) and output (sequence of characters) have different lengths, the model uses CTC Loss during training. CTC is ideal for tasks like handwriting recognition because it can handle varying input and output lengths without needing exact alignment.

The ctc_lambda_func() is defined to calculate the CTC loss.


# 9. Model Compilation:

The model is compiled using the SGD (Stochastic Gradient Descent) optimizer, and the CTC loss function is applied for training.

The model is trained using both the processed images and padded labels, with validation data provided to evaluate performance after each epoch.


# 10. Training the Model:

The model is trained using the Keras fit function, with parameters like batch size (8) and epochs (30). The model is evaluated on both the training and validation datasets during training, and performance metrics like accuracy are tracked.


# Key Concepts and Why They Were Used:

CNN: To handle the spatial structure of the handwritten text images, extracting features like edges and patterns.

BiLSTM: To capture the sequential structure of the characters, essential for recognizing words and ensuring the model understands the order of characters.

CTC Loss: For handling sequence-to-sequence problems where the input and output lengths vary (e.g., images of different widths and text lengths).

Normalization and Padding: To ensure consistent input dimensions and stabilize training by maintaining similar value ranges for images and labels.
