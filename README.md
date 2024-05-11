# Face Mask Detection Using Convolutional Neural Networks (CNN)
This project aims to detect whether a person is wearing a face mask or not using Convolutional Neural Networks (CNN). The model is trained on a dataset containing images of people with and without face masks.

## Dataset
The dataset used for training the CNN model consists of two classes:

+ With Mask: Images of people wearing face masks.
+ Without Mask: Images of people without face masks.<br>
The dataset is organized into training, validation, and test sets to evaluate the model's performance.

## Model Architecture
The CNN model architecture consists of multiple convolutional layers followed by max-pooling layers for feature extraction. It also includes fully connected layers and output layers for classification. The model is trained using the training dataset and optimized using techniques such as gradient descent and backpropagation.

### Usage
 1. Training the Model: Run the training script to train the CNN model on the provided dataset.

```
python train.py
```
 2. Evaluation: Evaluate the trained model's performance on the validation set.


```
python evaluate.py
```
 3. Prediction: Use the trained model to make predictions on new images.
```

python predict.py --image_path 
```
## Dependencies
+ Python 3.x
+ TensorFlow
+ Keras
+ OpenCV
+ NumPy
Install the required dependencies using the following command:
```
pip install -r requirements.txt
```
## Results
The model achieves an accuracy of [insert accuracy] on the validation set. Further fine-tuning and optimization can improve the model's performance.
