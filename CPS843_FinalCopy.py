import os
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)

def create_image_generators(base_path):
    """
    Create training and validation image data generators.
    
    :param: Base directory path containing 'train' and 'validation' directories
    :return: A pair of generators - the first for training data, the second for validation data.
    """
    train_dir = os.path.join(base_path, 'train')
    validation_dir = os.path.join(base_path, 'validation')

    # Image data generators with augmentation for training and simple rescaling for validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    return train_generator, validation_generator

def build_model():
    """
    Build and compile the CNN model.
    
    :return: A built CNN model that can be used for predictions
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)), #Convulution Layter
        MaxPooling2D(2, 2), #Spatial dimension reduction
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(), #Reshape it into a 1D array
        Dense(512, activation='relu'), #Affects and connects them to neurons
        Dropout(0.5), #Adjust for overfitting
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator, epochs=15):
    """
    Train the model with the provided data.

    :param model: Compiled CNN model made earlier
    :param train_generator: Training data used
    :param validation_generator: Validation data used
    :param epochs: Usage of epochs based on the amount of samples used (40)
    :return: Training metrics
    """
    history = model.fit(
        train_generator,
        steps_per_epoch=4,  # Adjust based on your dataset
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=4  # Adjust based on your dataset
    )
    return history

def predict_image(model, image_path):
    """
    Predict the class of an image.

    :param model: Trained CNN model
    :param image_path: Path to the image file
    :return: Sends the returned values of (file path, abnormality scale, and confidence) to main
    """
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    label = 'Abnormal' if confidence > 0.5 else 'Normal'
    return image_path, label, confidence

# Main execution
if __name__ == "__main__":
    #The execution grabs the project path, and starts generatoring the work models
    project_path = os.path.dirname(os.path.abspath(__file__))
    train_generator, validation_generator = create_image_generators(project_path) #tuple for the generator
    cnn_model = build_model()
    train_model(cnn_model, train_generator, validation_generator)

    # Outputs based on the test run and test files
    test_image_path = os.path.join(project_path, 'test', 'test3.jpg')
    file_path, label, confidence = predict_image(cnn_model, test_image_path)

    #Printing the final values. This is what the report wants
    print(f"File: {file_path}, Prediction: {label}, Confidence: {confidence}")
