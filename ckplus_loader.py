import os
import numpy as np
import imageio
from sklearn.model_selection import train_test_split


def load_ckplus_data(data_path='./CK+48', test_size=0.3, random_state=42):
    images = []
    labels = []
    label_mapping = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'sadness': 5, 'surprise': 6}

    for emotion, label in label_mapping.items():
        emotion_dir = os.path.join(data_path, emotion)
        for filename in os.listdir(emotion_dir):
            filepath = os.path.join(emotion_dir, filename)
            image = imageio.imread(filepath)
            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size,
                                                                            random_state=random_state, stratify=labels)

    return train_images, train_labels, test_images, test_labels
