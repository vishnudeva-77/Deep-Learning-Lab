import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import warnings
warnings.filterwarnings("ignore")

faces = fetch_lfw_people(min_faces_per_person=100, resize=1.0,
                         slice_=(slice(60, 188), slice(60, 188)), color=True)

print("Target names (classes):", faces.target_names)
print("Image shape:", faces.images.shape)

sns.set()
fig, ax = plt.subplots(3, 6, figsize=(18, 10))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i] / 255.0)
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
plt.tight_layout()
plt.show()

counts = Counter(faces.target)
names = {faces.target_names[k]: v for k, v in counts.items()}
df = pd.DataFrame.from_dict(names, orient='index', columns=["Count"])
df.plot(kind='bar', figsize=(10, 5), legend=False)
plt.ylabel("Number of Images")
plt.title("Images per Person")
plt.tight_layout()
plt.show()

mask = np.zeros(faces.target.shape, dtype=bool)
for target in np.unique(faces.target):
    mask[np.where(faces.target == target)[0][:100]] = 1

x_faces = faces.data[mask]
y_faces = faces.target[mask]
x_faces = np.reshape(x_faces, (x_faces.shape[0], faces.images.shape[1], faces.images.shape[2], faces.images.shape[3]))

x_faces = x_faces / 255.0
y_faces_cat = to_categorical(y_faces)

x_train, x_test, y_train, y_test = train_test_split(
    x_faces, y_faces_cat, train_size=0.8, stratify=y_faces_cat, random_state=0)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=x_faces.shape[1:]))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(faces.target_names), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                 epochs=20, batch_size=25, verbose=1)

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

plt.figure(figsize=(8, 6))
sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            cmap='Blues', xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

try:
    img = image.load_img('george.jpg', target_size=(x_faces.shape[1], x_faces.shape[2]))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Input Image")
    plt.show()

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    print("\nPrediction Probabilities:")
    for i in range(len(prediction)):
        print(f"{faces.target_names[i]}: {prediction[i]:.4f}")

    predicted_label = faces.target_names[np.argmax(prediction)]
    print("\nPredicted Person:", predicted_label)
except FileNotFoundError:
    print("\n[Warning] 'george.jpg' not found. Please place the image in the same directory.")
