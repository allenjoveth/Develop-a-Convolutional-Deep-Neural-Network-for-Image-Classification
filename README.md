# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a Convolutional Deep Neural Network (CNN) for image classification using the Fashion-MNIST dataset and to verify the response for new images.

---

## PROBLEM STATEMENT AND DATASET

### Problem Statement
The objective of this experiment is to build a Convolutional Neural Network (CNN) model that can classify grayscale fashion images into different clothing categories. The model is trained to identify clothing items such as T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

### Dataset
The **Fashion-MNIST** dataset is used for this experiment. It is a benchmark dataset consisting of grayscale images of fashion products.

**Dataset Details:**
- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Number of Classes: 10
- Input Shape: `(1, 28, 28)`

### Class Labels
1. T-shirt/top  
2. Trouser  
3. Pullover  
4. Dress  
5. Coat  
6. Sandal  
7. Shirt  
8. Sneaker  
9. Bag  
10. Ankle boot

---

## Neural Network Model

<img width="1749" height="903" alt="Screenshot 2026-05-12 162328" src="https://github.com/user-attachments/assets/d1245bc3-f06e-4088-bcd7-00c69df2eece" />

---

## DESIGN STEPS

### STEP 1: Data Loading and Preprocessing
The Fashion-MNIST dataset is loaded using `torchvision.datasets.FashionMNIST`. The images are transformed into tensors and normalized using `transforms.Compose()` to improve model performance.

### STEP 2: Dataset Analysis and Batch Processing
The shape of the training and testing images is verified. `DataLoader` is used to split the dataset into batches for efficient training and testing.

### STEP 3: CNN Model Construction
A Convolutional Neural Network model is developed using:
- Three Convolutional Layers (`Conv2D`)
- ReLU Activation Function
- Max Pooling Layers
- Fully Connected Dense Layers

The CNN extracts image features and classifies them into one of the ten fashion categories.

### STEP 4: Model Training
The model is trained using:
- **Loss Function:** Cross Entropy Loss
- **Optimizer:** Adam Optimizer
- **Learning Rate:** 0.001

The training is performed for **3 epochs**, and training loss is calculated for each epoch.

### STEP 5: Model Evaluation
The trained model is tested using the test dataset. Model performance is evaluated using:
- Test Accuracy
- Confusion Matrix
- Classification Report

### STEP 6: Prediction on New Image
A new sample image from the test dataset is passed to the trained CNN model. The predicted class label is compared with the actual label.

---

## PROGRAM

### Name: ALLEN JOVETH P

### Register Number: 212223240007

```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier(input_size=(1,28,28))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the Model
def train_model(model, train_loader, num_epochs=3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print('Name: Allen Joveth P')
        print('Register Number: 212223240007')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

---

## OUTPUT

### Training Loss per Epoch

<img width="260" height="160" alt="Screenshot 2026-05-12 174328" src="https://github.com/user-attachments/assets/a10941a7-58ae-475b-93a4-593fc3dfdb46" />

## Confusion Matrix

<img width="776" height="593" alt="Screenshot 2026-05-11 210402" src="https://github.com/user-attachments/assets/4e3dd7e2-78f7-4a2e-bad2-3deac78c2a6d" />

## Classification Report

<img width="488" height="355" alt="Screenshot 2026-05-12 175005" src="https://github.com/user-attachments/assets/b2e11960-5835-4e8c-995c-0b1ef562a5a8" />

---

### New Sample Data Prediction

<img width="549" height="621" alt="Screenshot 2026-05-12 174003" src="https://github.com/user-attachments/assets/43044d3b-9ca0-440a-9027-294e1c9b87bf" />

---

## RESULT

Thus, a **Convolutional Deep Neural Network (CNN)** for image classification was successfully developed using the **Fashion-MNIST dataset**. The model was trained and tested successfully, achieving an overall test accuracy of **89.48%**, and it correctly classified new sample images.

