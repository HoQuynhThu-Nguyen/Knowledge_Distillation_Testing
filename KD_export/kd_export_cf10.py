import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torchvision.datasets as datasets

###################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###################################################
# Deeper neural network class to be used as teacher:
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Lightweight neural network class to be used as student:
class LightNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


###################################################
print()
print("######################################################################")
print("Loading CIFAR-10 dataset:")
# Transformations for data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loading the CIFAR-10 dataset: (train set will later be split into train and val)
full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split trainset into train and validation datasets (80% train, 20% val)
train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])

# DataLoaders for train, validation, and test datasets
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=2)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Check if dataset loads correctly
print(f"Number of training samples: {len(trainset)}")
print(f"Number of validation samples: {len(valset)}")
print(f"Number of testing samples: {len(testset)}")

###################################################
def train(model, trainloader, valloader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    model.train()

    for epoch in range(epochs):

        # Training Step
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")

        # Validation Step
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() 

        avg_val_loss = val_loss / len(valloader)  # Average validation loss
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
    return train_losses, val_losses 

def test(model, testloader, device):
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
	
            # Collect predictions and true labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics using sklearn
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0)

    return cm, report


###################################################
print()
print("######################################################################")
print("Instantiate the teacher model.")
torch.manual_seed(42)
nn_deep = DeepNN(num_classes=10).to(device)
print("Cross-entropy runs with teacher model: ")
train_deep = train(nn_deep, trainloader, valloader, epochs=10, learning_rate=0.001, device=device)
test_deep = test(nn_deep, testloader, device)
test_accuracy_deep = test_deep[1]["accuracy"] * 100
print(f"Teacher Accuracy: {test_accuracy_deep:.2f}%")


###################################################
print("Instantiate the student model.")
torch.manual_seed(42)
nn_light = LightNN(num_classes=10).to(device)
print("Instantiate a copy of the student model.")
torch.manual_seed(42)
new_nn_light = LightNN(num_classes=10).to(device)


# # Print the norm of the first layer of the initial lightweight model
# print("To ensure we have created a copy of the student network, we inspect the norm of its first layer.")
# print("If it matches, then we are safe to conclude that the networks are indeed the same.")
# print("Norm of 1st layer of nn_light:", torch.norm(nn_light.features[0].weight).item())
# print("Norm of 1st layer of new_nn_light:", torch.norm(new_nn_light.features[0].weight).item())

# print("######################################################################")
# print("The total number of parameters in each model")
# total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
# print(f"Teacher model parameters: {total_params_deep}")
# total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
# print(f"Student model parameters: {total_params_light}")

print()
print("######################################################################")
print("Cross-entropy runs with student model: ")
train_light_ce = train(nn_light, trainloader, valloader, epochs=10, learning_rate=0.001, device=device)
test_light_ce = test(nn_light, testloader, device)
test_accuracy_light_ce = test_light_ce[1]["accuracy"] * 100
print(f"Student Accuracy: {test_accuracy_light_ce:.2f}%")

###################################################
def train_knowledge_distillation(teacher, student, trainloader, valloader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")

        # Validation Step
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                loss = ce_loss(outputs, labels)
                val_loss += loss.item() 

        avg_val_loss = val_loss / len(valloader)  # Average validation loss
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
    return train_losses, val_losses 


###################################################
print()
print("######################################################################")
print("Knowledge Distillation runs with the copy of the student model: ")
# Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
train_light_ce_and_kd = train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, trainloader=trainloader, valloader=valloader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_light_ce_and_kd = test(new_nn_light, testloader, device)
test_accuracy_light_ce_and_kd = test_light_ce_and_kd[1]["accuracy"] * 100
precision_light_ce_and_kd = test_light_ce_and_kd[1]["weighted avg"]["precision"]
recall_light_ce_and_kd = test_light_ce_and_kd[1]["weighted avg"]["recall"]
f1_light_ce_and_kd = test_light_ce_and_kd[1]["weighted avg"]["f1-score"]

# Compare the student test accuracy with and without the teacher, after distillation
# print("-----------------------------------------")
# print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
# print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")

print("-----------------------------------------")
print(f"Student accuracy with CE + KD:")
print(f"Accuracy: {test_accuracy_light_ce_and_kd:.2f}%")
# Print other value metrics:
print(f"Precision: {precision_light_ce_and_kd:.2f}")
print(f"Recall: {recall_light_ce_and_kd:.2f}")
print(f"F1 Score: {f1_light_ce_and_kd:.2f}")

###################################################
torch.save(nn_deep.state_dict(), "teacher_model_cf10.pth")
torch.save(nn_light.state_dict(), "student_model_cf10.pth")
torch.save(new_nn_light.state_dict(), "student_model_KD_cf10.pth")

print("Model saved as teacher_model_cf10.pth, student_model_cf10.pth and student_model_KD_cf10.pth")