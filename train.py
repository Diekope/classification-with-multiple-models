from model import CustomCNN
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train(model, num_epochs=2, batch_size=16, learning_rate=0.01):

    # Définir les transformations pour le dataset (à ajuster selon vos besoins)
    transform = transforms.Compose([
        transforms.Resize((524, 524)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Création des chemin vers les différents fichiers train, test et val
    root = input("Veuillez entrer le chemin vers la racine du dataset : ")
    if not os.path.exists(root):
        while (not os.path.exists(root)):
            print("Le chemin n'existe pas, veuillez vérifier.")
            root = input("Veuillez entrer le chemin vers la racine du dataset : ")

    # Détection de l'appareil (GPU si disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Chargement des datasets
    train_dataset = datasets.ImageFolder(root=f"{root}/train", transform=transform)
    valid_dataset = datasets.ImageFolder(root=f"{root}/valid", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{root}/test", transform=transform)

    # Créer des DataLoader pour l'entraînement et la validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)
    print(f"Nombre de classes : {num_classes}")

    # Configurer le modèle
    model = model.to(device)  # Déplacer le modèle sur le bon appareil
    model.train()  # Mettre le modèle en mode entraînement

    # Définir le critère de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialiser les listes pour stocker les métriques
    train_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")

        for i, (inputs, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()

            # Déplacer les données sur l'appareil
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calcul de la perte moyenne pour le lot courant
            average_loss = running_loss / i
            train_loader.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

        # Calculer et enregistrer la perte moyenne pour l'epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Évaluer la précision sur le validation_loader
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        valid_accuracies.append(accuracy)
        print(f"Validation Accuracy: {accuracy:.2f}%")

    # Enregistrer les graphiques de la perte et de la précision
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('training_loss.png')

    plt.figure()
    plt.plot(range(1, num_epochs + 1), valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.savefig('validation_accuracy.png')

    # Tester le modèle sur l'ensemble de test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Enregistrer le graphique de la précision du test
    plt.figure()
    plt.bar(['Test Accuracy'], [test_accuracy])
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.savefig('test_accuracy.png')
    