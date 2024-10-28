from model import GoogleNet
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def train(model, num_epochs=100, batch_size=128, learning_rate=0.0001):

    # Définir les transformations pour le dataset (à ajuster selon vos besoins)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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

    # Enregistrer le modèle
    model_save_path = "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Modèle enregistré à : {model_save_path}")

    # Enregistrer les graphiques de la perte, de la précision et de la heatmap
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Graphe de la perte d'entraînement
    axes[0, 0].plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='b')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss over Epochs')
    axes[0, 0].legend()

    # Graphe de la précision de validation
    axes[0, 1].plot(range(1, num_epochs + 1), valid_accuracies, label='Validation Accuracy', color='g')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Validation Accuracy over Epochs')
    axes[0, 1].legend()

    # Heatmap des métriques
    metrics = np.array([train_losses, valid_accuracies])
    sns.heatmap(metrics, annot=True, fmt='.2f', ax=axes[1, 0], cmap='viridis', cbar=True, yticklabels=['Loss', 'Accuracy'])
    axes[1, 0].set_title('Heatmap of Training Loss and Validation Accuracy')

    # Confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes=train_dataset.classes, normalize=True, title='Confusion Matrix Normalized', cmap=plt.cm.Blues)
    axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix Normalized')
    axes[1, 1].set_xticks(np.arange(len(train_dataset.classes)))
    axes[1, 1].set_yticks(np.arange(len(train_dataset.classes)))
    axes[1, 1].set_xticklabels(train_dataset.classes, rotation=90)
    axes[1, 1].set_yticklabels(train_dataset.classes)

    # Sauvegarder l'image combinée
    plt.tight_layout()
    plt.savefig('training_metrics.png')

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
    plt.bar(['Test Accuracy'], [test_accuracy], color='c')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.savefig('test_accuracy.png')
