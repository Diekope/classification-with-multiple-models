import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Définir les transformations pour le dataset (à ajuster selon tes besoins)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Création des chemin vers les différents fichiers train, test et val
root = input("Veuillez entrer le chemin vers la racine du dataset : ")
if not os.path.exists(root):
    while (not os.path.exists((root))):
        print("Le chemin n'existe pas, veuillez vérifier.")
        root = input("Veuillez entrer le chemin vers la racine du dataset : ")

train_dataset = datasets.ImageFolder(root=f"{root}/train", transform=transform)
valid_dataset = datasets.ImageFolder(root=f"{root}/val", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{root}/test", transform=transform)

# Créer des DataLoader pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_classes = len(train_dataset.classes)

print(num_classes)

def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Mettre le modèle en mode entraînement
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for epoch in range(num_epochs):
            running_loss = 0.0
            train_loader = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")

            for i, (inputs, labels) in enumerate(train_loader, 1):
                optimizer.zero_grad()

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

        # Afficher la perte moyenne de l'epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


print(num_classes)

# model = GoogleNet(num_classes)
# criterion = nn.CrossEntropyLoss()  # La perte standard pour la classification multi-classes
# optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Optimiseur Adam

# Entraînement du modèle
# train(model, train_loader, criterion, optimizer, num_epochs=50)

# Sauvegarder le modèle entier
# torch.save(model.state_dict(), 'googlenet_model.pth')