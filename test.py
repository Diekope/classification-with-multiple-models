from models import GoogleNet
import torch
from torchvision import transforms
from PIL import Image

# Charger l'architecture du modèle
model = GoogleNet(9)
model.load_state_dict(torch.load("trained_model_google.pth", map_location=torch.device('cpu'), weights_only=True))  # Remplacez "votre_modele.pth" par le nom de votre fichier
model.eval()  # Mettre le modèle en mode évaluation

### Étape 2: Transformer l'image
def transform_image(image_path):
    # Définir les transformations nécessaires en fonction de l'entraînement de votre modèle
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ajustez la taille en fonction de votre modèle
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Remplacez les valeurs par celles utilisées pour l'entraînement
    ])
    
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Ajouter une dimension batch

### Étape 3: Prédire la classe
def predict(image_path):
    image_tensor = transform_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()  # Retourner l'indice de la classe prédite

# Exemple d'utilisation
image_path = "test/boubou.png"  # Remplacez par le chemin de votre image
classe_predite = predict(image_path)
print(f"Classe prédite : {classe_predite}")