# main.py

from making_dataset import adapter_dataset
from model import *
from train import CustomCNN, train

# Optionnel : Adapter le dataset si nécessaire
# adapter_dataset()

# Créer une instance du modèle
greg = CustomCNN(9)
print(greg)

# Lancer l'entraînement
train(greg)
