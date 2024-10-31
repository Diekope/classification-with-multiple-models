from .AlexNet import AlexNet
from .GoogleNet import GoogleNet
from .GptCNN import CustomCNN  # Correction de la casse
from .BaseModel import BaseModel  # Import de la classe de base
import os
import importlib

def get_model_list():
    # Chemin vers le dossier où se trouvent les modèles
    models_dir = os.path.dirname(__file__)
    # Lister tous les fichiers Python (sauf __init__.py et base_model.py)
    model_files = [f[:-3] for f in os.listdir(models_dir) if f.endswith('.py') and f not in ('__init__.py', 'base_model.py')]
    # Exclure les fichiers qui contiennent "Base" ou qui ne sont pas des modèles valides
    model_classes = [name for name in model_files if not name.lower().startswith('base')]
    return model_classes

def load_model(model_name, num_classes=10):
    # Dynamique pour charger la classe du modèle à partir du nom du fichier
    module = importlib.import_module(f"models.{model_name}")
    model_class = getattr(module, model_name)
    return model_class(num_classes=num_classes)