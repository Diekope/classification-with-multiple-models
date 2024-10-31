import os
import shutil
import random
from tqdm import tqdm

def adapter_dataset():
    """
    Parcourt un dataset brut et répartit les images dans les dossiers 'train', 'test' et 'valid'
    dans un nouveau dossier créé, tout en respectant les catégories et la répartition spécifiée.

    :param dataset_path: Chemin du dataset brut contenant les catégories
    """

    # Chemin du dataset brut
    dataset_path = input("Veuillez entrer le nom de votre dataset brut : ")
    dataset_path = "datasets/"+dataset_path
    if not os.path.exists(dataset_path):
        while not os.path.exists(dataset_path):
            print("Le fichier n'existe pas, veuillez vérifier.")
            dataset_path = input("Veuillez entrer le nom de votre dataset brut : ")

    train_percent = int(input("Quel pourcentage de données d'entraînement voulez-vous : "))
    if (train_percent < 50 or train_percent > 90):
        print("Pourcentage incorrect, veuillez donner un nombre entre 50 et 90%")
        train_percent = int(input("Quel pourcentage de données d'entraînement voulez-vous : "))
    train_percent = (train_percent / 100)

    valid_percent = int(input("Quel pourcentage de données de validation voulez-vous : "))
    if (valid_percent < 10 or valid_percent > 30):
        print("Pourcentage incorrect, veuillez donner un nombre entre 10 et 30%")
        valid_percent = int(input("Quel pourcentage de données de validation voulez-vous : "))
    valid_percent = (valid_percent / 100)

    test_percent = 1 - (train_percent + valid_percent)

    # Nouveau dossier basé sur le nom du dataset
    dataset_name = os.path.basename(dataset_path)
    transformed_dataset_path = os.path.join(os.path.dirname(dataset_path), dataset_name + '-adapted')

    # Dossiers de sortie
    output_dirs = ['train', 'valid', 'test']
    repartition = {'train': train_percent, 'valid': valid_percent, 'test': test_percent}

    # Crée le dossier transformé s'il n'existe pas déjà
    os.makedirs(transformed_dataset_path, exist_ok=True)

    # Crée les dossiers de sortie 'train', 'valid', et 'test' dans le dossier transformé
    for dir_name in output_dirs:
        os.makedirs(os.path.join(transformed_dataset_path, dir_name), exist_ok=True)

    # Statistiques de répartition
    stats = {category: {'train': 0, 'valid': 0, 'test': 0} for category in os.listdir(dataset_path) if not category.startswith('.')}

    # Parcours les dossiers de catégories dans le dataset d'origine
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)

        # Ignorer les fichiers ou les dossiers cachés
        if not os.path.isdir(category_path) or category.startswith('.'):
            continue

        # Liste les images de la catégorie
        images = [img for img in os.listdir(category_path) if not img.startswith('.')]
        random.shuffle(images)  # Mélange les images pour une répartition aléatoire

        # Calcule les indices de répartition
        total_images = len(images)
        train_split = int(total_images * repartition['train'])
        valid_split = int(total_images * (repartition['train'] + repartition['valid']))

        # Répartir les images dans les dossiers respectifs avec une barre de progression
        for idx, image in enumerate(tqdm(images, desc=f"Répartition des images de la catégorie '{category}'", leave=False)):
            if idx < train_split:
                target_dir = 'train'
                stats[category]['train'] += 1
            elif idx < valid_split:
                target_dir = 'valid'
                stats[category]['valid'] += 1
            else:
                target_dir = 'test'
                stats[category]['test'] += 1

            # Crée le sous-dossier de catégorie dans le dossier cible s'il n'existe pas déjà
            target_category_dir = os.path.join(transformed_dataset_path, target_dir, category)
            os.makedirs(target_category_dir, exist_ok=True)

            # Copie l'image vers le sous-dossier correspondant
            src_image_path = os.path.join(category_path, image)
            dest_image_path = os.path.join(target_category_dir, image)
            shutil.copy2(src_image_path, dest_image_path)  # Utilisation de copy2 pour préserver les métadonnées

    print(f"\nVotre dataset {transformed_dataset_path} a été correctement créé.\n")

    # Affichage des statistiques
    print("Statistiques de répartition :")
    for category, count in stats.items():
        total_category = sum(count.values())
        print(f"- Catégorie '{category}': Total = {total_category} images")
        print(f"  Train : {count['train']} images")
        print(f"  Valid : {count['valid']} images")
        print(f"  Test  : {count['test']} images")
