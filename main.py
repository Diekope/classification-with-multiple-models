# import models   # Pour appeler les modèles, il faut mettre "models." avant
from models import get_model_list, load_model
from training import train
from dataset_creation import adapter_dataset


def modelChoice():
    # Choix du modèle
    print("#--------------\nChoix du modèle\n#--------------")
    model_list = get_model_list()
    for i, model_name in enumerate(model_list, start=1):
        print(f"{i}. {model_name}")

    try:
        choice = int(input("Entrez le numéro du modèle : "))
        selected_model_name = model_list[choice - 1]
    except (IndexError, ValueError):
        print("Choix incorrect. Veuillez sélectionner un numéro valide.")
        return

    # Charger le modèle choisi
    try:
        model = load_model(selected_model_name)
        print(f"Modèle sélectionné : {selected_model_name}")
        train(model)
    except ValueError as e:
        print(e)


def questionnaire():
    a = input("Que voulez-vous faire ?\n1 : Entrainer un modèle\n2 : Tester un modèle\nChoix : ")
    incorrectChoice = a != "1" and a != "2"
    while incorrectChoice:
        print("Choix incorrect !")
        a = input("Que voulez-vous faire ?\n1 : Entrainer un modèle\n2 : Tester un modèle\nChoix : ")
        incorrectChoice = a != "1" and a != "2"

    if a == "1":
        print("Entrainement !")
        goodDataset = input("Avez-vous un dataset qui est adapté à l'entrainement (Qui contient des fichiers train, test, val) (o/n)?\nRéponse : ")
        incorrectChoice = goodDataset != "o" and goodDataset != "n"
        while incorrectChoice:
            print("Réponse incomprise, Veuillez réessayer !")
            goodDataset = input("Avez-vous un dataset qui est adapté à l'entrainement (Qui contient des fichiers train, test, val) (o/n)?\nRéponse : ")
            incorrectChoice = goodDataset != "o" and goodDataset != "n"

        if goodDataset == "o":
            modelChoice()
        elif goodDataset == "n":
            adapter_dataset()
            modelChoice()

    # elif a == "2":

def main():
    questionnaire()

if __name__ == "__main__":
    main()

