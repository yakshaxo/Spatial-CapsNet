CapsNet-PyTorch : Advanced Spatial-Aware Modeling
Ce dépôt contient une implémentation robuste des Réseaux de Capsules (CapsNet), basée sur l'architecture pionnière proposée par Sara Sabour, Nicholas Frosst et Geoffrey Hinton. Contrairement aux réseaux de neurones convolutifs (CNN) traditionnels, les CapsNets utilisent le "Dynamic Routing" pour mieux préserver les relations hiérarchiques et spatiales entre les caractéristiques d'une image.

📌 Présentation du Projet
L'objectif de ce projet est de fournir une structure modulaire permettant d'expérimenter la puissance des capsules. Les CapsNets résolvent le problème de la "perte d'information spatiale" liée au pooling dans les CNN, rendant le modèle beaucoup plus performant face aux transformations (rotation, inclinaison, échelle).

Points Forts :
Dynamic Routing : Implémentation de l'algorithme de routage par accord.

Invariance Spatiale : Meilleure reconnaissance des objets sous différents angles.

Architecture Modulaire : Code structuré pour faciliter l'adaptation à de nouveaux jeux de données (MNIST, Fashion-MNIST, etc.).

🏗️ Architecture du Modèle
Le modèle est composé de trois blocs principaux :

Convolution Layer : Extraction initiale des caractéristiques.

Primary Capsules : Regroupement des neurones en vecteurs (capsules) représentant des entités physiques.

Digit Capsules : Couche finale traitée par l'algorithme de routage dynamique pour la classification.

🚀 Installation
Clonez le dépôt et installez les dépendances nécessaires :

Bash

git clone https://github.com/yakshaxo/capsnet.git
cd capsnet
pip install -r requirements.txt
📊 Utilisation
Entraînement
Pour lancer l'entraînement sur le dataset par défaut (MNIST) :

Bash

python train.py --epochs 50 --batch_size 128
Évaluation
Pour tester le modèle entraîné :

Bash

python evaluate.py --model_path ./models/capsnet_v1.pth
📈 Résultats
Le modèle atteint une précision de ~99% sur l'ensemble de test MNIST après 50 époques, démontrant sa capacité à généraliser avec moins de paramètres que certains réseaux profonds classiques.

📚 Références
Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic Routing Between Capsules. In NIPS.

Développé par [Votre Nom/Pseudo] Projet réalisé dans le cadre de recherches sur l'apprentissage profond et la vision par ordinateur.

Introduction aux réseaux de capsules (CapsNet)

Cette vidéo explique de manière pédagogique le fonctionnement des réseaux de capsules et pourquoi ils représentent une alternative prometteuse aux réseaux de neurones convolutifs classiques.
