# Chatd vs Chiens Classification d'Images : CNN From Scratch vs. ResNet18 Transfer Learning

## Objectif du Projet
Ce projet vise à concevoir, entraîner et comparer deux approches de Deep Learning sous PyTorch pour la classification binaire d'images (Chats vs. Chiens) :
1. **Expérience A** : Un réseau de neurones convolutif (*CNN*) personnalisé, conçu et entraîné *from scratch*.
2. **Expérience B** : Une approche par **Transfer Learning** basée sur l'architecture **ResNet18** pré-entraînée sur ImageNet.

L'objectif est d'analyser l'impact du transfert d'apprentissage sur les performances, le temps de convergence et la capacité de généralisation.

## Environnement et Installation

### Prérequis
- Python 3.8+
- GPU recommandé

### Installation
Pour installer toutes les dépendances nécessaires, exécutez la commande suivante :

```bash

pip install -r requirements.txt

```

## Organisation des Données

### Où télécharger les données ?
Le jeu de données utilisé est l'archive structurée `Cat_Dog_data.zip` :
- **Lien de téléchargement direct** : [Udacity Cat_Dog_data.zip](https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip)

### Où placer les dossiers ?
Téléchargez l'archive, dézippez-la et placez son contenu dans un dossier `data/` à la racine de votre projet selon l'arborescence suivante :

```text
cnn-catsdogs-<NomPrenom>/
└── data/
    ├── train/
    │   ├── cat/
    │   └── dog/
    ├── val/         
    │   ├── cat/
    │   └── dog/
    └── test/        
        ├── cat/
        └── dog/
  
```

## Commandes d'Entraînement

### 1. Expérience A : CNN From Scratch

Entraînement du modèle personnalisé intégrant *Data Augmentation*, *Batch Normalization* et *Dropout* :

```bash

python train.py --model scratch --batch_size 32 --lr 0.001 --epochs 10 --optimizer adam --dropout 0.4 --use_bn --scheduler StepLR --step_size 5 --gamma 0.1

```

* **Arguments clés** :
* `--lr` : Taux d'apprentissage initial ($0.001$).
* `--dropout` : Désactivation aléatoire de $40\%$ des neurones sur le classifieur.
* `--use_bn` : Activation de la *Batch Normalization* après chaque couche convolutive.
* `--scheduler` : Réduction du LR d'un facteur $0.1$ toutes les 5 époques.

### 2. Expérience B : Transfer Learning (ResNet18)

Entraînement de la nouvelle tête de classification sur la base ResNet18 :

```bash

python train.py --model resnet18 --base imagenet --freeze_features True --batch_size 32 --lr 0.001 --epochs 10 --optimizer adam --dropout 0.4 --scheduler StepLR --step_size 5 --gamma 0.1

```

* **Arguments clés** :
* `--base` : Poids pré-entraînés `ResNet18_Weights.DEFAULT` (ImageNet).
* `--freeze_features` : Gel des couches convolutives (`requires_grad = False`). Seule la nouvelle tête `fc` (Linear -> ReLU -> Dropout 0.4 -> Linear) est entraînée.

## Évaluation et Chargement des Checkpoints

> **Note** : Les fichiers checkpoints (`best_model_scratch.pt` et `best_model_transfer.pt`) sont automatiquement générés localement lors de l'exécution de l'entraînement.

Pour recharger un modèle sauvegardé et exécuter l'évaluation sur le jeu de test :

```bash
# Évaluation de l'Expérience A (CNN From Scratch)
python evaluate.py --model scratch --checkpoint best_model_scratch.pt

# Évaluation de l'Expérience B (ResNet18)
python evaluate.py --model resnet18 --checkpoint best_model_transfer.pt

```

---

## Résultats et Analyse Comparative

### Tableau Comparatif (Jeu de Test - 2 500 images)

| Métrique | Expérience A (CNN From Scratch) | Expérience B (ResNet18 Transfer) | Gain / Évolution |
| --- | --- | --- | --- |
| **Test Accuracy** | **71.04%** | **97.76%** | **+26.72%**  |
| **Test Loss** | **0.5527** | **0.0560** | **-0.4967**  |
| **Test Precision** | **0.8593** | **0.9893** | **+0.1300** |
| **Test Recall** | **0.5032** | **0.9656** | **+0.4624** |
| **F1-Score (Chat / Chien)** | 0.76 / 0.63 | 0.98 / 0.98 | Équilibre parfait |

### Analyse synthétique

L'architecture **CNN From Scratch** atteint une exactitude de **71.04%**. Bien que la combinaison de la *Batch Normalization*, du *Dropout* (0.4) et de la *Data Augmentation* ait permis d'éviter le surapprentissage (les courbes d'entraînement et de validation restant alignées), le modèle présente une limite majeure : un fort biais en faveur de la classe *Chat*. Avec un rappel de seulement **50.32%** sur la classe *Chien*, le réseau peine à extraire des descripteurs complexes sur un volume de données restreint sans poids pré-entraînés.

À l'inverse, l'approche par **Transfer Learning avec ResNet18** surpasse nettement la baseline avec une exactitude de **97.76%** et une perte réduite à **0.0560**. Le gel du *backbone* pré-entraîné sur ImageNet permet non seulement d'éliminer totalement le biais (rappel de **96.56%** sur les chiens et **99.00%** sur les chats), mais assure également une convergence extrêmement rapide dès la première époque.

En conclusion, le transfert d'apprentissage démontre sa supériorité incontestable pour les tâches de vision par ordinateur sur des jeux de données de taille modérée, offrant une précision maximale pour un coût computationnel réduit.

---

## Limites & Pistes d’Amélioration

1. **Fine-Tuning Partiel / Total** : Défiger les derniers blocs convolutifs de ResNet18 avec un taux d'apprentissage très faible ($\eta = 10^{-5}$) pour spécialiser encore plus les descripteurs visuels.
2. **Data Augmentation Avancée** : Tester des techniques d'augmentation plus poussées comme *CutMix*, *MixUp* ou l'ajustement dynamique de couleur (*ColorJitter*).
3. **Exploration d'architectures plus récentes** : Comparer ResNet18 avec des architectures modernes comme *ConvNeXt* ou des *Vision Transformers* (ViT).

