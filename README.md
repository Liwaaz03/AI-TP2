# TP : Classification de texte avec sklearn

**Date limite :** 31 mars 2025

## Introduction

Dans ce TP, vous allez explorer des techniques de classification de texte en utilisant la bibliothèque scikit-learn. L’objectif est de construire des modèles de machine learning pour classer des textes. 

### Objectifs :
1. **Partie 1 :** Approches classiques comme Bag of Words et TF-IDF.
2. **Partie 2 :** Méthode moderne avec les embeddings générés par un modèle Sentence Transformers.
3. **Partie 3 :** Implémentation d’un Super Learner pour combiner plusieurs modèles.

Vous devrez rédiger un rapport court (2-3 pages max) expliquant vos choix, vos résultats et une comparaison des approches. Le code doit être clair et commenté.

---

## Prérequis

- **Bibliothèques :** `scikit-learn`, `numpy`, `pandas`, `sentence-transformers`, `matplotlib`.
- **Jeu de données :** Utilisez le dataset SMS Spam Collection disponible sur Google Drive. Ce dataset contient des messages SMS étiquetés comme "spam" ou "ham" (non-spam).

---

## Partie 1 : Classification avec Bag of Words et TF-IDF

Dans cette partie, vous allez transformer des textes en représentations numériques avec deux méthodes classiques : Bag of Words (BoW) et Term Frequency-Inverse Document Frequency (TF-IDF). Ensuite, vous entraînerez un modèle de classification.

### Consignes
1. **Chargement des données :** Chargez le dataset SMS Spam Collection à partir du fichier CSV disponible sur Kaggle.
2. **Prétraitement :**
    - Supprimez les caractères spéciaux et convertissez le texte en minuscules.
3. **Bag of Words :**
    - Utilisez `CountVectorizer` pour transformer les textes en une matrice de fréquences.
    - Limitez le vocabulaire à 5000 mots maximum (`max_features=5000`).
4. **TF-IDF :**
    - Utilisez `TfidfVectorizer` pour transformer les textes en une matrice TF-IDF.
    - Gardez les mêmes paramètres que pour BoW.
5. **Modèles de classification :**
    - Entraînez des modèles de classification (`LogisticRegression`, `RandomForestClassifier`, `MLPClassifier`) sur les deux représentations (BoW et TF-IDF).
6. **Évaluation :**
    - Évaluez les performances avec une validation croisée (5 folds) et calculez l’accuracy et le F1-score.

### Questions
- Quelle méthode (BoW ou TF-IDF) donne les meilleurs résultats ? Pourquoi pensez-vous que c’est le cas ?
- Que se passe-t-il pour les métriques si vous diminuez `max_features` ? (Montrez un graphique)

---

## Partie 2 : Classification avec Sentence Transformers

Dans cette partie, vous utiliserez un modèle pré-entraîné de Sentence Transformers pour générer des embeddings de texte, puis vous les utiliserez comme features pour un modèle de classification.

### Consignes
1. Installez la bibliothèque `sentence-transformers`.
2. Chargez un modèle léger comme `all-MiniLM-L6-v2` pour rester "CPU friendly".
3. Transformez chaque texte du jeu de données en un embedding.
4. Utilisez ces embeddings comme features pour entraîner un modèle de classification (`LogisticRegression`, `RandomForestClassifier`, `MLPClassifier`).
5. Évaluez les performances avec une validation croisée (5 folds) et comparez avec les résultats de la Partie 1.

### Questions
- Comment les embeddings se comparent-ils à BoW et TF-IDF en termes de performance ?
- Quels sont les avantages et inconvénients d’utiliser des embeddings pré-entraînés ?

---

## Partie 3 : Le Super Learner

### Qu’est-ce que le Super Learner ?
Le Super Learner est une méthode ensembliste qui combine les prédictions de plusieurs modèles d’apprentissage automatique pour obtenir une performance optimale. Contrairement à un simple vote majoritaire, il utilise une validation croisée pour apprendre une combinaison pondérée des modèles de base (ou "base learners"), optimisant ainsi une métrique comme l’accuracy.

### Algorithme en pseudo-code
```plaintext
Algorithm 1 Super Learner
Require: Données d’entraînement (X, y), liste de modèles de base M = {M1, M2, ..., Mk}, modèle méta Mméta, nombre de plis K
Ensure: Prédictions combinées
1: Diviser (X, y) en K plis pour la validation croisée
2: for chaque pli k = 1 à K do
3: Entraîner chaque modèle Mi sur les K − 1 plis restants
4: Prédire sur le pli k avec chaque Mi, stocker les prédictions Zi,k
5: end for
6: Construire une matrice Z où chaque colonne contient les prédictions d’un modèle Mi sur tous les échantillons
7: Entraîner le modèle méta Mméta sur (Z, y) pour apprendre les poids des modèles de base
8: Pour les nouvelles données Xtest :
9: Prédire avec chaque Mi pour obtenir Ztest
10: Appliquer Mméta sur Ztest pour obtenir la prédiction finale
```

### Consignes
1. **Implémentation du Super Learner :**
    - Implémentez le Super Learner en suivant l’exemple ci-dessus ou utilisez une bibliothèque comme `mlens` (optionnel).
    - Utilisez 3 modèles de base adaptés à la classification de texte : `LogisticRegression`, `RandomForestClassifier`, et `SVC`, en utilisant les features TF-IDF de la Partie 1.
    - Choisissez un modèle méta simple, comme une régression logistique (`LogisticRegression`).
2. **Tests de performance :**
    - Testez les performances (en accuracy et F1-score) du Super Learner avec les features TF-IDF.
    - Présentez les résultats dans un tableau ou un graphique comparatif global (TF-IDF, Embeddings, Super Learner).

### Questions
1. Commentez les graphiques obtenus.
2. Quels sont les poids que votre méta-modèle a attribués à chaque modèle de base ?

---

## Bonus

Il vous est possible de soumettre votre meilleur modèle de la partie 1 ou 2 pour participer à une compétition et comparer vos performances avec celles des autres. Un bonus de 10% et 5% est accordé aux premier et deuxième participants, respectivement. Vous pouvez soumettre votre code [ici](#).

---

## Livrables

- Un notebook Jupyter (ou script Python) avec le code commenté.
- Un court rapport (2 - 3 pages) résumant vos résultats et analyses, avec tableaux ou graphiques.

---

## Ressources principales

- [Documentation de scikit-learn](https://scikit-learn.org/)
- [mlens (optionnel)](https://mlens.readthedocs.io/)
- Implémentation pratique d’un Super Learner
- Dataset SMS Spam Collection sur Kaggle
- Sentence Transformers : [Documentation](https://www.sbert.net/)

---