
# Generation de Musique avec un RNN

## Introduction
Ce projet est une tentative de génération de musique à l'aide de réseaux LSTM. Mon objectif était de comprendre les bases des réseaux neuronaux récurrents (RNN) et leur application dans le domaine musical.
J'ai essayé de le réaliser en m'inspirant de plusieurs sources, notamment des tutoriels sur YouTube, des articles et des projets similaires. Ces ressources m'ont beaucoup aidé à comprendre les concepts et à les appliquer dans ce projet.

## Motivation
L'idée est née de ma curiosité pour les applications créatives de l'intelligence artificielle. J'ai voulu explorer comment un modèle peut apprendre des motifs dans des données musicales et créer quelque chose de nouveau ainsi qu'apprendre en exerçant

## Étapes du Projet

-  **Préparation des données** : Extraction et encodage des notes à partir de fichiers MIDI.
-  **Construction du modèle** : Réseau LSTM capable de prédire la prochaine note dans une séquence.
-  **Entraînement et génération** : Le modèle est formé sur des séquences musicales et utilisé pour créer de nouvelles mélodies.

## Résultats
Actuellement, le modèle ne génère pas encore des séquences musicales convaincantes. La précision reste faible, mais cela m'a permis d'acquérir une meilleure compréhension des étapes suivantes :
- Préparation des données musicales.
- Construction et entraînement de modèles LSTM.
- Utilisation de bibliothèques comme `Music21` pour manipuler les données.

## Défis rencontrés
- Manque de données.
- Modèle pas assez convaincant(besoin d'énormement d'optimisation).

## Prérequis

Pour exécuter ce projet, tu dois installer les bibliothèques suivantes :

``
pip install tensorflow numpy pandas music21
``


## Remarque

L'exécution de ce projet peut prendre un temps considérable, surtout lors de l'entraînement du modèle. Cela dépend de la taille des données et de la puissance de votre machine (GPU recommandé pour des performances optimales).





