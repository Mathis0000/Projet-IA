import time
import cv2
import numpy as np
import mss
import pyautogui
import tensorflow as tf
import os

# Définir la fonction de perte et l'optimiseur
loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam()
# Définir la variable d'environnement
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Vérifier si la variable d'environnement est définie
if 'TF_ENABLE_ONEDNN_OPTS' in os.environ:
    print("La variable d'environnement est définie.")
else:
    print("La variable d'environnement n'est pas définie.")

from keras.models import load_model
import time
def create_model():
    # Créer un modèle séquentiel
    model = tf.keras.models.Sequential()

    # Ajouter une couche de convolution 2D
    model.add(tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 1)))
    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    # Aplatir le tenseur en une dimension
    model.add(tf.keras.layers.Flatten())

    # Ajouter une couche dense
    model.add(tf.keras.layers.Dense(512, activation='relu'))

    # Ajouter une couche de sortie avec deux neurones (pour les deux actions possibles)
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    # Compiler le modèle avec une fonction de perte et un optimiseur
    model.compile(loss=loss_fn, optimizer=optimizer)

    # Retourner le modèle
    return model

save_freq = 5 # sauvegarder le modèle toutes les 5 minutes


try:
    # Charger le modèle sauvegardé si disponible
    if os.path.exists('model.h5'):
        model = load_model('model.h5')
        print('\n  \n    \n   Modèle chargé avec succès !\n')
    else:
        # Créer un nouveau modèle si aucun modèle sauvegardé n'est disponible
        model = create_model()
        model.save('model.h5')
        print('       \n        \n    \n           Nouveau modèle créé !\n')

except Exception as e:
    # Afficher l'erreur
    print(" \n    \n    \n   Une erreur s'est produite :", str(e))

    # Afficher le chemin d'accès actuel
    print("\n\n\nChemin d'accès actuel :", os.getcwd())

import screeninfo

# Détecter les dimensions de l'écran
screen = screeninfo.get_monitors()[0]
image_width = screen.width
image_height = screen.height

# Définir la région de la capture d'écran
monitor = {"top": 0, "left": 0, "width": image_width, "height": image_height}

# Créer un objet MSS pour prendre des captures d'écran
sct = mss.mss()

# Définir les seuils de détection d'obstacles en fonction des dimensions de l'écran
min_obstacle_area = int(image_width * image_height * 0.001)  # 0.1% de la surface de l'écran
max_obstacle_area = int(image_width * image_height * 0.1)  # 10% de la surface de l'écran
min_obstacle_width = int(image_width * 0.02)  # 2% de la largeur de l'écran
max_obstacle_height = int(image_height * 0.5)  # 50% de la hauteur de l'écran
min_obstacle_height = int(image_height * 0.02)  # 2% de la hauteur de l'écran

# Définir les seuils de détection du joueur en fonction des dimensions de l'écran
min_player_area = int(image_width * image_height * 0.005)  # 0.5% de la surface de l'écran
max_player_area = int(image_width * image_height * 0.05)  # 5% de la surface de l'écran
min_player_width = int(image_width * 0.03)  # 3% de la largeur de l'écran
max_player_height = int(image_height * 0.5)  # 50% de la hauteur de l'écran

# Définir les seuils de prise de décision
jump_threshold = 150
stay_threshold = 300

# Définir l'espace d'état et l'espace d'action
state_space = (84, 84, 1)
action_space = 2


# Créer un modèle de deep learning
model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0, input_shape=state_space),
    tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
    tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(action_space, activation='linear')
])



# Définir la fonction de récompense
def reward_fn(state, action, done):
    # Récompense négative pour chaque étape
    reward = -1

    # Récompense positive pour avoir sauté par-dessus un obstacle
    if done:
        reward += 10

    return reward


# Définir la fonction pour collecter les données d'expérience
def collect_experience(model, state, epsilon):
    # Sélectionner une action en fonction de la politique actuelle
    if np.random.rand() < epsilon:
        action = np.random.choice(action_space)
    else:
        action = np.argmax(model.predict(state))

    # Exécuter l'action dans l'environnement et obtenir le nouvel état et la récompense
    next_state, reward, done = take_action(action)

    # Stocker les données d'expérience
    experience = (state, action, reward, next_state, done)

    return experience

def detect_objects(image_gray):
    # Appliquer un filtre de seuil pour obtenir une image binaire
    _, image_bin = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)

    # Trouver les contours dans l'image binaire
    contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrer les contours en fonction de leur taille et de leur forme
    obstacles = []
    player = None
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Si le contour est trop petit ou trop grand, l'ignorer
        if area < min_obstacle_area or area > max_obstacle_area:
            continue

        # Si le contour est trop large ou trop étroit, l'ignorer
        if aspect_ratio < 0.25 or aspect_ratio > 4.0:
            continue

        # Si le contour est trop haut ou trop bas, l'ignorer
        if y < image_height * 0.2 or y > image_height * 0.8:
            continue

        # Si le contour est un obstacle, l'ajouter à la liste
        obstacles.append(contour)

        # Si le contour est le personnage, le sauvegarder
        if area > min_player_area and area < max_player_area:
            player = contour

    return obstacles, player
def draw_contours(image, contours, color=(0, 255, 0)):
    for contour in contours:
        cv2.drawContours(image, [contour], -1, color, 2)

def take_action(action):
    # Prendre une capture d'écran
    screenshot = sct.grab(monitor)

    # Convertir la capture d'écran en image NumPy
    image = np.array(screenshot)

    # Convertir l'image en niveaux de gris
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détecter les obstacles et le personnage
    obstacles, player = detect_objects(image_gray)

    # Dessiner les contours détectés sur l'image
    draw_contours(image, obstacles)

    # Décider de l'action à prendre en fonction de la distance de l'obstacle le plus proche
    if action == 0:
        # Ne pas sauter
        pass
    else:
        # Sauter
        pyautogui.press('space')

    # Redimensionner l'image et la convertir en noir et blanc
    state = cv2.resize(image_gray, (84, 84))
    state = np.expand_dims(state, axis=2)
    state = np.expand_dims(state, axis=0)

    # Vérifier si le jeu est terminé
    done = False
    if len(obstacles) == 0:
        done = True
                                                   
    # Calculer la récompense
    reward = reward_fn(state, action, done)

    return state, reward, done



# Définir la fonction pour entraîner le modèle
def train_model(model, experiences, gamma, episode):
    # Calculer la cible Q pour chaque expérience
    for experience in experiences:
        state, action, reward, next_state, done = experience
        target_q = reward
        if not done:
            target_q += gamma * np.max(model.predict(next_state))
        # Calculer la perte et effectuer une étape d'optimisation
        with tf.GradientTape() as tape:
            current_q = model(state)
            loss = loss_fn(target_q, current_q[action])
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Afficher les progrès de l'entraînement toutes les 10 itérations
                     
        print('Épisode :', episode, 'Perte :', loss.numpy())

        
        
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        # Exploration : sélectionner une action aléatoire
        action = np.random.randint(0, 2)
    else:
        # Exploitation : sélectionner l'action avec la valeur Q maximale
        q_values = model.predict(state)
        action = np.argmax(q_values[0])

    return action

# Boucle d'entraînement
# Boucle d'entraînement
num_episodes = 10
gamma = 0.99
save_freq = 10

for episode in range(num_episodes):
    state = take_action(0)[0]
    experiences = []
    epsilon = max(0.1, 1.0 - episode / num_episodes)
    for step in range(1000):
        # Sélectionner une action en fonction de la politique actuelle
        action = select_action(state, epsilon)

        # Exécuter l'action dans l'environnement et obtenir le nouvel état et la récompense
        next_state, reward, done = take_action(action)

        # Stocker les données d'expérience
        experiences.append((state, action, reward, next_state, done))

        # Mettre à jour l'état actuel
        state = next_state

        # Arrêter l'épisode si le jeu est terminé
        if done:
            break

def create_model():
    # Créer un modèle séquentiel
    model = tf.keras.models.Sequential()

    # Ajouter une couche de convolution 2D
    model.add(tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 1)))
    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    # Aplatir le tenseur en une dimension
    model.add(tf.keras.layers.Flatten())

    # Ajouter une couche dense
    model.add(tf.keras.layers.Dense(512, activation='relu'))

    # Ajouter une couche de sortie avec deux neurones (pour les deux actions possibles)
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    # Compiler le modèle avec une fonction de perte et un optimiseur
    model.compile(loss=loss_fn, optimizer=optimizer)

    # Retourner le modèle
    return model



# Boucle d'entraînement
for episode in range(num_episodes):
    # Réinitialiser l'environnement
    state = take_action(0)[0]
    done = False
    experiences = []
    start_time = time.time()

    while not done:
        # Sélectionner une action en fonction de la politique actuelle
        action = select_action(state, epsilon)

        # Exécuter l'action et obtenir le nouvel état, la récompense et le booléen "done"
        next_state, reward, done, _ = take_action(action)

        # Stocker l'expérience dans la mémoire
        experiences.append((state, action, reward, next_state, done))

        # Mettre à jour l'état actuel
        state = next_state

    # Calculer le temps écoulé depuis le début de l'épisode
    elapsed_time = time.time() - start_time

    # Entraîner le modèle sur les données collectées
    train_model(model, experiences, gamma)

    # Sauvegarder le modèle toutes les N minutes
    if elapsed_time > 60 * save_freq:
        model.save('model.h5')
        print(f"Modèle sauvegardé après {episode + 1} épisodes.")
