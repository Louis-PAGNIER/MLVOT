from functools import cache

import cv2
import numpy as np

from TP2.tp2 import compute_iou
from tp3 import hungarian_algorithm
from tp4 import Track, init_detections, draw_bounding_boxes

from typing import List, Dict

from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity

# Chemin vers le dossier contenant les images
images_path = 'img1/'

# Chemin vers le fichier contenant les données de détection
detections_data_path = 'det/det.txt'

# Seuil pour le calcul de l'IoU
SIGMA_IOU = 0.25

# Lecture des données de détection
detections = open(detections_data_path, 'r')
detections = detections.readlines()

# Importation du modèle ResNet18 pré-entraîné sur ImageNet
resnet_model = models.resnet18(pretrained='imagenet')
scaler = transforms.Resize((224, 224), antialias=True)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
layer = resnet_model._modules.get('avgpool')


@cache
def extract_features(img: torch.Tensor) -> torch.Tensor:
    """
    Extrait les features d'une image avec le modèle ResNet18.

    :param img: Image à traiter
    :return: Vecteur de features
    """
    t_img = Variable(normalize(scaler(img)).unsqueeze(0))
    my_embedding = torch.zeros(512)

    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))

    # On récupère les features de l'avant-dernière couche
    h = layer.register_forward_hook(copy_data)
    resnet_model(t_img)
    h.remove()

    return my_embedding


def get_distance(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """
    Calcule la distance cosinus entre deux vecteurs de features.
    :param embedding1: Le premier vecteur de features.
    :param embedding2: Le deuxième vecteur de features.
    :return: La distance cosinus entre les deux vecteurs.
    """
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


def compute_similarity_matrix_hungarian(detections: List[Track], tracks: List[Track], img: Image) -> List[List[float]]:
    """
    Calcule une matrice de similarité entre les détections et les tracks existantes.

    :param detections: Liste des objets Track pour les détections actuelles.
    :param tracks: Liste des objets Track pour les tracks existants.
    :param img: Image actuelle.
    :return: Matrice de similarité entre les détections et les tracks.
    """
    similarity_matrix = []
    for detection in detections:
        x1, y1, width1, height1 = detection.get_coordinates()
        a1, a2 = (x1, y1), (x1 + width1, y1 + height1)
        row = []
        for track in tracks:
            x2, y2, width2, height2 = track.get_coordinates()
            b1, b2 = (x2, y2), (x2 + width2, y2 + height2)
            score = compute_iou(a1, a2, b1, b2)

            img1 = img.crop((x1, y1, x1 + width1, y1 + height1)).convert('RGB')
            img2 = img.crop((x2, y2, x2 + width2, y2 + height2)).convert('RGB')

            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)

            embedding1 = extract_features(img1)
            embedding2 = extract_features(img2)

            score += (get_distance(embedding1, embedding2) - 0.5)
            score /= 2
            row.append(score)

        similarity_matrix.append(row)

    return similarity_matrix


def update_tracks(tracks: List[Track], detections_dict: Dict[int, List[Track]], i_frame: int, sigma_iou: float, counter: List[int]) -> List[Track]:
    """
    Met à jour les tracks existantes et en crée de nouvelles si nécessaire.

    :param tracks: Liste des tracks existantes.
    :param detections_dict: Dictionnaire des détections organisées par frame.
    :param i_frame: Indice de la frame actuelle.
    :param sigma_iou: Seuil IoU pour la mise à jour des tracks.
    :param counter: Compteur pour assigner des ID uniques aux nouvelles tracks.
    :return: Liste mise à jour des tracks.
    """
    img = Image.open(images_path + str(i_frame).zfill(6) + '.jpg')
    matrix = compute_similarity_matrix_hungarian(detections_dict[i_frame], tracks, img)
    row_ind, col_ind = hungarian_algorithm(matrix)

    seen = set()

    for i in range(len(row_ind)):
        if matrix[row_ind[i]][col_ind[i]] < sigma_iou:
            tracks[col_ind[i]].track_id = -1
        else:
            track = tracks[col_ind[i]]
            detection = detections_dict[i_frame][row_ind[i]]
            detection.track_id = track.track_id
            seen.add(detection.track_id)

    for i in range(len(detections_dict[i_frame])):
        if detections_dict[i_frame][i].track_id is None:
            detections_dict[i_frame][i].track_id = counter[0]
            tracks.append(detections_dict[i_frame][i].clone())
            seen.add(counter[0])
            counter[0] += 1

    tracks = [track for track in tracks if track.track_id != -1 and track.track_id in seen]

    return tracks


def process_frames_kalman(detections_list: List[str], sigma_iou: float) -> List[str]:
    """
    Traite chaque frame en utilisant un filtre de Kalman pour suivre les détections.

    :param detections_list: Liste des lignes contenant les informations de détection.
    :param sigma_iou: Seuil IoU pour la mise à jour des tracks.
    :return: Liste des résultats pour chaque frame.
    """

    detections_dict = init_detections(detections_list)
    tracks = []
    counter = [0]
    results = []

    for i in range(len(detections_dict[1])):
        if detections_dict[1][i].track_id is None:
            detections_dict[1][i].track_id = counter[0]
            tracks.append(detections_dict[1][i].clone())
            counter[0] += 1

    for i_frame in range(1, 526):

        # On prédit la position des tracks
        for track in tracks:
            track.predict()
        tracks = update_tracks(tracks, detections_dict, i_frame, sigma_iou, counter)

        # On met à jour la position des tracks avec les détections associées
        for track in tracks:
            for detection in detections_dict[i_frame]:
                if track.track_id == detection.track_id:
                    track.update(np.array([[detection.centroid[0]], [detection.centroid[1]]]))
                    break

        # On enregistre les résultats obtenus pour cette frame
        for detection in detections_dict[i_frame]:
            results.append(f"{i_frame},{detection.track_id},{detection.get_coordinates()[0]},{detection.get_coordinates()[1]},{detection.get_coordinates()[2]},{detection.get_coordinates()[3]},-1,-1,-1,-1")

        image_path = images_path + str(i_frame).zfill(6) + '.jpg'
        image = draw_bounding_boxes(image_path, detections_dict[i_frame])

        for track in tracks:
            x, y, width, height = track.get_coordinates()
            cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), (0, 0, 255), 2)

        cv2.imshow('image', image)
        cv2.waitKey(100)

    return results


if __name__ == '__main__':
    results = process_frames_kalman(detections, SIGMA_IOU)
    with open('TP5__.txt', 'w') as file:
        for result in results:
            file.write(result + '\n')
