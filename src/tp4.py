import cv2
import numpy as np
from TP2.tp2 import compute_iou
from tp3 import hungarian_algorithm
from KalmanFilter import KalmanFilter
from typing import List, Tuple, Dict

# Chemin vers le dossier contenant les images
images_path = 'img1/'

# Chemin vers le fichier contenant les données de détection
detections_data_path = 'det/det.txt'

# Seuil pour le calcul de l'IoU
SIGMA_IOU = 0.1

# Lecture des données de détection
detections = open(detections_data_path, 'r')
detections = detections.readlines()


class Track:
    """
    Classe représentant une piste de suivi (track) pour une détection.

    :param x: Coordonnée x du coin supérieur gauche de la détection.
    :param y: Coordonnée y du coin supérieur gauche de la détection.
    :param width: Largeur de la détection.
    :param height: Hauteur de la détection.
    :param conf_d: Confiance de la détection.
    :param track_id: Identifiant de la piste de suivi.
    """
    def __init__(self, x: int, y: int, width: int, height: int, conf_d: float, track_id: int = None) -> None:
        self.width = width
        self.height = height
        self.centroid = (x + width / 2, y + height / 2)
        self.old_centroid = self.centroid
        self.conf_d = conf_d
        self.conf_h = 1
        self.track_id = track_id
        self.kalman_filter = KalmanFilter(dt=0.3, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1, start_x=self.centroid[0], start_y=self.centroid[1])
        self.predict()

    def get_coordinates(self) -> Tuple[int, int, int, int]:
        # Retourne les coordonnées actuelles de la track: (x, y, largeur, hauteur)
        return self.kalman_filter.x_k_[0][0] - self.width / 2, self.kalman_filter.x_k_[1][0] - self.height / 2, self.width, self.height

    def predict(self) -> None:
        # Met à jour la track avec la prédiction du filtre de Kalman
        tmp_centroid = self.kalman_filter.x_k_[0][0], self.kalman_filter.x_k_[1][0]
        self.kalman_filter.predict()
        self.old_centroid = tmp_centroid
        self.centroid = self.kalman_filter.x_k_[0][0], self.kalman_filter.x_k_[1][0]

    def update(self, z_k: np.ndarray) -> None:
        # Met à jour la track avec une vraie détection
        self.kalman_filter.update(z_k)
        self.centroid = self.kalman_filter.x_k_[0][0], self.kalman_filter.x_k_[1][0]

    def clone(self) -> 'Track':
        # Créer une copie de la track
        return Track(int(self.centroid[0] - self.width / 2), int(self.centroid[1] - self.height / 2), self.width, self.height, self.conf_d, self.track_id)


def compute_similarity_matrix_hungarian(detections: List[Track], tracks: List[Track]) -> List[List[float]]:
    """
    Calcule une matrice de similarité entre les détections et les tracks existantes.

    :param detections: Liste des objets Track pour les détections actuelles.
    :param tracks: Liste des objets Track pour les tracks existants.
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
            row.append(compute_iou(a1, a2, b1, b2))

        similarity_matrix.append(row)

    return similarity_matrix


def init_detections(detections_lines: List[str]) -> Dict[int, List[Track]]:
    """
    Initialise les détections en les organisant par frame.

    :param detections_lines: Liste des lignes contenant les données de détection.
    :return: Dictionnaire de détections organisées par frame, avec chaque détection représentée par un objet Track.
    """
    detections_dict = {}

    for detection in detections_lines:
        detection = detection.split(',')
        frame, _, x, y, width, height, conf_d, _, _, _ = [int(float(d)) for d in detection]

        if frame not in detections_dict:
            detections_dict[frame] = []

        detections_dict[frame].append(Track(x, y, width, height, conf_d, None))

    return detections_dict


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
    matrix = compute_similarity_matrix_hungarian(detections_dict[i_frame], tracks)
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


def draw_bounding_boxes(image_path: str, detections: List[Track]) -> np.ndarray:
    """
    Dessine des boîtes englobantes sur l'image pour chaque détection.

    :param image_path: Chemin de l'image à traiter.
    :param detections: Liste des objets Track pour les détections à dessiner.
    :return: Image avec les boîtes englobantes dessinées.
    """
    image = cv2.imread(image_path)
    for d in detections:
        if d.track_id != -1:
            x, y, width, height = d.get_coordinates()
            x, y, width, height = int(x), int(y), int(width), int(height)
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(image, str(d.track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image


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

        # On associe les détections aux tracks, et on crée de nouvelles tracks si nécessaire
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
    with open('TP4.txt', 'w') as file:
        for result in results:
            file.write(result + '\n')
