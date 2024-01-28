import cv2
from tp2 import draw_bounding_boxes, compute_iou
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

# Chemin vers le dossier contenant les images
images_path = 'img1/'

# Chemin vers le fichier contenant les données de détection
detections_data_path = 'det/det.txt'

# Seuil pour le calcul de l'IoU
SIGMA_IOU = 0.25

# Lecture des données de détection
detections = open(detections_data_path, 'r')
detections = detections.readlines()


def hungarian_algorithm(similarity_matrix: List[List[float]]) -> Tuple[List[int], List[int]]:
    """
    Exécute l'algorithme hongrois pour une matrice de similarité.
    (On cherche à trouver la combinaison de lignes et de colonnes qui maximise la somme des valeurs de la matrice.)

    :param similarity_matrix: Matrice de similarité entre les détections et les tracks.
    :return: Indices de lignes et de colonnes optimisés.
    """
    row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
    return row_ind, col_ind


def compute_similarity_matrix_hungarian(detections_infos: List[List[int]], tracks_infos: List[List[int]]) -> List[List[float]]:
    """
    Calcule une matrice de similarité entre les détections et les tracks existantes.

    :param detections_infos: Liste des détections pour une frame.
    :param tracks_infos: Liste des tracks existantes.
    :return: Matrice de similarité.
    """
    similarity_matrix = []
    for detection in detections_infos:
        x1, y1, width1, height1, conf1, _, _ = detection
        a1, a2 = (x1, y1), (x1 + width1, y1 + height1)
        row = []

        for track in tracks_infos:
            x2, y2, width2, height2, conf2, _, _ = track
            b1, b2 = (x2, y2), (x2 + width2, y2 + height2)
            row.append(compute_iou(a1, a2, b1, b2))

        similarity_matrix.append(row)

    return similarity_matrix


def init_detections(detections_lines: List[str]) -> dict:
    """
    Initialise les détections en les organisant par frame.

    :param detections_lines: Liste des lignes contenant les données de détection.
    :return: Dictionnaire de détections organisées par frame.
    """
    detections_dict = {}
    for detection in detections_lines:
        detection = detection.split(',')
        frame, _, x, y, width, height, conf, _, _, _ = [int(float(d)) for d in detection]
        if frame not in detections_dict:
            detections_dict[frame] = []
        detections_dict[frame].append([x, y, width, height, conf, None, 1])
    return detections_dict


def update_tracks(tracks: List[List[int]], detections_dict: dict, i_frame: int, sigma_iou: float, counter: List[int]) -> List[List[int]]:
    """
    Met à jour les tracks existantes et en crée de nouvelles si nécessaire.

    :param tracks: Liste des tracks existantes.
    :param detections_dict: Dictionnaire des détections organisées par frame.
    :param i_frame: Indice de la frame actuelle.
    :param sigma_iou: Seuil IoU pour la mise à jour des tracks.
    :param counter: Compteur pour assigner des ID uniques aux nouvelles tracks.
    (On utilise une liste pour le compteur pour pouvoir le modifier par référence.)

    :return: Liste mise à jour des tracks.
    """
    matrix = compute_similarity_matrix_hungarian(detections_dict[i_frame], tracks)
    row_ind, col_ind = hungarian_algorithm(matrix)

    for i in range(len(row_ind)):
        if matrix[row_ind[i]][col_ind[i]] < sigma_iou:
            tracks[col_ind[i]][5] = -1
        else:
            track_id = tracks[col_ind[i]][5]
            tracks[col_ind[i]] = detections_dict[i_frame][row_ind[i]]
            tracks[col_ind[i]][5] = track_id
            tracks[col_ind[i]][6] = matrix[row_ind[i]][col_ind[i]]

    for i in range(len(detections_dict[i_frame])):
        if detections_dict[i_frame][i][5] is None:
            detections_dict[i_frame][i][5] = counter[0]
            tracks.append(detections_dict[i_frame][i])
            counter[0] += 1

    tracks = [track for track in tracks if track[5] != -1]
    return tracks


def process_frames_hungarian(detections_list: List[str], sigma_iou: float) -> List[str]:
    """
    Traite chaque frame en utilisant l'algorithme hongrois pour la mise à jour des tracks.

    :param detections_list: Liste des lignes contenant les informations de détection.
    :param sigma_iou: Seuil IoU pour la mise à jour des tracks.
    :return: Liste des résultats pour chaque frame.
    """
    detections_dict = init_detections(detections_list)
    tracks = []
    results = []
    counter = [0]

    for i_frame in range(1, 526):
        tracks = update_tracks(tracks, detections_dict, i_frame, sigma_iou, counter)

        # On enregistre les résultats obtenus pour cette frame
        # for track in tracks:
        #     results.append(f"{i_frame},{track[5]},{track[0]},{track[1]},{track[2]},{track[3]},{track[6]},-1,-1,-1")
        for detection in detections_dict[i_frame]:
            results.append(f"{i_frame},{detection[5]},{detection[0]},{detection[1]},{detection[2]},{detection[3]},{detection[4]},-1,-1,-1")

        image_path = images_path + str(i_frame).zfill(6) + '.jpg'
        image = draw_bounding_boxes(image_path, detections_dict[i_frame])
        cv2.imshow('image', image)
        cv2.waitKey(50)

    return results


def save_results(results: List[str]) -> None:
    """
    Sauvegarde les résultats dans un fichier.

    :param results: Liste des résultats à sauvegarder.
    """
    with open('TP3.txt', 'w') as file:
        for result in results:
            file.write(result + '\n')


if __name__ == '__main__':
    res = process_frames_hungarian(detections, SIGMA_IOU)
    save_results(res)
    cv2.destroyAllWindows()