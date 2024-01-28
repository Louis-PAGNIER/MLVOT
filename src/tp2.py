import cv2
from typing import Tuple, List

# Chemin vers le dossier contenant les images
images_path = 'img1/'

# Chemin vers le fichier contenant les données de détection
detections_data_path = 'det/det.txt'

# Seuil pour le calcul de l'IoU
SIGMA_IOU = 0.25

# Lecture des données de détection
detections = open(detections_data_path, 'r')
detections = detections.readlines()


def compute_iou(a1: Tuple[int, int], a2: Tuple[int, int], b1: Tuple[int, int], b2: Tuple[int, int]) -> float:
    """
    Calcule l'IoU entre 2 rectangles.

    :param a1: Point en haut à gauche du 1er rectangle.
    :param a2: Point en bas à droite du 1er rectangle.
    :param b1: Point en haut à gauche du 2ème rectangle.
    :param b2: Point en bas à droite du 2ème rectangle.
    :return: Le score IoU.
    """
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2

    intersection = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
    return intersection / union


def compute_similarity_matrix(detections_dict: dict, i: int, j: int) -> List[List[float]]:
    """
    Calcule une matrice de similarité entre les détections de deux frames.

    :param detections_dict: Dictionnaire contenant les détections.
    :param i: Indice de la frame 1.
    :param j: Indice de la frame 2.
    :return: Matrice de similarité.
    """
    similarity_matrix = []
    for detection1 in detections_dict[i]:
        x1, y1, width1, height1, conf1, counter1 = detection1
        a1, a2 = (x1, y1), (x1 + width1, y1 + height1)
        row = []
        for detection2 in detections_dict[j]:
            x2, y2, width2, height2, conf2, counter2 = detection2
            b1, b2 = (x2, y2), (x2 + width2, y2 + height2)
            row.append(compute_iou(a1, a2, b1, b2))
        similarity_matrix.append(row)
    return similarity_matrix


def draw_bounding_boxes(image_path: str, detections_infos: List[List[int]]) -> cv2.Mat:
    """
    Dessine des boîtes englobantes sur l'image en fonction des détections et affiche l'ID des personnes détectées.

    :param image_path: Chemin de l'image.
    :param detections_infos: Liste des détections pour cette image.
    :return: Image avec les boîtes englobantes dessinées.
    """
    image = cv2.imread(image_path)
    for detection in detections_infos:
        if detection[5] != -1:
            x, y, width, height, conf, counter, *_ = detection
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(image, str(counter), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image


def process_frames(detections_lines: List[str], sigma_iou: float) -> None:
    """
    Traite chaque frame pour dessiner les boîtes englobantes et assigner des identifiants aux objets détectés.

    :param detections_lines: Liste des lignes contenant les informations de détection.
    :param sigma_iou: Seuil IoU pour considérer deux détections comme similaires.
    """

    # On commence par créer un dictionnaire contenant les détections pour chaque frame
    detections_dict = {}
    for detection in detections_lines:
        detection = detection.split(',')
        frame, _, x, y, width, height, conf, _, _, _ = [int(float(d)) for d in detection]
        if frame not in detections_dict:
            detections_dict[frame] = []
        detections_dict[frame].append([x, y, width, height, conf, None])

    # On traite chaque frame en utilisant la matrice de similarité
    counter = 0
    for i_frame in range(1, 526):
        matrix = compute_similarity_matrix(detections_dict, i_frame, i_frame + 1)
        for i in range(len(matrix)):
            max_iou = max(matrix[i])

            if detections_dict[i_frame][i][5] is None:
                # Si la détection n'a pas encore d'identifiant, on lui en assigne un nouveau
                detections_dict[i_frame][i][5] = counter
                counter += 1

            if max_iou > sigma_iou:
                # Si la détection a une similarité suffisante avec une autre détection, on lui assigne le même ID
                j = matrix[i].index(max_iou)
                detections_dict[i_frame + 1][j][5] = detections_dict[i_frame][i][5]

        image_path = images_path + str(i_frame).zfill(6) + '.jpg'
        image = draw_bounding_boxes(image_path, detections_dict[i_frame])
        cv2.imshow('image', image)
        cv2.waitKey(50)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_frames(detections, SIGMA_IOU)
