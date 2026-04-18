import cv2
import numpy as np


def extract_color_histogram(image_bytes: bytes, bins: int = 32) -> np.ndarray:
    """
    Extrae un histograma de color RGB de una imagen.
    - Redimensiona a 128x128
    - Calcula histograma por canal (B, G, R) con `bins` bins
    - Normaliza y concatena → vector de bins*3 dimensiones (96 dims)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")

    img = cv2.resize(img, (128, 128))

    histograms = []
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist.flatten())

    return np.concatenate(histograms)
