import cv2

# Carrega o classificador em cascata para detecção de faces
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


def detectar_faces(frame):
    """
    Detecta faces na imagem com robustez melhorada para câmeras de baixa
    qualidade.

    Args:
        frame: frame de vídeo (numpy array)

    Returns:
        Lista de coordenadas das faces detectadas [(x, y, w, h), ...]
    """
    # Pré-processamento para melhorar detecção em câmeras de baixa qualidade

    # 1. Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Equalizar histograma para melhorar contraste
    gray = cv2.equalizeHist(gray)

    # 3. Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Suavizar levemente para reduzir ruído
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Detecta faces com parâmetros otimizados
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.50,      # Reduzido para maior sensibilidade
        minNeighbors=4,        # Reduzido para detectar faces menores
        minSize=(30, 30),      # Tamanho mínimo da face
        maxSize=(300, 300),    # Tamanho máximo da face
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces
