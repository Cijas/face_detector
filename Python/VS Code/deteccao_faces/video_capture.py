import cv2


def iniciar_captura():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Não foi possível acessar a webcam.")
        return None

    print("Webcam iniciada...")
    return cam
