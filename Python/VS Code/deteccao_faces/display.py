import cv2


def desenhar_rosto(frame, faces, face_names, face_confidences,
                   ROI_X1, ROI_Y1, ROI_X2, ROI_Y2):
    """
    Desenha retângulos, nomes e confiança dos rostos detectados.

    Args:
        frame: imagem do vídeo
        faces: lista de coordenadas das faces (x, y, w, h)
        face_names: lista de nomes reconhecidos
        face_confidences: lista de confiança de reconhecimento
        ROI_X1, ROI_Y1, ROI_X2, ROI_Y2: coordenadas da ROI
    """
    # Desenha a ROI em azul
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2),
                  (255, 0, 0), 2)
    cv2.putText(frame, "ROI", (ROI_X1 + 5, ROI_Y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    for i, (x, y, w, h) in enumerate(faces):
        name = face_names[i] if i < len(face_names) else "Desconhecido"
        confidence = face_confidences[i] if i < len(
            face_confidences) else 0

        # Cor baseada no reconhecimento
        if name != "Desconhecido":
            color = (0, 255, 0)  # Verde para reconhecido
        else:
            color = (0, 165, 255)  # Laranja para desconhecido

        # Desenha o retângulo ao redor da face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Desenha o nome e confiança
        label = f"{name}"
        if confidence > 0:
            label += f" ({confidence:.2f})"

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Verifica se a face está dentro da ROI
        if (x >= ROI_X1 and x + w <= ROI_X2
                and y >= ROI_Y1 and y + h <= ROI_Y2):
            cv2.putText(frame, "Na ROI", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame
