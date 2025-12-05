import cv2
from video_capture import iniciar_captura
from face_detection import detectar_faces, convert_bbox_to_face_location
from face_recognition_db import FaceDatabase
from display import desenhar_rosto

print("O script foi iniciado com sucesso... Aguarde")


def main():
    cam = iniciar_captura()

    if cam is None:
        exit()

    # Aumentar resolução da câmera para melhor qualidade
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Obter dimensões reais da câmera
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Configuração do VideoWriter para gravar o vídeo processado
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (width, height))

    print("Definindo ROI (Região de Interesse)...")

    # Ajuste do ROI conforme necessário
    ROI_X1, ROI_Y1 = 150, 100
    ROI_X2, ROI_Y2 = width - 150, height - 100

    # Inicializar banco de dados de rostos
    face_db = FaceDatabase()

    frame_count = 0
    registration_mode = False
    registration_name = ""

    print("\n" + "="*60)
    print("             SISTEMA DE RECONHECIMENTO FACIAL")
    print("="*60)
    print("\n=== CONTROLES ===")
    print("'r' - Modo de cadastro (digite o nome)")
    print("'s' - Salvar rosto no modo cadastro")
    print("'l' - Listar rostos cadastrados")
    print("'d' - Deletar rosto cadastrado")
    print("'t' - Alternar tolerance (0.6 -> 0.4 -> 0.6)")
    print("'q' - Sair da aplicação")
    print("="*60 + "\n")

    tolerance = 0.6

    while True:
        ret, frame = cam.read()

        if not ret or frame is None:
            print("Erro ao capturar frame da webcam")
            break

        # Detectar faces
        if frame_count % 2 == 0:
            faces = detectar_faces(frame)

        frame_count += 1

        face_names = []
        face_confidences = []

        # Processar cada face detectada
        for (x, y, w, h) in faces:
            face_location = convert_bbox_to_face_location(x, y, w, h)

            if registration_mode:
                # Modo de cadastro
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 3)
                cv2.putText(frame, f"Cadastrando: {registration_name}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                cv2.putText(
                    frame, "Pressione 's' para salvar",
                    (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
                face_names.append(registration_name)
                face_confidences.append(0)
            else:
                # Modo de reconhecimento
                name, confidence = face_db.recognize_face(
                    frame, face_location, tolerance=tolerance)
                face_names.append(name if name else "Desconhecido")
                face_confidences.append(confidence)

            # Aplica borramento nos rostos dentro da ROI
            if (x >= ROI_X1 and y >= ROI_Y1
                    and x + w <= ROI_X2 and y + h <= ROI_Y2):
                face_region = frame[y:y + h, x:x + w]
                blurred_region = cv2.GaussianBlur(
                    face_region, (99, 99), 30)
                frame[y:y + h, x:x + w] = blurred_region

        # Desenha a caixa da face
        frame = desenhar_rosto(frame, faces, face_names,
                               face_confidences, ROI_X1, ROI_Y1,
                               ROI_X2, ROI_Y2)

        # Exibir status do tolerance
        status_text = (f"Tolerance: {tolerance} | "
                       f"Frames: {frame_count} | "
                       f"""Modo: {'CADASTRO'
                                if registration_mode else 'RECONHECIMENTO'}""")
        cv2.putText(frame, status_text, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Grava o frame processado
        out.write(frame)

        cv2.imshow("Detecting faces", frame)

        # Captura input do teclado
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nEncerrando aplicação...")
            break
        elif key == ord('r'):
            registration_mode = not registration_mode
            if registration_mode:
                registration_name = input(
                    "\nDigite o nome da pessoa: ").strip()
                if not registration_name:
                    registration_name = "Pessoa"
                print(f"✓ Modo cadastro ativo para: '{registration_name}'")
                print("→ Coloque o rosto na frente da câmera")
                print("→ Pressione 's' para salvar cada foto")
            else:
                print("✓ Modo cadastro desativado")

        elif key == ord('s') and registration_mode and len(faces) > 0:
            # 's' para salvar o rosto no modo de cadastro
            face_location = convert_bbox_to_face_location(
                faces[0][0], faces[0][1], faces[0][2], faces[0][3])
            face_db.register_face(registration_name, frame, face_location)

        elif key == ord('l'):
            face_db.list_registered_faces()

        elif key == ord('d'):
            name_to_delete = input(
                "\nDigite o nome do rosto a deletar: ").strip()
            face_db.delete_face(name_to_delete)

        elif key == ord('t'):
            tolerance = 0.4 if tolerance == 0.6 else 0.6
            modo = "Rigoroso" if tolerance == 0.4 else "Permissivo"
            print(f"✓ Tolerance alterado para {tolerance} ({modo})")

    cam.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Total de frames processados: {frame_count}")
    print("Aplicação encerrada.")


if __name__ == "__main__":
    main()
