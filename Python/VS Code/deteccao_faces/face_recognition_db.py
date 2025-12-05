import os
import pickle
import face_recognition
import cv2


class FaceDatabase:
    """Gerencia cadastro e reconhecimento de rostos."""

    def __init__(self, db_path="face_encodings.pkl"):
        self.db_path = db_path
        self.known_faces = {}  # {nome: [encoding1, encoding2, ...]}
        self.load_database()

    def load_database(self):
        """Carrega o banco de dados de rostos cadastrados."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.known_faces = pickle.load(f)
            print(f"Banco de dados carregado: {len(self.known_faces)} "
                  f"pessoas")
        else:
            print("Novo banco de dados criado")

    def save_database(self):
        """Salva o banco de dados de rostos."""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.known_faces, f)
        print(f"Banco de dados salvo: {len(self.known_faces)} pessoas")

    def extract_face_encoding(self, frame, face_location):
        """
        Extrai o encoding de um rosto detectado.

        Args:
            frame: imagem do vídeo
            face_location: localização do rosto (top, right, bottom, left)

        Returns:
            encoding do rosto ou None se falhar
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(
                rgb_frame, [face_location])
            return face_encodings[0] if face_encodings else None
        except Exception as e:
            print(f"Erro ao extrair encoding: {e}")
            return None

    def register_face(self, name, frame, face_location):
        """
        Cadastra um novo rosto ou adiciona variação de um rosto existente.
        Recomenda-se cadastrar 3-5 fotos em ângulos diferentes.

        Args:
            name: nome da pessoa
            frame: imagem do vídeo
            face_location: localização do rosto
        """
        encoding = self.extract_face_encoding(frame, face_location)

        if encoding is None:
            print("Não foi possível extrair características do rosto")
            return False

        if name not in self.known_faces:
            self.known_faces[name] = []

        self.known_faces[name].append(encoding)
        self.save_database()

        total_fotos = len(self.known_faces[name])
        quality = "Excelente" if total_fotos >= 5 else (
            "Boa" if total_fotos >= 3 else "Básica")

        print(f"✓ Rosto de '{name}' cadastrado! "
              f"(Foto {total_fotos}/5 - Qualidade: {quality})")

        if total_fotos < 3:
            print("Dica: Cadastre mais 3-5 fotos em ângulos diferentes "
                  "para melhor reconhecimento")
        return True

    def recognize_face(self, frame, face_location, tolerance=0.6):
        """
        Reconhece um rosto no frame com tolerance otimizado.

        Args:
            frame: imagem do vídeo
            face_location: localização do rosto
            tolerance: tolerância (0.6=permissivo, 0.4=rigoroso)

        Returns:
            (nome, confiança) ou (None, 0)
        """
        encoding = self.extract_face_encoding(frame, face_location)

        if encoding is None:
            return None, 0

        best_match_name = None
        best_match_distance = float('inf')

        for name, encodings in self.known_faces.items():
            distances = face_recognition.compare_faces(
                encodings, encoding, tolerance=tolerance)
            face_distances = face_recognition.face_distance(
                encodings, encoding)

            best_distance = min(face_distances) if (
                face_distances.size > 0) else 1

            if best_distance < best_match_distance:
                best_match_distance = best_distance
                best_match_name = (name if any(distances) else None)

        confidence = (1 - best_match_distance if best_match_distance < 1
                      else 0)
        return best_match_name, confidence

    def list_registered_faces(self):
        """Lista todos os rostos cadastrados com detalhes."""
        if not self.known_faces:
            print("Nenhum rosto cadastrado")
            return
        print("\n" + "="*50)
        print("              ROSTOS CADASTRADOS")
        print("="*50)
        for i, (name, encodings) in enumerate(self.known_faces.items(), 1):
            quality = "Excelente" if len(encodings) >= 5 else (
                "Boa" if len(encodings) >= 3 else "Básica")
            print(f"{i}. {name:20} | Fotos: {len(encodings):2}/5 | "
                  f"Qualidade: {quality}")
        print("="*50 + "\n")

    def delete_face(self, name):
        """Deleta um rosto do banco de dados."""
        if name in self.known_faces:
            del self.known_faces[name]
            self.save_database()
            print(f"✓ Rosto de '{name}' removido!")
            return True
        print(f"✗ '{name}' não encontrado no banco de dados")
        return False
