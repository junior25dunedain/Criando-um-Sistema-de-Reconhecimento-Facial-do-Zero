import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Inicializar o detector de faces
detector = MTCNN()

# Carregar o modelo FaceNet pré-treinado
model = load_model('/caminho/para/facenet_model.h5')

# Função para extrair o embedding facial
def get_face_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Função para classificar a face detectada
def classify_face(face_embedding, known_embeddings, known_labels):
    similarities = cosine_similarity([face_embedding], known_embeddings)
    label = known_labels[np.argmax(similarities)]
    return label

# Carregar uma imagem
image = cv2.imread('/caminho/para/sua/imagem.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detectar faces
result = detector.detect_faces(image_rgb)

# Dados fictícios para embeddings conhecidos e labels (você deve substituir isso pelos seus dados reais)
known_embeddings = np.random.rand(5, 128)  # Exemplo de embeddings conhecidos
known_labels = ['Pessoa 1', 'Pessoa 2', 'Pessoa 3', 'Pessoa 4', 'Pessoa 5']

# Processar cada face detectada
for face in result:
    x, y, width, height = face['box']
    face_pixels = image_rgb[y:y+height, x:x+width]
    
    # Extrair o embedding facial e classificar a face
    face_embedding = get_face_embedding(model, face_pixels)
    label = classify_face(face_embedding, known_embeddings, known_labels)
    
    # Desenhar o retângulo e o label na imagem
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Salvar ou exibir a imagem com as faces detectadas e classificadas
cv2.imwrite('/caminho/para/salvar/imagem_resultante.jpg', image)
cv2.imshow('Detecção e Classificação de Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
