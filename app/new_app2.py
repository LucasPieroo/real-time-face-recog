import os
import time
import io
import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# -----------------------------
# Configurações
# -----------------------------
MODEL_PATH   = 'models/enhanced_vggface.h5'
KNN_PATH     = 'models/knn_model.pkl'
EMB_PATH     = 'models/embeddings.pkl'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Carrega modelos
full_model = load_model(MODEL_PATH)
knn_model, _ = joblib.load(KNN_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# -----------------------------
# Funções auxiliares
# -----------------------------
def augment_images(image_paths, augment_count=20):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    augmented_images = []
    for path in image_paths:
        img = load_img(path, target_size=(224, 224))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            augmented_images.append(batch[0])
            i += 1
            if i >= augment_count:
                break
    return augmented_images


def add_new_person(image_paths, label_name, model, emb_path=EMB_PATH, knn_path=KNN_PATH, augment_count=20):
    if os.path.exists(emb_path):
        embeddings, labels = joblib.load(emb_path)
    else:
        embeddings, labels = np.array([]).reshape(0, 2048), np.array([])

    augmented_images = augment_images(image_paths, augment_count=augment_count)
    new_embeddings = [model.predict(np.expand_dims(img / 255.0, axis=0)).flatten() for img in augmented_images]
    new_labels = [label_name] * len(new_embeddings)

    embeddings = np.vstack([embeddings, new_embeddings])
    labels = np.hstack([labels, new_labels])

    joblib.dump((embeddings, labels), emb_path)
    retrain_knn(embeddings, labels, knn_path)


def retrain_knn(embeddings, labels, knn_path):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(embeddings, labels)
    joblib.dump((knn, None), knn_path)

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("📸 Real-Time Face Recognition & Registration")

# Menu lateral: captura ou cadastro
menu = st.sidebar.selectbox("Escolha a opção", ("🎥 Captura em Tempo Real", "➕ Adicionar Nova Pessoa"))

# Captura em tempo real
if menu == "🎥 Captura em Tempo Real":
    def toggle_capture():
        st.session_state.run = not st.session_state.get('run', False)

    if 'run' not in st.session_state:
        st.session_state.run = False

    button_label = "Parar Captura" if st.session_state.run else "Iniciar Captura"
    st.button(button_label, on_click=toggle_capture)

    frame_slot = st.empty()
    cap = cv2.VideoCapture(0)

    if st.session_state.run:
        while cap.isOpened() and st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Não foi possível capturar a webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                rgb_img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_img, (224, 224)).astype('float32') / 255.0
                inp = np.expand_dims(resized, axis=0)
                probs = full_model.predict(inp)
                label = knn_model.predict(probs)[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        cv2.destroyAllWindows()

# Cadastro de nova pessoa
elif menu == "➕ Adicionar Nova Pessoa":
    st.subheader("➕ Cadastro de Nova Pessoa")
    label_name = st.text_input("Nome da nova pessoa:")
    uploaded_files = st.file_uploader("Envie ao menos 2 imagens", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if st.button("Adicionar ao Modelo"):
        if label_name and len(uploaded_files) >= 2:
            temp_paths = []
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                img = Image.open(io.BytesIO(bytes_data)).convert('RGB')
                path = f"temp_{uploaded_file.name}"
                img.save(path)
                temp_paths.append(path)

            add_new_person(temp_paths, label_name, full_model)
            st.success(f"{label_name} adicionado com sucesso!")

            # recarrega KNN
            knn_model, _ = joblib.load(KNN_PATH)

            # apaga temporários
            for path in temp_paths:
                os.remove(path)
        else:
            st.warning("Informe um nome e envie ao menos 2 imagens!")
