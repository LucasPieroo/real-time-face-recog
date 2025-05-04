# 📸 Real-Time Face Recognition with Dynamic Retraining

This project is a real-time face recognition system built with a fine-tuned deep learning model (based on VGGFace) and a KNN classifier for flexible and fast retraining. It features a simple user interface built with Streamlit and was designed to allow **adding new people on-the-fly**, even with very few images.

## 🧪 Background & Motivation

Before this project, I worked on a celebrity face recognition model using the [Dataset of other project](https://github.com/LucasPieroo/Face-Recognition). That project focused on classifying known identities with relatively large image sets per person.

From that experience, I became interested in **making a face recognition model adaptable to new people in real time** — with minimal effort from the user — and deployable in an interactive way using **Streamlit**.

## 🛠️ How the System Was Built

At first, I fine-tuned the VGGFace model to improve embedding quality and better distinguish between people.

However, I faced a challenge: when a new class (person) was added, the entire model needed to be retrained. This was time-consuming and inefficient for a real-time, user-facing system.

### 🧠 Solution: Swap the Final Classifier with KNN

To address this, I kept the embedding extractor (the CNN part) and **replaced the final classification layer with a K-Nearest Neighbors (KNN)** classifier. The KNN groups embeddings based on proximity, allowing:

- Fast retraining by simply adding new embeddings
- Easy extension to new classes without touching the core model

### 📷 Synthetic Data with Image Generator

Another challenge was **image scarcity**: in the original dataset, each class had 20–30 images. But asking users to upload that many photos is impractical.

To solve this, I used an **image generator** that creates **synthetic variations** of the user’s face from just 2 input images — applying small transformations (rotation, blur, brightness, etc.). These generated images form a new class and are added to the KNN, making the system flexible and data-efficient.

### ✅ Current Status

With this setup, the app can now:

- Recognize known users in real-time
- Allow adding new users with just 2–3 photos
- Retrain instantly via KNN
- Be used through a simple Streamlit interface

---

## 🔮 Next Steps

One important future improvement is handling **unknown faces** — i.e., detecting when a face does not belong to any of the registered users. This would involve a confidence threshold or outlier detection in the embedding space.

---

## 🤝 Contributions

Ideas, improvements, or pull requests are welcome!

