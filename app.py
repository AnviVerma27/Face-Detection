import streamlit as st
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def main():
    st.title("Webcam Streamlit App")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to access the webcam.")
        return
    frame_width = 640
    frame_height = 480
    cap.set(3, frame_width)
    cap.set(4, frame_height)
    placeholder = st.empty()
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if not ret:
            st.error("Error: Unable to capture frame.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(rgb_frame, channels='RGB')
    cap.release()

if __name__ == "__main__":
    main()
