import streamlit as st
import cv2
import tempfile

def main():

    st.sidebar.title("Navbar")
    page_options = ["Upload Video", "Object Data", "Violent Data"]

    selected_page = st.sidebar.selectbox("Page", page_options)

    if selected_page == "Upload Video":
        st.header("Upload Video")
        st.write("Nội dung của Upload Video.")

        st.title("Tải lên và Hiển thị Video")

        uploaded_file = st.file_uploader("Chọn một file video", type=["mp4", "avi"])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            cap = cv2.VideoCapture(temp_file_path)

            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                stframe.image(frame, channels="BGR")

            cap.release()

    elif selected_page == "Object Data":
        st.header("Object Data")
        st.write("Nội dung của trang Object Data.")

    elif selected_page == "Violent Data":
        st.header("Violent Data")
        st.write("Nội dung của trang Violent Data.")

if __name__ == "__main__":
    main()
