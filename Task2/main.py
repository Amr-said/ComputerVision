import streamlit as st
# import streamlit_tags as stt
import cv2
import numpy as np
import Hough
import Contour
# from skimage import io
# from skimage.io import imread
# import io
from io import BytesIO
from PIL import Image
from io import StringIO
import base64
from inspect import getclosurevars
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy







def main():
    
    # Include the CSS file
    st.set_page_config(page_title="Task2", layout="wide")
     
    # st.sidebar.title("Task 2")
    st.markdown("<h1 style='text-align: center; color: red;'>Task 2</h1>", unsafe_allow_html=True)




    with st.sidebar:
        kernel_size = st.sidebar.slider("Kernel size", min_value=3, max_value=15, step=2, value=7)
        sigma = st.sidebar.slider("Sigma", min_value=0.0, max_value=15.0, step=0.1, value=5.0)



    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Canny edge detector", "Hough Lines", "Hough Circles", "Hough elipse", "Active Contour"])

    with tab1:
        # Upload image
        uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="file1")
        if uploaded_file is not None:
            # uploaded_file = Image.open(uploaded_file)
            file = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            file = cv2.imdecode(file, cv2.IMREAD_GRAYSCALE)






            # Create two columns
            col1, col2 = st.columns(2)

            # Display the images in the columns
            with col1:
                
                # Show original image
                st.image(file, caption="GrayScale Image", use_column_width=True)

            with col2:
                # Apply filters
            
                cny_img = Hough.canny_apply(file,kernel_size,sigma)
                img_array = np.asarray(cny_img)
                image_bytes = cv2.imencode('.png', img_array)[1].tobytes()
                # Show filtered images
                
                st.image(
                    io.BytesIO(image_bytes),
                    caption="Canny Filtered Image",
                    use_column_width=True
                )






            

            

    with tab2:
         # Upload image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],key="file2")
        if uploaded_file is not None:
            # uploaded_file = Image.open(uploaded_file)
            file = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            file = cv2.imdecode(file, cv2.IMREAD_GRAYSCALE)



            # Create two columns
            col1, col2 = st.columns(2)

            # Display the images in the columns
            with col1:
                # Show original image
                # st.image(file, caption="GrayScale Image", use_column_width=True)
                # Show line image
                st.image(uploaded_file, caption="Line Image",use_column_width=True)

            with col2:
                file_content = uploaded_file.read()
                encoded_string = base64.b64encode(file_content).decode('utf-8')
                lin_img = Hough.hough_lines(encoded_string,kernel_size,sigma)



        
            
            

    with tab3:
         # Upload image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file3")
        if uploaded_file is not None:
            # uploaded_file = Image.open(uploaded_file)
            file = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            file = cv2.imdecode(file, cv2.IMREAD_GRAYSCALE)

            # Create two columns
            col1, col2 = st.columns(2)

            # Display the images in the columns
            with col1:
                # Show original image
                st.image(file, caption="GrayScale Image", use_column_width=True)

            with col2:
                file_content = uploaded_file.read()
                encoded_string = base64.b64encode(file_content).decode('utf-8')
                cir_img =Hough.hough_circles(file,kernel_size,sigma)


        
            






            


    with tab4:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
            edges = Hough.preprocess_image(image)


            st.sidebar.markdown("<hr>", unsafe_allow_html=True)



            threshold = st.sidebar.slider("Threshold", 1, 100, 50)
            min_minor_axis = st.sidebar.slider("Min Minor Axis", 1, 100, 10)
            max_minor_axis = st.sidebar.slider("Max Minor Axis", 10, 200, 100)




             # Create Three columns
            col1, col2, col3 = st.columns(3)

            # Display the images in the columns
            with col1:
                # Show original image
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="BGR", caption="GrayScale Image")

                # st.image(file, caption="GrayScale Image", use_column_width=True)

            with col2:
                st.image(edges, caption="Edge Image")

            with col3:
                candidates = Hough.hough_ellipse(edges, threshold, min_minor_axis, max_minor_axis)
                result = Hough.draw_ellipses(image, candidates, min_minor_axis, max_minor_axis)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="BGR", caption= "Detected Ellipses")



















    with tab5:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # k = 40
            # neighbors = np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])

            # Read image    
            with open("uploaded_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
                

            # img = io.imread(uploaded_file, as_gray=True)
            img = imread(uploaded_file, as_gray=True)    
            x_center = st.sidebar.number_input('X Center', min_value=0, max_value=img.shape[1] - 1, value=680)
            y_center = st.sidebar.number_input('Y Center', min_value=0, max_value=img.shape[0] - 1, value=800)
            radius = st.sidebar.number_input('Radius', min_value=1, max_value=max(img.shape), value=550)
            num_points = st.sidebar.number_input('Number of Points', min_value=4, max_value=100, value=20)
            alpha = st.sidebar.slider('Alpha (Continuity)', min_value=0, max_value=400, value=200)
            beta = st.sidebar.slider('Beta (Curvature)', min_value=0, max_value=10, value=2)
            gamma = st.sidebar.slider('Gamma', min_value=1, max_value=1000, value=200)    



            # Create two columns
            col1, col2 = st.columns(2)

            # Display the images in the columns
            with col1:
                
                st.image(uploaded_file, caption='Original Image', use_column_width=True)
                img_out="uploaded_image.jpg"
                img_out, snake = Contour.activeContour(img_out, (x_center,y_center), radius,alpha, beta ,gamma,num_points)
                x = [point[0] for point in snake]
                y = [point[1] for point in snake]
                fig, ax = plt.subplots()
                plt.plot(x, y, 'g-')
                plt.axis('off')
                img_out=ax.imshow(img_out, cmap='gray')




            with col2:
                st.pyplot(fig, caption= "Active Contour")





if __name__ == "__main__":
    main()