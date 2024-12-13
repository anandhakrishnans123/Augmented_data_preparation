import streamlit as st
import google.generativeai as genai
import pandas as pd
from io import StringIO
from PIL import Image
import fitz  # PyMuPDF to handle PDF
import io

st.title("Image and PDF to CSV Converter")

# Input for API key
api_key = "AIzaSyBA3sUF2AFbcYwrsuY7zVu38dB-pOA-v9c"

if api_key:
    # Configure the Gemini Pro API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Upload an image or PDF
    uploaded_file = st.file_uploader("Upload an image or PDF", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        # If the uploaded file is an image
        if file_type in ["image/png", "image/jpeg", "image/jpg"]:
            img = Image.open(uploaded_file)
            
            # Initialize session state to track image rotation
            if 'rotation_angle' not in st.session_state:
                st.session_state.rotation_angle = 0
            
            # Button to rotate the image by 90 degrees
            if st.button("Rotate Image 90°"):
                st.session_state.rotation_angle += 90
                st.session_state.rotation_angle %= 360  # Ensure angle stays within 0-359 degrees
            
            # Rotate the image and expand it to ensure full dimensions are kept
            rotated_img = img.rotate(st.session_state.rotation_angle, expand=True)
            st.image(rotated_img, caption='Uploaded Image (Rotated)', use_column_width=True)

            # Generate CSV from image
            if st.button("Convert Image to CSV"):
                try:
                    # Create a prompt for the model
                    prompt = "Extract data from the uploaded image, extract the data including handwritten text, numbers and convert it to a CSV format. If possible, identify the type of data (e.g., names, dates, numbers) and structure the CSV accordingly. Also only give out the output table no other specific information is required"
                    response = model.generate_content([prompt, rotated_img])  # Assuming this format works with your API
                    csv_result = response.text

                    # Use StringIO to simulate a file-like object for pandas
                    data_io = StringIO(csv_result)

                    # Read the data into a DataFrame
                    df = pd.read_csv(data_io)

                    # Save the DataFrame to a CSV file
                    csv_file_path = 'csv_output.csv'
                    df.to_csv(csv_file_path, index=False)

                    st.success(f"CSV file saved as {csv_file_path}")
                    st.write(df)  # Display the DataFrame

                    # Provide a download link
                    with open(csv_file_path, "rb") as f:
                        st.download_button("Download CSV", f, file_name=csv_file_path)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # If the uploaded file is a PDF
        elif file_type == "application/pdf":
            # Read PDF file using PyMuPDF
            pdf_document = fitz.open(uploaded_file)
            pdf_text = ""

            # Extract text from all pages of the PDF
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                pdf_text += page.get_text()

            # Display the extracted text
            st.text_area("Extracted Text", pdf_text, height=300)

            # Generate CSV from PDF text
            if st.button("Convert PDF to CSV"):
                try:
                    # Create a prompt for the model
                    prompt = f"Extract data from the uploaded PDF text: {pdf_text}. Identify tables or structured data and convert it into CSV format. Do not include any other specific information."
                    response = model.generate_content([prompt])  # Assuming this format works with your API
                    csv_result = response.text

                    # Use StringIO to simulate a file-like object for pandas
                    data_io = StringIO(csv_result)

                    # Read the data into a DataFrame
                    df = pd.read_csv(data_io)

                    # Save the DataFrame to a CSV file
                    csv_file_path = 'csv_output.csv'
                    df.to_csv(csv_file_path, index=False)

                    st.success(f"CSV file saved as {csv_file_path}")
                    st.write(df)  # Display the DataFrame

                    # Provide a download link
                    with open(csv_file_path, "rb") as f:
                        st.download_button("Download CSV", f, file_name=csv_file_path)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter your API key to proceed.")
