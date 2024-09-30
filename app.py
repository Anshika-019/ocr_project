import gradio as gr
from ocr_module import OCRProcessor
import tempfile
from PIL import Image

def ocr_and_search(image, keyword):
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        temp_file_path = temp_file.name

    # Process the image using the file path
    extracted_text = ocr_processor.process_image(temp_file_path)
    
    if not extracted_text:
        return "Failed to extract text.", ""
    
    # Highlight the keyword in the extracted text
    highlighted_text = extracted_text.replace(keyword, f"**{keyword}**")
    
    return extracted_text, highlighted_text

# Create Gradio interface
iface = gr.Interface(
    fn=ocr_and_search,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Keyword")],
    outputs=[gr.Textbox(label="Extracted Text"), gr.Textbox(label="Highlighted Text")],
    title="OCR and Keyword Search",
    description="Upload an image and search for keywords in the extracted text."
)

if __name__ == "__main__":
    iface.launch()
