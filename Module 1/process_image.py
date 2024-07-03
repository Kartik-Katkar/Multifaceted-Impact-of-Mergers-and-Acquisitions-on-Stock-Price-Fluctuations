import easyocr

def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

def append_text_to_file(text, file_path):
    with open(file_path, 'a') as file:
        file.write(text + '\n')

# def process_images_in_directory(directory_path, output_file):
#     for filename in os.listdir(directory_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#             image_path = os.path.join(directory_path, filename)
#             text = extract_text_from_image(image_path)
#             append_text_to_file(text, output_file)
#             print(f"Processed {filename} and appended text to {output_file}")

def process_single_image(image_path, output_file):
    text = extract_text_from_image(image_path)
    append_text_to_file(text, output_file)
    print(f"Processed {image_path} and appended text to {output_file}")