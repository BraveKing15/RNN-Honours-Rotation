
import pypdf
import os

pdf_path = "A_Convex_Parameterization_of_Robust_Recurrent_Neural_Networks.pdf"

try:
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    print("PDF content extracted to pdf_content.txt")

except Exception as e:
    print(f"Error extracting PDF: {e}")
