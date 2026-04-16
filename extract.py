import fitz  # PyMuPDF

doc = fitz.open('Computational Mirror Cup and Saucer Art.pdf')
text = ""
for page in doc:
    text += page.get_text()

with open('paper_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)
