import os
import json
import zipfile
from bs4 import BeautifulSoup

def parse_epub(epub_path):
    with zipfile.ZipFile(epub_path, 'r') as z:
        for filename in z.namelist():
            if filename.endswith(('.html', '.xhtml')):
                with z.open(filename, 'r') as f:
                    content = f.read().decode('utf-8')
                    soup = BeautifulSoup(content, 'xml')
                    for para in soup.find_all('p'):
                        # Skip metadata paragraphs
                        if para.find_parent(class_='calibre3') or para.find_parent(class_='calibre14'):
                            continue
                        yield para.get_text(strip=True)

def extract_and_save_text(EBOOKS_DIR, BIN_DIR):
        # Extract text from txt and epub files and save to JSON
        for filename in os.listdir(EBOOKS_DIR):
            if filename.endswith(('.txt', '.epub')):
                file_path = os.path.join(EBOOKS_DIR, filename)
                out_file = os.path.join(BIN_DIR, f"{os.path.splitext(filename)[0]}.json")
                if filename.endswith('.txt'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            lines = file.readlines()
                    except UnicodeDecodeError:
                        # If UTF-8 fails, try with 'latin-1' encoding
                        with open(file_path, 'r', encoding='latin-1') as file:
                            lines = file.readlines()
                    paragraphs = []
                    for line in lines:
                        stripped_line = line.strip()
                        if not stripped_line.startswith('>') and '---' not in stripped_line and '***' not in stripped_line and '* * *' not in stripped_line and '@gmail.com' not in stripped_line and stripped_line != '':
                            if len(stripped_line) < 4 and paragraphs:
                                paragraphs[-1] += ' ' + stripped_line
                            else:
                                paragraphs.append(stripped_line)
                else:  # .epub file
                    paragraphs = list(parse_epub(file_path))
                # Apply the same protections to both txt and epub content
                filtered_paragraphs = []
                for para in paragraphs:
                    stripped_para = para.strip()
                    if not stripped_para.startswith('>') and '---' not in stripped_para and '***' not in stripped_para and '* * *' not in stripped_para and '@gmail.com' not in stripped_para and stripped_para != '':
                        if len(stripped_para) < 4 and filtered_paragraphs:
                            filtered_paragraphs[-1] += ' ' + stripped_para
                        else:
                            filtered_paragraphs.append(stripped_para)
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_paragraphs, f, ensure_ascii=False, indent=4)

def clear_bin_dir(BIN_DIR):
    for filename in os.listdir(BIN_DIR):
        file_path = os.path.join(BIN_DIR, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing {filename}: {e}")