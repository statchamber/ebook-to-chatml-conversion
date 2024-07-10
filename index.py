import os
import re
import requests
import json
import sys
import yaml
import zipfile
import time
import difflib
from bs4 import BeautifulSoup
from typing import List, Dict
from flair.data import Sentence
from flair.models import SequenceTagger

from Conversion.prompts import Prompts
from Conversion.text_processing import call_ner, string_similarity
from Conversion.api_calls import kobold_generate_text, gemini_generate_text, get_koboldai_context_limit
from Conversion.file_operations import parse_epub, clear_bin_dir, extract_and_save_text
from Conversion.conversion_logic import start_conversion_of_book

# config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

CONTEXT_PARAGRAPHS = config.get('chunk', {}).get('context', 20)
MAX_PARAGRAPHS_TO_CONVERT = config.get('chunk', {}).get('max_convert', 40)
CONFIDENCE = config.get('entity_detection', {}).get('confidence', 0.4)
ENTITY_DETECTION_MODEL = config.get('entity_detection', {}).get('model', "flair/ner-english-large")
KOBOLDAPI = config.get('api', {}).get('kobold', "https://127.0.0.1:5001/api/")
GEMINI = config.get('api', {}).get('gemini', "")
USE_GEMINI_SUMMARIZATION = config.get('summarization', {}).get('gemini', True)
SUMMARIZE_EVERY = config.get('summarization', {}).get('summarize_every', 10)
DEBUG = config.get('other', {}).get('debug', False)
EBOOKS_DIR = "./ebooks"
BIN_DIR = "./bin"
OUTPUT_DIR = "./output"
STOP_SEQUENCES = ["### Input:", "Previous Summaries:"]
SIMILARITY_THRESHOLD = 0.6  # 60% similarity for name matching

# Convert character names from config to a list
CHARACTER_LIST = []
for char_name, include in config.get('character', {}).items():
    if include:
        words = char_name.replace('_', ' ').split()
        capitalized_name = ' '.join(word.capitalize() for word in words)
        CHARACTER_LIST.append(capitalized_name)

if KOBOLDAPI.endswith('/'):
    KOBOLDAPI = KOBOLDAPI[:-1]

if DEBUG:
    print("Debug mode is enabled.")

# load tagger
print(f"Trying to load entity detection model {ENTITY_DETECTION_MODEL}, if this step fails edit config.yaml")
tagger = SequenceTagger.load(ENTITY_DETECTION_MODEL)

if DEBUG:
    print(f"{ENTITY_DETECTION_MODEL} loaded")

def main():
    for directory in [BIN_DIR, EBOOKS_DIR, OUTPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            if directory == EBOOKS_DIR:
                print("Created ./ebooks, put your books there")
                sys.exit(1)
    clear_bin_dir(BIN_DIR)

    context_limit = get_koboldai_context_limit(KOBOLDAPI)
    if DEBUG:
        print(f"Context limit: {context_limit}")

    extract_and_save_text(EBOOKS_DIR, BIN_DIR)
    
    # Start conversion of books
    for filename in os.listdir(BIN_DIR):
        if filename.endswith('.json') and not filename.endswith('_chunk_info.json') and not filename.endswith('_converted.json'):
            start_conversion_of_book(filename, context_limit, BIN_DIR, OUTPUT_DIR, SUMMARIZE_EVERY, MAX_PARAGRAPHS_TO_CONVERT, CONTEXT_PARAGRAPHS, CHARACTER_LIST, CONFIDENCE, USE_GEMINI_SUMMARIZATION, DEBUG, SIMILARITY_THRESHOLD, tagger, KOBOLDAPI, GEMINI, STOP_SEQUENCES)

    clear_bin_dir(BIN_DIR)

if __name__ == "__main__":
    main()