import os
import sys
import yaml

import concurrent.futures
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Conversion.api_calls import get_koboldai_context_limit
from Conversion.file_operations import clear_bin_dir, extract_and_save_text
from Conversion.conversion_logic import start_conversion_of_book

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

CONTEXT_PARAGRAPHS = config.get('chunk', {}).get('context', 20)
MAX_PARAGRAPHS_TO_CONVERT = config.get('chunk', {}).get('max_convert', 40)
CONFIDENCE = config.get('entity_detection', {}).get('confidence', 0.4)
KOBOLDAPI = config.get('api', {}).get('kobold', {}).get('url', "http://localhost:5001/api/")
GEMINI_API_KEY = config.get('api', {}).get('gemini', {}).get('api_key', "")
USE_GEMINI_SUMMARIZATION = config.get('summarization', {}).get('api', {}).get('gemini', {}).get('enabled', False)
SUMMARIZE_EVERY = config.get('summarization', {}).get('summarize_every', 10)
DEBUG = config.get('other', {}).get('debug', False)
OPENAI_API_KEY = config.get('api', {}).get('openai', {}).get('api_key', "")
OPENAI_API_BASE = config.get('api', {}).get('openai', {}).get('api_base', "")
OPENAI_MODEL = config.get('api', {}).get('openai', {}).get('model', "")
EBOOKS_DIR = "./ebooks"
BIN_DIR = "./bin"
OUTPUT_DIR = "./output"
STOP_SEQUENCES = ["### Input:", "Previous Summaries:"]
SIMILARITY_THRESHOLD = config.get('other', {}).get('string_similarity', 0.6)

# Convert character names from config to a list
CHARACTER_LIST = []
for char_name, include in config.get('character', {}).items():
    if include:
        words = char_name.replace('_', ' ').split()
        capitalized_name = ' '.join(word.capitalize() for word in words)
        CHARACTER_LIST.append(capitalized_name)

if KOBOLDAPI.endswith('/'):
    KOBOLDAPI = KOBOLDAPI[:-1]

if OPENAI_API_BASE and not OPENAI_API_BASE.endswith('/'):
    OPENAI_API_BASE = OPENAI_API_BASE + '/'

if DEBUG:
    print("Debug mode is enabled.")

def main():
    for directory in [BIN_DIR, EBOOKS_DIR, OUTPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            if directory == EBOOKS_DIR:
                print("Created ./ebooks, put your books there")
                sys.exit(1)
    clear_bin_dir(BIN_DIR)

    if config['api']['kobold']['enabled']:
        context_limit = get_koboldai_context_limit(KOBOLDAPI)
    else:
        context_limit = 0
    if DEBUG:
        print(f"Context limit: {context_limit}")

    extract_and_save_text(EBOOKS_DIR, BIN_DIR)
    
    # Get the list of JSON files to process
    json_files = [f for f in os.listdir(BIN_DIR) if f.endswith('.json') and not f.endswith('_chunk_info.json') and not f.endswith('_converted.json')]
    
    # Create a thread pool with the specified number of concurrent stories
    with concurrent.futures.ThreadPoolExecutor(max_workers=config['other']['concurrent_stories']) as executor:
        # Submit each file for processing
        futures = [executor.submit(start_conversion_of_book, filename, context_limit, BIN_DIR, OUTPUT_DIR, SUMMARIZE_EVERY, MAX_PARAGRAPHS_TO_CONVERT, CONTEXT_PARAGRAPHS, CHARACTER_LIST, CONFIDENCE, USE_GEMINI_SUMMARIZATION, DEBUG, SIMILARITY_THRESHOLD, KOBOLDAPI, OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL, GEMINI_API_KEY, STOP_SEQUENCES, config) for filename in json_files]
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    # Clear the progress display
    print("\n" * len(json_files))

    clear_bin_dir(BIN_DIR)

if __name__ == "__main__":
    main()