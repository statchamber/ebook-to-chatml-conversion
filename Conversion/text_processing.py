import re
import difflib
import threading
from typing import List, Dict
from flair.data import Sentence
from flair.models import SequenceTagger
import logging
import yaml
import torch
import gc
import concurrent.futures

# Disable flair logging
logging.getLogger('flair').setLevel(logging.ERROR)

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)
ENTITY_DETECTION_MODEL = config.get('entity_detection', {}).get('model', "flair/ner-english-large")
print(f"Trying to load entity detection model {ENTITY_DETECTION_MODEL}, if this step fails edit config.yaml")

tagger = None
tagger_reset_counter = 0
tagger_lock = threading.Lock()

MAX_WORKERS = config.get('other', {}).get('concurrent_stories', 1)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

def get_tagger():
    global tagger, tagger_reset_counter
    with tagger_lock:
        if tagger is None or tagger_reset_counter >= config['entity_detection'].get('reset_every', 100):
            if tagger is not None:
                del tagger
                torch.cuda.empty_cache()
            tagger = SequenceTagger.load(ENTITY_DETECTION_MODEL)
            if torch.cuda.is_available():
                tagger = tagger.to('cuda')
            tagger_reset_counter = 0
        else:
            tagger_reset_counter += 1
        return tagger

def process_ner(text: str, CONFIDENCE: float) -> List[Dict]:
    current_tagger = get_tagger()
    sentence = Sentence(text)
    
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            current_tagger.predict(sentence)
    else:
        current_tagger.predict(sentence)
    
    entities = [
        {
            'text': entity.text,
            'type': entity.tag,
            'confidence': entity.score
        }
        for entity in sentence.get_spans('ner')
        if entity.tag == "PER" and entity.score > CONFIDENCE
    ]
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return entities

def call_ner(text: str, CONFIDENCE: float) -> List[Dict]:
    return executor.submit(process_ner, text, CONFIDENCE).result()

def string_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# Periodic garbage collection
def periodic_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    threading.Timer(30, periodic_gc).start()

periodic_gc()