import re
import difflib
from typing import List, Dict
from flair.data import Sentence
from flair.models import SequenceTagger

def call_ner(text: str, tagger: SequenceTagger, CONFIDENCE: float) -> List[Dict]:
    # create a sentence
    sentence = Sentence(text)
    
    # predict NER tags
    tagger.predict(sentence)
    
    # extract entities
    entities = []
    for entity in sentence.get_spans('ner'):
        entities.append({
            'text': entity.text,
            'type': entity.tag,
            'confidence': entity.score
        })
    
    # filter high confidence persons
    high_confidence_persons = list({entity['text']: entity for entity in entities
        if entity["type"] == "PER" and entity["confidence"] > CONFIDENCE}.values())
    
    return high_confidence_persons

def string_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()