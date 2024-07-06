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
from prompts import Prompts
from typing import List, Dict
from flair.data import Sentence
from flair.models import SequenceTagger


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

PARAGRAPH_CHUNK_SIZE = config.get('chunk', {}).get('size', 20)
CONTEXT_PARAGRAPHS = config.get('chunk', {}).get('context', 20)
MAX_PARAGRAPHS_TO_CONVERT = config.get('chunk', {}).get('max_convert', 40)
ADD_NARRATOR_TO_CHARACTERLIST = config.get('character', {}).get('narrator', True)
ADD_UNKNOWN_TO_CHARACTERLIST = config.get('character', {}).get('unknown', True)
CONFIDENCE = config.get('entity_detection', {}).get('confidence', 0.4)
ENTITY_DETECTION_MODEL = config.get('entity_detection', {}).get('model', "flair/ner-english-large")
KOBOLDAPI = config.get('api', {}).get('kobold', "https://127.0.0.1:5001/api/")
GEMINI = config.get('api', {}).get('gemini', "")
USE_GEMINI_SUMMARIZATION = config.get('summarization', {}).get('gemini', True)
DEBUG = config.get('other', {}).get('debug', False)
EBOOKS_DIR = "./ebooks"
BIN_DIR = "./bin"
OUTPUT_DIR = "./output"
STOP_SEQUENCES = ["### Input:", "Previous Summaries:", "}\n "]
SIMILARITY_THRESHOLD = 0.6  # 60% similarity for name matching

if KOBOLDAPI.endswith('/'):
    KOBOLDAPI = KOBOLDAPI[:-1]

if DEBUG:
    print("Debug mode is enabled.")
    print(f"Trying to load entity detection model {ENTITY_DETECTION_MODEL}, if this step fails edit config.yaml")

# load tagger
tagger = SequenceTagger.load(ENTITY_DETECTION_MODEL)

if DEBUG:
    print(f"{ENTITY_DETECTION_MODEL} loaded")

def call_ner(text: str) -> List[Dict]:
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

# Function to generate text using KoboldAI API
def kobold_generate_text(prompt, temperature, grammar, max_length, max_token_count, cleanse):
    attempts = 0
    maxattempts = 10
    while attempts < maxattempts:
        try:
            response = requests.post(f'{KOBOLDAPI}/v1/generate', 
                                     headers={'accept': 'application/json', 'Content-Type': 'application/json'}, 
                                     json={
                                         "max_context_length": max_token_count,
                                         "max_length": max_length,
                                         "prompt": prompt,
                                         "quiet": False,
                                         "temperature": temperature,
                                         "grammar": grammar,
                                         "stop_sequence": STOP_SEQUENCES
                                     })
            
            if response.status_code == 503:
                print("KoboldAI server is busy")
                time.sleep(5)
                attempts += 1
                continue
            
            response.raise_for_status()

            # Cleanse stop sequences
            text = response.json()['results'][0]['text']
            if cleanse:
                for stop_sequence in STOP_SEQUENCES:
                    text = text.replace(stop_sequence, "")
            cleaned_text = text
            if attempts > 1:
                print(f"KoboldAI API request successful after {attempts} attempts")
            return cleaned_text.strip()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempts + 1}: KoboldAI API not available on {KOBOLDAPI}, is the API running?")
            attempts += 1
            if attempts == maxattempts:
                sys.exit(1)
            time.sleep(1)  # Wait a second before reattempting

def gemini_generate_text(prompt, conversation_history=None):
    if GEMINI == "":
        raise ValueError("Gemini API is not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI}"
    headers = {'Content-Type': 'application/json'}

    contents = []
    if conversation_history:
        contents.extend(conversation_history)
    
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    data = {
        "contents": contents,
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            }
        ],
        "generationConfig": {
            "stopSequences": STOP_SEQUENCES,
            "temperature": 1.0,
            "maxOutputTokens": 800,
            "topP": 0.8,
            "topK": 10
        }
    }

    for attempt in range(10):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                if attempt < 9:  # If it's not the last attempt
                    time.sleep(1)  # Wait for 1 second before retrying
                    continue
                else:
                    return "No response generated after 10 attempts."
        except requests.exceptions.RequestException:
            if attempt < 9:  # If it's not the last attempt
                time.sleep(1)  # Wait for 1 second before retrying
            else:
                raise  # Re-raise the exception if all attempts failed

def get_koboldai_context_limit():
    attempts = 0
    while attempts < 3:
        try:
            max_token_count_response = requests.get(f'{KOBOLDAPI}/api/extra/true_max_context_length', headers={'accept': 'application/json'})
            max_token_count = max_token_count_response.json()['value']
            return max_token_count
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempts + 1}: KoboldAI API not available on {KOBOLDAPI}, is the API running?")
            attempts += 1
            if attempts == 3:
                sys.exit(1)
    return None  # This line will only be reached if all attempts fail


def string_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# Function to parse epub files
def parse_epub(epub_path):
    """Parses an epub file and yields each paragraph of text."""
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

def start_conversion_of_book(filename, context_limit):
    filename = os.path.splitext(filename)[0]
    print(f"Starting conversion of book: {filename}")
    with open(os.path.join(BIN_DIR, f"{filename}.json"), 'r', encoding='utf-8') as f:
        paragraphs = json.load(f)
    
    previous_summary = ""
    converted_prompts = []
    high_confidence_persons = []
    masked_names = {}
    speakers_list = []

    if ADD_NARRATOR_TO_CHARACTERLIST:
        speakers_list.append("Narrator")
        masked_names["Narrator"] = "Narrator"
    if ADD_UNKNOWN_TO_CHARACTERLIST:
        speakers_list.append("Unknown")
        masked_names["Unknown"] = "Unknown"
    
    total_chunks = 0
    for i in range(0, MAX_PARAGRAPHS_TO_CONVERT, PARAGRAPH_CHUNK_SIZE):
        total_chunks += 1
        print(f"Converting chunk {total_chunks} out of {MAX_PARAGRAPHS_TO_CONVERT // PARAGRAPH_CHUNK_SIZE}")
        
        prompt = "\n".join(paragraphs[i:i+PARAGRAPH_CHUNK_SIZE])
        
        # Update high_confidence_persons and masked_names for each chunk
        new_persons = call_ner(prompt)
        for person in new_persons:
            if DEBUG:
                print(person)
                print("-"*100)
            if person['text'] not in [p['text'] for p in high_confidence_persons]:
                high_confidence_persons.append(person)
                masked_names[person['text']] = f"Character_{len(masked_names) + 1}"

        # Check for possible aliases and update masked_names
        for i, person1 in enumerate(high_confidence_persons):
            for person2 in high_confidence_persons[i+1:]:
                if person1['text'].lower() in person2['text'].lower() or person2['text'].lower() in person1['text'].lower():
                    masked_names[person2['text']] = masked_names[person1['text']]

        # Replace all persons and aliases in the prompt with masked names
        for original_name, masked_name in sorted(masked_names.items(), key=lambda x: len(x[0]), reverse=True):
            prompt = re.sub(r'\b' + re.escape(original_name) + r'\b', masked_name, prompt, flags=re.IGNORECASE)
        
        # Apply the same masking to the total_lines
        total_lines = paragraphs[i:i+PARAGRAPH_CHUNK_SIZE+CONTEXT_PARAGRAPHS]
        masked_total_lines = []
        for line in total_lines:
            masked_line = line
            for original_name, masked_name in sorted(masked_names.items(), key=lambda x: len(x[0]), reverse=True):
                masked_line = re.sub(r'\b' + re.escape(original_name) + r'\b', masked_name, masked_line, flags=re.IGNORECASE)
            masked_total_lines.append(masked_line)
        
        formatted_speakers = set([f'"\\"{masked_name}\\""' for masked_name in masked_names.values()])
        formatted_speakers = ' | '.join(formatted_speakers) + " | string"
        
        summaryprompt = Prompts.SummarizationPrompt.replace("{speakers}", ', '.join(set(masked_names.values()))).replace("{prompt}", prompt).replace("{previous_summary}", f"Previous Summary:\n{previous_summary}" if previous_summary else "")
        if USE_GEMINI_SUMMARIZATION:
            summary = gemini_generate_text(summaryprompt)
            if summary == "No response generated after 10 attempts.":
                summary = kobold_generate_text(summaryprompt, 0.5, "", 500, context_limit, True)
        else:
            summary = kobold_generate_text(summaryprompt, 0.5, "", 500, context_limit, True)

        if DEBUG:
            print("Summary prompt:")
            print(summaryprompt)
            print("-"*100)
            print("Summary:")
            print(summary)
            print("-"*100)
        previous_summary = summary

        prompt_lines = prompt.split('\n')
        converted_prompt = []

        # Prepend the previous chunk's converted prompt for context
        if converted_prompts:
            inter_chunk_context = "\n".join(converted_prompts[-CONTEXT_PARAGRAPHS:]) # Use last CONTEXT_PARAGRAPHS lines for context
        else:
            inter_chunk_context = ""

        for j in range(0, len(prompt_lines), 5):
            # Prepend the previous chunk's converted prompt for context
            if converted_prompt:
                intra_chunk_context = "\n".join(converted_prompt[-CONTEXT_PARAGRAPHS:])  # Use last CONTEXT_PARAGRAPHS lines for context
            else:
                intra_chunk_context = ""
            
            # If intra_chunk_context is not full, supplement with inter_chunk_context
            if len(intra_chunk_context.split('\n')) < CONTEXT_PARAGRAPHS:
                remaining_lines = CONTEXT_PARAGRAPHS - len(intra_chunk_context.split('\n'))
                intra_chunk_context = inter_chunk_context.split('\n')[-remaining_lines:] + intra_chunk_context.split('\n')
                intra_chunk_context = "\n".join(intra_chunk_context)

            if DEBUG and intra_chunk_context == "" and inter_chunk_context != "":
                print("!" * 200)
            
            chunk = prompt_lines[j:j+5]
            context_lines = masked_total_lines[j+5:j+5+CONTEXT_PARAGRAPHS]  # Get the next CONTEXT_PARAGRAPHS lines for context from masked_total_lines
            excerpt = intra_chunk_context + "\n" + ("\n".join(chunk + context_lines))  # Combine chunk and context lines
            extracted_lines = "\n".join([f"Line{k+1}: {line}" for k, line in enumerate(chunk)])
            conversionprompt = Prompts.ConversionPrompt.replace("{speakers}", ', '.join(set(masked_names.values()))).replace("{summary}", summary).replace("{excerpt}", excerpt).replace("{extracted_lines}", extracted_lines)
            conversion = kobold_generate_text(conversionprompt, 0.5, Prompts.ConversionGrammar.replace("{speakers}", formatted_speakers), 500, context_limit, True)
            if DEBUG:
                print("Conversion prompt:")
                print(conversionprompt)
                print("-"*100)
                print("Conversion:")
                print(conversion)
                print("-"*100)
            try:
                conversion_json = json.loads(conversion)
                if DEBUG:
                    print(conversion_json)
                    print("-"*100)
                for k, line in enumerate(chunk):
                    line_key = f"Line{k+1}"
                    if line_key in conversion_json:
                        speaker = conversion_json[line_key].get("speaker", "narrator")
                        
                        # Remove "The " or "A " from the start of the speaker name
                        if speaker.lower().startswith("the "):
                            speaker = speaker[4:]
                        elif speaker.lower().startswith("a "):
                            speaker = speaker[2:]
                        
                        # Capitalize the first letter of the speaker name
                        speaker = speaker.capitalize()

                        # Remove everything in parentheses from the speaker name
                        speaker = re.sub(r'\([^)]*\)', '', speaker).strip()
                        
                        # Check if the speaker is similar to "Not specified" or "N/A"
                        if string_similarity(speaker.lower(), "not specified") >= 0.8 or string_similarity(speaker.lower(), "n/a") >= 0.8:
                            if ADD_NARRATOR_TO_CHARACTERLIST:
                                speaker = "Narrator"
                            elif ADD_UNKNOWN_TO_CHARACTERLIST:
                                speaker = "Unknown"
                        
                        # Check if the speaker does not include "Character_"
                        if "character_" not in speaker.lower():
                            # Check if the speaker is similar to any existing speaker
                            similar_speaker = next((s for s in speakers_list if string_similarity(s.lower(), speaker.lower()) >= SIMILARITY_THRESHOLD), None)
                            if similar_speaker:
                                if DEBUG and string_similarity(speaker.lower(), similar_speaker.lower()) < 1:
                                    print(f"Speaker {speaker} is similar to {similar_speaker}")
                                    print(f"Similarity: {string_similarity(speaker.lower(), similar_speaker.lower())}")
                                    print("-"*100)
                                speaker = similar_speaker
                            elif speaker not in speakers_list:
                                if DEBUG:
                                    print(f"Speaker {speaker} is not in speakers_list")
                                    print("-"*100)
                                speakers_list.append(speaker)
                        elif "character_" in speaker.lower(): # speaker includes "character_", we gotta make sure its exactly "Character_X" and nothing else for example "Dr. Character_X"
                            # Extract the number after "character_"
                            character_number = re.search(r'character_(\d+)', speaker.lower())
                            if character_number:
                                speaker = f"Character_{character_number.group(1)}"
                            else:
                                # If no number is found, default to "Unknown"
                                if DEBUG:
                                    print(f"Speaker {speaker} is glitched and has no number, defaulting to Unknown")
                                    print("-"*100)
                                speaker = "Unknown"
                            if speaker not in speakers_list:
                                speakers_list.append(speaker)
                        
                        converted_prompt.append(f"{speaker}: {line}")
                    else:
                        converted_prompt.append(line)
            except json.JSONDecodeError:
                if DEBUG:
                    print(f"Error parsing JSON for chunk {j//5 + 1}. Skipping this chunk.")
                converted_prompt.extend(chunk)

        converted_prompts.extend(converted_prompt)
        if DEBUG:
            print("Converted prompt:")
            print("\n".join(converted_prompt))
            print("-"*100)
    
    print(f"Converted {total_chunks} chunks out of {MAX_PARAGRAPHS_TO_CONVERT // PARAGRAPH_CHUNK_SIZE} total chunks")

    # final results
    output_file_path = os.path.join(OUTPUT_DIR, f"{filename}_converted.txt")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in converted_prompts:
            # Create a dictionary to store the shortest original name for each masked name
            shortest_names = {}
            for original_name, masked_name in masked_names.items():
                if masked_name not in shortest_names or len(original_name) < len(shortest_names[masked_name]):
                    shortest_names[masked_name] = original_name
            
            # Replace masked names with their shortest original names (could be bad, idk)
            # Sort masked names by their numeric value in descending order
            sorted_masked_names = sorted(shortest_names.keys(), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0, reverse=True)
            
            for masked_name in sorted_masked_names:
                original_name = shortest_names[masked_name]
                line = line.replace(masked_name, original_name)
            output_file.write(line + "\n")
    
    # Save the converted prompts as ChatML format
    chatml_output_path = os.path.join(OUTPUT_DIR, f"{filename}_chatml.txt")
    with open(chatml_output_path, 'w', encoding='utf-8') as chatml_file:
        current_speaker = None
        current_message = []
        
        for line in converted_prompts:
            if ':' in line:
                speaker, message = line.split(':', 1)
                speaker = speaker.strip()
                message = message.strip()
                
                # Replace masked names with original names in the speaker
                for masked_name, original_name in shortest_names.items():
                    speaker = speaker.replace(masked_name, original_name)
                
                if current_speaker and current_speaker != speaker:
                    chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")
                    current_message = []
                
                current_speaker = speaker
                
                # Replace masked names with original names in the message
                for masked_name, original_name in shortest_names.items():
                    message = message.replace(masked_name, original_name)
                
                current_message.append(message + "\n")
            else:
                if current_speaker:
                    chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")
                    current_message = []
                    current_speaker = None
                
                # Replace masked names with original names in the narrator's line
                for masked_name, original_name in shortest_names.items():
                    line = line.replace(masked_name, original_name)
                
                chatml_file.write(f"<|im_start|>Narrator\n{line.strip()}<|im_end|>\n")
        
        if current_speaker:
            chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")
    # Remove empty lines
    with open(chatml_output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(chatml_output_path, 'w', encoding='utf-8') as f:
        f.writelines(line for line in lines if line.strip())
    print(f"ChatML format saved to {chatml_output_path}")
    print(f"Final output saved to {output_file_path}")

def clear_bin_dir():
    for filename in os.listdir(BIN_DIR):
        file_path = os.path.join(BIN_DIR, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing {filename}: {e}")

def main():
    for directory in [BIN_DIR, EBOOKS_DIR, OUTPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            if directory == EBOOKS_DIR:
                print("Created ./ebooks, put your books there")
                sys.exit(1)
    clear_bin_dir()

    context_limit = get_koboldai_context_limit()
    if DEBUG:
        print(f"Context limit: {context_limit}")

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
    
    # Start conversion of books
    for filename in os.listdir(BIN_DIR):
        if filename.endswith('.json') and not filename.endswith('_chunk_info.json') and not filename.endswith('_converted.json'):
            start_conversion_of_book(filename, context_limit)

    clear_bin_dir()

if __name__ == "__main__":
    main()
