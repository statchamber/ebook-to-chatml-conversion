import re
import os
import json
import time
import copy
import threading
from .text_processing import call_ner, string_similarity
from .api_calls import generate_text, generate_summary_text
from .prompts import Prompts
import shutil
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def update_progress(story_name, percentage, time_index, total_detected, eta_hours, eta_minutes, eta_seconds):
    with print_lock:
        terminal_width = shutil.get_terminal_size().columns
        progress_str = f"{story_name}: {percentage:.2f}% ({time_index}/{total_detected}) | ETA: {int(eta_hours)}h {int(eta_minutes)}m {int(eta_seconds)}s"
        progress_str = progress_str.ljust(terminal_width)[:terminal_width]
        print(f"\r{progress_str}", end="", flush=True)

def start_conversion_of_book(filename, context_limit, BIN_DIR, OUTPUT_DIR, SUMMARIZE_EVERY, MAX_PARAGRAPHS_TO_CONVERT, CONTEXT_PARAGRAPHS, CHARACTER_LIST, CONFIDENCE, USE_GEMINI_SUMMARIZATION, DEBUG, SIMILARITY_THRESHOLD, KOBOLDAPI, OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL, GEMINI_API_KEY, STOP_SEQUENCES, config):
    filename = os.path.splitext(filename)[0]
    safe_print("-"*100)
    safe_print(f"Starting conversion of book: {filename}")
    safe_print("-"*100)
    with open(os.path.join(BIN_DIR, f"{filename}.json"), 'r', encoding='utf-8') as f:
        paragraphs = json.load(f)
        unmasked_paragraphs = copy.deepcopy(paragraphs)

    # Setup variables we need throughout converting
    previous_summary = ""
    masked_names = {character: character for character in CHARACTER_LIST}
    high_confidence_characters = []
    speakers_list = list(CHARACTER_LIST)
    technical_data = []
    character_last_mentioned = {character: 0 for character in CHARACTER_LIST}
    
    if SUMMARIZE_EVERY > CONTEXT_PARAGRAPHS:
        SUMMARIZE_EVERY = CONTEXT_PARAGRAPHS

    def process_speaker(speaker, update):
        # Remove leading "the " or "a", capitalize first letter, and remove parentheses
        speaker = re.sub(r'^(the |a )', '', speaker, flags=re.IGNORECASE).capitalize()
        speaker = re.sub(r'\([^)]*\)', '', speaker).strip()
            
        # Handle unspecified speakers
        if any(string_similarity(speaker.lower(), s) >= 0.8 for s in ["not specified", "n/a", "unnamed", "null"]):
            return next((s for s in ["Narrator", "Unknown"] if s in speakers_list), speaker)
        
        if "character_" not in speaker.lower():
            # Find similar speaker or use the original
            similar_speaker = next((s for s in speakers_list if string_similarity(s.lower(), speaker.lower()) >= SIMILARITY_THRESHOLD), speaker)
            if update and similar_speaker not in speakers_list:
                speakers_list.append(similar_speaker)
            return similar_speaker
        else:
            # Special case for "'s" (possession)
            character_match = re.search(r'Character_(\d+)\'s\s+(.+)', speaker)
            if character_match:
                new_speaker = character_match.group(2).capitalize()
                if update and new_speaker not in speakers_list:
                    speakers_list.append(new_speaker)
                return new_speaker
            else:
                # Extract character number or use "Unknown"
                character_number = re.search(r'character_(\d+)', speaker.lower())
                speaker = f"Character_{character_number.group(1)}" if character_number else "Unknown"
                if update and speaker not in speakers_list:
                    speakers_list.append(speaker)
                return speaker
        
    # For shorter stories    
    total_detected = len(paragraphs)
    if len(paragraphs) < MAX_PARAGRAPHS_TO_CONVERT:
        total_detected = len(paragraphs)
    else:
        total_detected = MAX_PARAGRAPHS_TO_CONVERT
    
    # Start converting the story 5 lines at a time
    start_time = time.time()
    for i in range(0, MAX_PARAGRAPHS_TO_CONVERT, 5):
        current_lines = paragraphs[i:i+5]
        prev_lines = paragraphs[max(0, i-CONTEXT_PARAGRAPHS):i]
        next_lines = paragraphs[i+5:i+5+CONTEXT_PARAGRAPHS]

        if not current_lines:
            continue

        unmasked_lines = unmasked_paragraphs[i:i+5]
        unmasked_prev_lines = unmasked_paragraphs[max(0, i-CONTEXT_PARAGRAPHS):i]
        unmasked_next_lines = unmasked_paragraphs[i+5:i+5+CONTEXT_PARAGRAPHS]
        
        prompt = "\n".join(prev_lines + current_lines + next_lines)
        unmasked_prompt = "\n".join(unmasked_prev_lines + unmasked_lines + unmasked_next_lines)

        # Detect and mask character names
        detected_characters = call_ner(unmasked_prompt, CONFIDENCE)
        for character in detected_characters:
            if character['text'] not in [p['text'] for p in high_confidence_characters]:
                high_confidence_characters.append(character)
                masked_names[character['text']] = f"Character_{len(masked_names) + 1}"
            character_last_mentioned[character['text']] = i

        # Check for possible aliases
        for i, character1 in enumerate(high_confidence_characters):
            for character2 in high_confidence_characters[i+1:]:
                if character1['text'].lower() in character2['text'].lower() or character2['text'].lower() in character1['text'].lower():
                    masked_names[character2['text']] = masked_names[character1['text']]
        
        # Print detected characters and their masked names if debug is on
        if DEBUG:
            safe_print("\nDetected characters and their masked names:")
            for original_name, masked_name in masked_names.items():
                safe_print(f"{original_name}: {masked_name}")
            safe_print()  # Add an empty line for better readability

        # Replace characters with masked names
        def replace_names(text):
            for original_name, masked_name in sorted(masked_names.items(), key=lambda x: len(x[0]), reverse=True):
                text = re.sub(r'\b' + re.escape(original_name) + r'\b', masked_name, text, flags=re.IGNORECASE)
            return text

        changed_current_lines = [replace_names(line) for line in current_lines]
        changed_prev_lines = [replace_names(line) for line in prev_lines]
        changed_next_lines = [replace_names(line) for line in next_lines]
        changed_prompt = replace_names(prompt)

        # Create a summary from the prompt every x lines
        index = paragraphs.index(current_lines[0]) 
        if index % SUMMARIZE_EVERY == 0:
            # Only include characters mentioned in the last SUMMARIZE_EVERY lines
            recent_characters = [char for char, last_mention in character_last_mentioned.items() 
                                 if index - last_mention <= SUMMARIZE_EVERY]
            recent_masked_names = [masked_names[char] for char in recent_characters]
            
            summaryprompt = Prompts.SummarizationPrompt.replace("{speakers}", ', '.join(set(recent_masked_names))).replace("{prompt}", changed_prompt).replace("{previous_summary}", f"Previous Summary:\n{previous_summary}" if previous_summary else "")
            summary = generate_summary_text(summaryprompt, 0.5, "", 500, context_limit, True, KOBOLDAPI, OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL, GEMINI_API_KEY, STOP_SEQUENCES)
            
            # If summary failed to generate, fallback to previous summary
            if not summary or summary == "Failed":
                summary = previous_summary

            previous_summary = summary
            technical_data.append({"line": index, "summary": summary})

            if DEBUG:
                safe_print(f"\nUpdated summary at index: {index}")
                safe_print("-"*100)
                safe_print(summary)
                safe_print("-"*100)
        else:
            summary = ""

        # Prepare data for conversion
        excerpt = "\n".join(changed_prev_lines + changed_current_lines + changed_next_lines)
        
        # Only include characters mentioned in the last CONTEXT_PARAGRAPHS lines
        recent_characters = [char for char, last_mention in character_last_mentioned.items() 
                             if index - last_mention <= CONTEXT_PARAGRAPHS]
        recent_masked_names = [masked_names[char] for char in recent_characters]
        
        formatted_speakers = ' | '.join([f'"\\"{masked_name}\\""' for masked_name in set(recent_masked_names)]) + " | string"
        extracted_lines = "\n".join([f"Line{k+1}: {line}" for k, line in enumerate(changed_current_lines)])

        # Convert the lines
        max_retries = config['chunk'].get('max_retries', 3)  # Default to 3 if not specified
        for attempt in range(max_retries):
            conversionprompt = Prompts.ConversionPrompt.replace("{speakers}", ', '.join(set(recent_masked_names))).replace("{summary}", summary).replace("{excerpt}", excerpt).replace("{extracted_lines}", extracted_lines)
            conversion = generate_text(conversionprompt, 0.5, Prompts.ConversionGrammar.replace("{speakers}", formatted_speakers), 500, context_limit, True, KOBOLDAPI, OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL, GEMINI_API_KEY, STOP_SEQUENCES)
            
            # Attempt to extract only the JSON content between the first { and last }
            json_match = re.search(r'\{[\s\S]*\}', conversion)
            if json_match:
                conversion = json_match.group(0)
            else:
                safe_print(f"\nAttempt {attempt + 1}: No valid JSON found in the response:\n\n{conversion}\n\nRetrying...")
                continue

            try:
                conversion_json = json.loads(conversion)
                if DEBUG:
                    safe_print(f"\n{conversionprompt}\n{'-'*100}{conversion}\n{'-'*100}")
                break  # Successfully parsed JSON, exit the retry loop
            except json.JSONDecodeError:
                if DEBUG:
                    safe_print(f"\nAttempt {attempt + 1}: JSONDecodeError. Retrying...")
                if attempt == max_retries - 1:
                    safe_print("Max retries reached. Using empty JSON.")
                    conversion_json = {}

            # Quick test to see if Line1 speaker exists or if json is broken
            try:
                conversion_json["Line1"]["speaker"]
            except KeyError:
                if DEBUG:
                    safe_print("KeyError: Line1")
                if attempt == max_retries - 1:
                    safe_print("Max retries reached. Using empty JSON.")
                    conversion_json = {}
                else:
                    continue
        
        for k, line in enumerate(changed_current_lines):
            line_key = f"Line{k+1}"
            if line_key in conversion_json:
                try:
                    speaker = process_speaker(str(conversion_json[line_key].get("speaker", "Narrator")), True)
                    action = str(conversion_json[line_key].get("action", ""))
                    talking_to = process_speaker(str(conversion_json[line_key].get("talking_to", "")), False)
                    
                    # Update last mentioned for speaker and talking_to
                    if speaker in character_last_mentioned:
                        character_last_mentioned[speaker] = i + k
                    if talking_to in character_last_mentioned:
                        character_last_mentioned[talking_to] = i + k
                    
                    # Calculate and print progress
                    time_index = paragraphs.index(current_lines[-1]) + 1
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    percentage = time_index / total_detected * 100
                    eta_seconds = (elapsed_time / time_index) * (total_detected - time_index)
                    eta_hours, eta_remainder = divmod(eta_seconds, 3600)
                    eta_minutes, eta_seconds = divmod(eta_remainder, 60)

                    update_progress(filename, percentage, time_index, total_detected, eta_hours, eta_minutes, eta_seconds)
                    
                    # Update line with converted information
                    if current_lines[k] in paragraphs:
                        # Add to technical_data
                        technical_data.append({
                            "line": paragraphs.index(current_lines[k]),
                            "speaker": speaker,
                            "alias": [alias for alias, masked in masked_names.items() if masked == speaker and alias != speaker],
                            "talking_to": talking_to,
                            "action": action,
                            "content": str(line).strip()
                        })
                        paragraphs[paragraphs.index(current_lines[k])] = f"{speaker} talking to {talking_to} ({action}): {str(line).strip()}"
                except Exception as e:
                    safe_print(f"\nUnknown Error! \n\nJson: {conversion}\n\nError: {str(e)}")
                    # Unknown error, fallback to original line
                    if current_lines[k] in paragraphs:
                        paragraphs[paragraphs.index(current_lines[k])] = line.strip()
            else:
                # Error parsing JSON, fallback to original line
                if current_lines[k] in paragraphs:
                    paragraphs[paragraphs.index(current_lines[k])] = line.strip()

    safe_print(f"\n{'-'*100}\nConversion completed for {filename}")

    # Prepare for output
    output_file_path = os.path.join(OUTPUT_DIR, f"{filename}_converted.txt")
    chatml_output_path = os.path.join(OUTPUT_DIR, f"{filename}_chatml.txt")
    technical_output_path = os.path.join(OUTPUT_DIR, f"{filename}_technical.json")
    current_speaker = None
    current_message = []

    # Create a dictionary to store the shortest original name for each masked name
    shortest_names = {masked: min((original for original, mask in masked_names.items() if mask == masked), key=len)
                      for masked in set(masked_names.values())}

    def unmask_names(text):
        for masked, original in sorted(shortest_names.items(), key=lambda x: int(x[0].split('_')[-1]) if x[0].split('_')[-1].isdigit() else 0, reverse=True):
            text = text.replace(masked, original)
        return text

    def clean_unicode(text):
        return text.translate({
            ord('\u2018'): "'",
            ord('\u2019'): "'",
            ord('\u201c'): '"',
            ord('\u201d'): '"',
            ord('\u201e'): '"',
            ord('\u201f'): '"',
            ord('\u2014'): '--',
            ord('\u2013'): '-',
            ord('\u2026'): '...',
            ord('\u00a0'): ' ',
            ord('\u00b0'): '°',
            ord('\u00e9'): 'e',
            ord('\u00e8'): 'e',
            ord('\u00f1'): 'n',
            ord('\u00fc'): 'u',
            ord('\u00f6'): 'o',
            ord('\u00e4'): 'a',
            ord('\u00df'): 'ss',
        })

    # Save the converted lines
    with open(config['output']['regular'] and output_file_path, 'w', encoding='utf-8') as output_file, \
         open(config['output']['chatml'] and chatml_output_path, 'w', encoding='utf-8') as chatml_file:
        for line in paragraphs[:MAX_PARAGRAPHS_TO_CONVERT]:
            if ':' in line:
                speaker, content = map(str.strip, line.split(':', 1))
                speaker, content = map(unmask_names, (speaker, content))
                content = clean_unicode(content)

                # Extract only the speaker name without additional information
                speaker_name = speaker.split(' talking to ')[0]

                if current_speaker and current_speaker != speaker_name:
                    if config['output']['chatml']:
                        chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")
                    current_message = []

                current_speaker = speaker_name
                current_message.append(content + "\n")
                if config['output']['regular']:
                    output_file.write(f"{speaker_name}: {content}\n")
            else:
                if current_speaker:
                    if config['output']['chatml']:
                        chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")
                    current_message = []
                    current_speaker = None
                if config['output']['regular']:
                    cleaned_line = clean_unicode(unmask_names(line))
                    output_file.write(f"{cleaned_line}\n")

        # Write any remaining message
        if current_speaker and config['output']['chatml']:
            chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")

    # Remove empty lines from ChatML file if it was created
    if config['output']['chatml']:
        with open(chatml_output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(chatml_output_path, 'w', encoding='utf-8') as f:
            f.writelines(line for line in lines if line.strip())
        safe_print(f"ChatML format saved to {chatml_output_path}")

    if config['output']['regular']:
        safe_print(f"Regular format saved to {output_file_path}")

    # Save technical data
    if config['output']['technical']:
        unmasked_technical_data = [
            {
                "line": data["line"] if "line" in data else None,
                "speaker": clean_unicode(unmask_names(data["speaker"])) if "speaker" in data else None,
                "alias": [clean_unicode(unmask_names(alias)) for alias in data.get("alias", [])],
                "talking_to": clean_unicode(unmask_names(data["talking_to"])) if "talking_to" in data else None,
                "action": clean_unicode(unmask_names(data["action"])) if "action" in data else None,
                "content": clean_unicode(unmask_names(data["content"])) if "content" in data else None,
                "summary": clean_unicode(unmask_names(data["summary"])) if "summary" in data else None
            }
            for data in technical_data
        ]
        with open(technical_output_path, 'w', encoding='utf-8') as f:
            json.dump(unmasked_technical_data, f, indent=2)
        safe_print(f"Technical data saved to {technical_output_path}")
