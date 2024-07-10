import re
import os
import json
import time
from .text_processing import call_ner, string_similarity
from .api_calls import kobold_generate_text, gemini_generate_text
from .prompts import Prompts

def start_conversion_of_book(filename, context_limit, BIN_DIR, OUTPUT_DIR, SUMMARIZE_EVERY, MAX_PARAGRAPHS_TO_CONVERT, CONTEXT_PARAGRAPHS, CHARACTER_LIST, CONFIDENCE, USE_GEMINI_SUMMARIZATION, DEBUG, SIMILARITY_THRESHOLD, tagger, KOBOLDAPI, GEMINI, STOP_SEQUENCES):
    filename = os.path.splitext(filename)[0]
    print("-"*100)
    print(f"Starting conversion of book: {filename}")
    print("-"*100)
    with open(os.path.join(BIN_DIR, f"{filename}.json"), 'r', encoding='utf-8') as f:
        paragraphs = json.load(f)

    # Setup variables we need throughout converting
    previous_summary = ""
    masked_names = {character: character for character in CHARACTER_LIST}
    high_confidence_characters = []
    speakers_list = list(CHARACTER_LIST)
    
    if SUMMARIZE_EVERY > CONTEXT_PARAGRAPHS:
        SUMMARIZE_EVERY = CONTEXT_PARAGRAPHS

    def process_speaker(speaker, update):
        # Remove leading "the" or "a", capitalize first letter, and remove parentheses
        speaker = re.sub(r'^(the |a )', '', speaker, flags=re.IGNORECASE).capitalize()
        speaker = re.sub(r'\([^)]*\)', '', speaker).strip()
            
        # Handle unspecified speakers
        if any(string_similarity(speaker.lower(), s) >= 0.8 for s in ["not specified", "n/a", "unnamed"]):
            return next((s for s in ["Narrator", "Unknown"] if s in speakers_list), speaker)
        
        if "character_" not in speaker.lower():
            # Find similar speaker or use the original
            similar_speaker = next((s for s in speakers_list if string_similarity(s.lower(), speaker.lower()) >= SIMILARITY_THRESHOLD), speaker)
            if update and similar_speaker not in speakers_list:
                speakers_list.append(similar_speaker)
            return similar_speaker
        else:
            # Special case for 's (possession)
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
        
        prompt = "\n".join(prev_lines + current_lines + next_lines)

        # Detect and mask character names
        detected_characters = call_ner(prompt, tagger, CONFIDENCE)
        for character in detected_characters:
            if character['text'] not in [p['text'] for p in high_confidence_characters]:
                high_confidence_characters.append(character)
                masked_names[character['text']] = f"Character_{len(masked_names) + 1}"
        
        # Check for possible aliases
        for i, character1 in enumerate(high_confidence_characters):
            for character2 in high_confidence_characters[i+1:]:
                if character1['text'].lower() in character2['text'].lower() or character2['text'].lower() in character1['text'].lower():
                    masked_names[character2['text']] = masked_names[character1['text']]

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
            summaryprompt = Prompts.SummarizationPrompt.replace("{speakers}", ', '.join(set(masked_names.values()))).replace("{prompt}", changed_prompt).replace("{previous_summary}", f"Previous Summary:\n{previous_summary}" if previous_summary else "")
            if USE_GEMINI_SUMMARIZATION:
                summary = gemini_generate_text(summaryprompt, [], GEMINI, STOP_SEQUENCES)
                if summary == "Failed": # Fallback to kobold if gemini fails
                    summary = kobold_generate_text(summaryprompt, 0.5, "", 500, context_limit, True, KOBOLDAPI, STOP_SEQUENCES)
            else:
                summary = kobold_generate_text(summaryprompt, 0.5, "", 500, context_limit, True, KOBOLDAPI, STOP_SEQUENCES)

            previous_summary = summary

            if DEBUG:
                print(f"\nUpdated summary at index: {index}")
                print("-"*100)
                print(summary)
                print("-"*100)
        else:
            summary = ""

        # Prepare data for conversion
        excerpt = "\n".join(changed_prev_lines + changed_current_lines + changed_next_lines)
        formatted_speakers = ' | '.join([f'"\\"{masked_name}\\""' for masked_name in set(masked_names.values())]) + " | string"
        extracted_lines = "\n".join([f"Line{k+1}: {line}" for k, line in enumerate(changed_current_lines)])

        # Convert the lines
        conversionprompt = Prompts.ConversionPrompt.replace("{speakers}", ', '.join(set(masked_names.values()))).replace("{summary}", summary).replace("{excerpt}", excerpt).replace("{extracted_lines}", extracted_lines)
        conversion = kobold_generate_text(conversionprompt, 0.5, Prompts.ConversionGrammar.replace("{speakers}", formatted_speakers), 500, context_limit, True, KOBOLDAPI, STOP_SEQUENCES)

        try:
            conversion_json = json.loads(conversion)
            if DEBUG:
                print(f"\n{conversionprompt}\n{'-'*100}{conversion}\n{'-'*100}")
                
            for k, line in enumerate(changed_current_lines):
                line_key = f"Line{k+1}"
                if line_key in conversion_json:
                    speaker = process_speaker(conversion_json[line_key].get("speaker", "Narrator"), True)
                    action = conversion_json[line_key].get("action", "")
                    talking_to = process_speaker(conversion_json[line_key].get("talking_to", ""), False)
                    
                    # Calculate and print progress
                    time_index = paragraphs.index(current_lines[-1]) + 1 # :(
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    percentage = time_index / total_detected * 100
                    eta_seconds = (elapsed_time / time_index) * (total_detected - time_index)
                    eta_hours, eta_remainder = divmod(eta_seconds, 3600)
                    eta_minutes, eta_seconds = divmod(eta_remainder, 60)
                    
                    print(f"\rProgress: {percentage:.2f}% ({time_index}/{total_detected} lines converted) | ETA: {int(eta_hours)} hours, {int(eta_minutes)} minutes, {int(eta_seconds)} seconds", end="")
                    
                    # Update line with converted information
                    if current_lines[k] in paragraphs:
                        paragraphs[paragraphs.index(current_lines[k])] = f"{speaker} talking to {talking_to} ({action}): {line.strip()}"
                else:
                    print(f"\nThere has been an issue parsing the JSON. Make sure you are using the latest koboldcpp version.")
                    if current_lines[k] in paragraphs:
                        paragraphs[paragraphs.index(current_lines[k])] = line.strip()
        except json.JSONDecodeError:
            print(f"\nError parsing JSON, skipping this chunk. Make sure you are using the latest koboldcpp version.")

    print(f"\n{'-'*100}\nConversion completed for {filename}")

    # Prepare for output
    output_file_path = os.path.join(OUTPUT_DIR, f"{filename}_converted.txt")
    chatml_output_path = os.path.join(OUTPUT_DIR, f"{filename}_chatml.txt")
    current_speaker = None
    current_message = []

    # Create a dictionary to store the shortest original name for each masked name
    shortest_names = {masked: min((original for original, mask in masked_names.items() if mask == masked), key=len)
                      for masked in set(masked_names.values())}

    def unmask_names(text):
        for masked, original in sorted(shortest_names.items(), key=lambda x: int(x[0].split('_')[-1]) if x[0].split('_')[-1].isdigit() else 0, reverse=True):
            text = text.replace(masked, original)
        return text

    # Save the converted lines
    with open(output_file_path, 'w', encoding='utf-8') as output_file, open(chatml_output_path, 'w', encoding='utf-8') as chatml_file:
        for line in paragraphs[:MAX_PARAGRAPHS_TO_CONVERT]:
            if ':' in line:
                speaker, content = map(str.strip, line.split(':', 1))
                speaker, content = map(unmask_names, (speaker, content))

                # Extract only the speaker name without additional information
                speaker_name = speaker.split(' talking to ')[0]

                if current_speaker and current_speaker != speaker_name:
                    chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")
                    current_message = []

                current_speaker = speaker_name
                current_message.append(content + "\n")
                output_file.write(f"{speaker_name}: {content}\n")
            else:
                if current_speaker:
                    chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")
                    current_message = []
                    current_speaker = None
                output_file.write(f"{unmask_names(line)}\n")

        # Write any remaining message
        if current_speaker:
            chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")

    # Remove empty lines from ChatML file
    with open(chatml_output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(chatml_output_path, 'w', encoding='utf-8') as f:
        f.writelines(line for line in lines if line.strip())

    print(f"ChatML format saved to {chatml_output_path}")
    print(f"Final output saved to {output_file_path}")