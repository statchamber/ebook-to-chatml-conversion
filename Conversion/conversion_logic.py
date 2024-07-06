import re
import os
import json
from .text_processing import call_ner, string_similarity
from .api_calls import kobold_generate_text, gemini_generate_text
from .prompts import Prompts

def start_conversion_of_book(filename, context_limit, BIN_DIR, OUTPUT_DIR, PARAGRAPH_CHUNK_SIZE, MAX_PARAGRAPHS_TO_CONVERT, CONTEXT_PARAGRAPHS, ADD_NARRATOR_TO_CHARACTERLIST, ADD_UNKNOWN_TO_CHARACTERLIST, CONFIDENCE, USE_GEMINI_SUMMARIZATION, DEBUG, SIMILARITY_THRESHOLD, tagger, KOBOLDAPI, GEMINI, STOP_SEQUENCES):
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
        # Skip this iteration if prompt is empty
        if not prompt.strip():
            continue
        
        # Update high_confidence_persons and masked_names for each chunk
        new_persons = call_ner(prompt, tagger, CONFIDENCE)
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
            summary = gemini_generate_text(summaryprompt, [], GEMINI, STOP_SEQUENCES)
            if summary == "No response generated after 10 attempts.":
                summary = kobold_generate_text(summaryprompt, 0.5, "", 500, context_limit, True, KOBOLDAPI, STOP_SEQUENCES)
        else:
            summary = kobold_generate_text(summaryprompt, 0.5, "", 500, context_limit, True, KOBOLDAPI, STOP_SEQUENCES)

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
            
            chunk = prompt_lines[j:j+5]
            context_lines = masked_total_lines[j+5:j+5+CONTEXT_PARAGRAPHS]  # Get the next CONTEXT_PARAGRAPHS lines for context from masked_total_lines
            excerpt = intra_chunk_context + "\n" + ("\n".join(chunk + context_lines))  # Combine chunk and context lines
            extracted_lines = "\n".join([f"Line{k+1}: {line}" for k, line in enumerate(chunk)])
            conversionprompt = Prompts.ConversionPrompt.replace("{speakers}", ', '.join(set(masked_names.values()))).replace("{summary}", summary).replace("{excerpt}", excerpt).replace("{extracted_lines}", extracted_lines)
            conversion = kobold_generate_text(conversionprompt, 0.5, Prompts.ConversionGrammar.replace("{speakers}", formatted_speakers), 500, context_limit, True, KOBOLDAPI, STOP_SEQUENCES)
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
                        speaker = conversion_json[line_key].get("speaker", "Narrator")
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

                        # Regular expression to match quoted and non-quoted parts
                        pattern = re.compile(r'"([^"]+)"|([^"]+)')
                            
                        # Find all matches
                        matches = pattern.finditer(line)
                            
                        # Check if there is no unquoted text, just a quote (example: "What?")
                        result = ""
                        group_2_found = False
                        for match in matches:
                            if match.group(2): # unquoted
                                group_2_found = True
                                break
                        if not group_2_found and speaker != "Narrator": # narrator cant talk
                            line += f" {speaker} says."
                            
                        converted_prompt.append(f"{speaker}: {line.strip()}")
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
    chatml_output_path = os.path.join(OUTPUT_DIR, f"{filename}_chatml.txt")

    # Create a dictionary to store the shortest original name for each masked name
    shortest_names = {masked: min((original for original, mask in masked_names.items() if mask == masked), key=len)
                for masked in set(masked_names.values())}
    def unmask_names(text):
        for masked, original in sorted(shortest_names.items(), key=lambda x: int(x[0].split('_')[-1]) if x[0].split('_')[-1].isdigit() else 0, reverse=True):
            text = text.replace(masked, original)
        return text

    with open(output_file_path, 'w', encoding='utf-8') as output_file, open(chatml_output_path, 'w', encoding='utf-8') as chatml_file:
        current_speaker = None
        current_message = []

        for line in converted_prompts:
            if ':' in line:
                speaker, message = map(str.strip, line.split(':', 1))
                speaker = unmask_names(speaker)
                message = unmask_names(message)

                if current_speaker and current_speaker != speaker:
                    chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")
                    current_message = []

                current_speaker = speaker
                current_message.append(message + "\n")
                output_file.write(f"{speaker}: {message}\n")
            elif current_speaker:
                chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")
                current_message = []
                current_speaker = None

        if current_speaker:
            chatml_file.write(f"<|im_start|>{current_speaker}\n{''.join(current_message).strip()}<|im_end|>\n")

    # Remove empty lines from ChatML file
    with open(chatml_output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(chatml_output_path, 'w', encoding='utf-8') as f:
        f.writelines(line for line in lines if line.strip())

    print(f"ChatML format saved to {chatml_output_path}")
    print(f"Final output saved to {output_file_path}")