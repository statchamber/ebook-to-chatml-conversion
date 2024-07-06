import requests
import time
import sys
import json

def kobold_generate_text(prompt, temperature, grammar, max_length, max_token_count, cleanse, KOBOLDAPI, STOP_SEQUENCES):
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
            if attempts > 1:
                print(f"Attempt {attempts + 1}: KoboldAI API not available on {KOBOLDAPI}, is the API running?")
            attempts += 1
            if attempts == maxattempts:
                sys.exit(1)
            time.sleep(1)  # Wait a second before reattempting

def gemini_generate_text(prompt, conversation_history, GEMINI, STOP_SEQUENCES):
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

def get_koboldai_context_limit(KOBOLDAPI):
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