# ebook to chatml conversion tool
## description
this tool converts ebooks (in .txt or .epub format) into dialogue based and chatml formatted groupchats. you can then use this format for creating datasets. this script uses [koboldcpp](https://github.com/LostRuins/koboldcpp/releases) for gbnf and alpaca for all of the prompts, which can be seen in [prompts.py](Conversion/prompts.py). the script works decently well with 7B models but requires some editing. note that you can use this script with any context size you want (4096, 8192, or even 32K context size) by editing [config.yaml](config.yaml), but it is better to have 8192+ context.
## examples (200 lines)
Killed Once, Lived Twice by Gary Whitmore - kunoichi dpo v2 7B Q8_0 @ 8192 context ([chatml](examples/Killed-Once-Lived-Twice_chatml.txt) | [regular](examples/Killed-Once-Lived-Twice_converted.txt))

Drone World by Jim Kochanoff - gemma 2 9B @ 8192 context ([chatml](examples/Drone-World_chatml.txt) | [regular](examples/Drone-World_converted.txt))

The awakening by L C Ainsworth - kukulemon 7B Q8_0 @ 4096 context ([chatml](examples/The-awakening-Dark-Passenger_chatml.txt) | [regular](examples/The-awakening-Dark-Passenger_converted.txt))
## how does it work in 10 steps?
1. load book text (.txt or .epub) into a json ile
2. break the text into smaller chunks (5 lines at a time)
3. detect character names and aliases using an entity detection model and mask them with generic labels (Character_1, Character_2, etc)
4. create summaries of text occasionally to use in prompts and to improve accuracy
5. add context lines to the start and end of each chunk to improve accuracy
6. label/convert each chunk using few-shot prompts and GBNF grammar
7. process the converted text
8. track progress and give eta
9. unmask character names, replacing the generic lables with original names
10. save the converted lines in both plaintext and chatml format
## setup
1. run `git clone https://github.com/statchamber/ebook-to-chatml-conversion.git`
2. install [koboldcpp](https://github.com/LostRuins/koboldcpp/releases/) and load a gguf model with at least 4096 context
3. install dependencies `pip install -r requirements.txt`
4. edit [config.yaml](config.yaml) and change settings for example `max_convert` to how many paragraphs you want to convert
5. Create a folder called `./ebooks` and put your ebooks in it
6. run `python index.py` and the results should show up in `./output`
## config help
- chunk: decrease chunk.context if you are using lower context like 4096. increase it if you are using higher values like 32k. the more context lines you add, the less the AI will make mistakes. but, it will generate slower as it takes up more tokens
- api: you can use gemini from google to speed up summarization. get the api key here: https://aistudio.google.com/app/apikey
