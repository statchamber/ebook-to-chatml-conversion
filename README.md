# ebook to chatml conversion tool
## description
this tool converts ebooks (in .txt or .epub format) into both dialogue based and chatml formats. you can then use these formats for creating datasets. this script uses [koboldcpp](https://github.com/LostRuins/koboldcpp/releases) for gbnf and alpaca for all of the prompts, which can be seen in [prompts.py](prompts.py). the script works decently well with 7B models but requires some editing. note that you can use this script with any context size you want (4096, 8192, or even 32K context size) by editing [config.yaml](config.yaml), but it is better to have 8192+ context. to see examples of llm conversions scroll all the way down for examples or open the `examples` folder
## how does it work?
1. get text from ./ebooks (.txt or .epub)
2. breaks text into chunks
3. finds character names using AI and replace them with masked names
4.  creates short summaries of each chunk
5. using gbnf grammar and a summary, convert each chunk to dialogue format
6. turns masked names back into the original character names
7.  saves as plain text and chatml format
## setup
1. clone the repo
2. install [koboldcpp](https://github.com/LostRuins/koboldcpp/releases/) and load a gguf model with at least 4096 context
3. install dependencies `pip install -r requirements.txt`
4. edit [config.yaml](config.yaml) and change settings for example `max_convert` to how many paragraphs you want to convert
5. Create a folder called `./ebooks` and put your ebooks in it
6. run `python index.py` and the result should show up in `./output`
## config help
- chunk: decrease chunk.size and chunk.context if you are using lower context like 4096. increase it if you are using higher values like 32k.
- api: you can use gemini from google to speed up summarization. get the api key here: https://aistudio.google.com/app/apikey
## examples (200 lines)
Killed Once, Lived Twice by Gary Whitmore - kunoichi dpo v2 7B Q8_0 @ 8192 context ([chatml](examples/Killed-Once-Lived-Twice_chatml.txt) | [regular](examples/Killed-Once-Lived-Twice_converted.txt))

Drone World by Jim Kochanoff - gemma 2 9B @ 8192 context ([chatml](examples/Drone-World_chatml.txt) | [regular](examples/Drone-World_converted.txt))

The awakening by L C Ainsworth - kukulemon 7B Q8_0 @ 4096 context ([chatml](examples/The-awakening-Dark-Passenger_chatml.txt) | [regular](examples/The-awakening-Dark-Passenger_converted.txt))
