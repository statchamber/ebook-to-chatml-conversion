# ebook to chatml conversion tool
## description
This tool converts ebooks (in .txt or .epub format) into both dialogue based and chatml formats. You can then use these formats for creating datasets. This script uses [koboldcpp](https://github.com/LostRuins/koboldcpp) for gbnf and alpaca for all of the prompts, which can be seen in [prompts.py](prompts.py). The script works decently well even with 7B models. Note that you can use this script with any context size you want (2048, 4096, 8192, or even 32K context size) by editing [config.yaml](config.yaml), but it is better to have 8192+ context.
## how does it work?
1. get text from ./ebooks (.txt or .epub)
2. breaks text into chunks
3. finds character names using AI and replace them with masked names
4.  creates short summaries of each chunk
5. using gbnf grammar and a summary, convert each chunk to dialogue format
6. turns masked names back into the original character names
7.  saves as plain text and ChatML format
## setup
1. Install [koboldcpp](https://github.com/LostRuins/koboldcpp) and load a gguf model with at least 4096 context
2. install dependencies `pip install -r requirements.txt`
3. edit [config.yaml](config.yaml) and change settings for example `max_convert` to how many paragraphs you want to convert (default chunk settings are for 8192 context)
4. put your ebooks in `./ebooks`
5. run `python index.py` and the result should show up in `./output`
## config help
- chunk: decrease chunk.size and chunk.context if you are using lower conntext like 2048 or 4096. increase it if you are using higher values like 32k.
- api: you can use gemini from google to speed up summarization. get the api key here: https://aistudio.google.com/app/apikey
## examples
