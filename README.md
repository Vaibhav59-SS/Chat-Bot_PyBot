# PyBot — NLP Chatbot for Python Queries

A retrieval-based chatbot that uses **NLTK** for tokenization/lemmatization and
**TF-IDF + cosine similarity** (scikit-learn) to find the most relevant answer
to a user's question from a knowledge base of Python Q&A and module
descriptions.

## Files

- `bot.py` — core NLP logic (loads the text knowledge base, builds TF-IDF
  vectors, handles greetings/small talk, and returns the best-matching answer)
- `gui.py` — Tkinter desktop chat window (with text-to-speech via `pyttsx3`)
- `app.py` — Flask web app exposing PyBot in a browser-based chat UI
- `templates/index.html` — chat page used by `app.py`
- `static/favicon.ico` — browser tab icon
- `nlp_python_answer_finals.txt` — general Python Q&A knowledge base
- `modules_pythons.txt` — Python standard library module descriptions
- `i.ico` — desktop window icon (used by `gui.py`)

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. The first time you run `bot.py` (or `gui.py`), NLTK will automatically
   download the `punkt` (sentence/word tokenizer) and `wordnet`
   (lemmatizer) data packages — make sure you're connected to the internet
   for that first run.

3. **(Optional) Enable general Q&A via Gemini.** If a question doesn't match
   anything in the local knowledge base, PyBot will ask the **Gemini API**
   (free tier via Google AI Studio) for an answer. To enable this, get a free
   API key from https://aistudio.google.com/app/apikey, then set it as an
   environment variable before running:

   ```bash
   # macOS/Linux
   export GEMINI_API_KEY="your-api-key-here"

   # Windows (Command Prompt)
   set GEMINI_API_KEY=your-api-key-here

   # Windows (PowerShell)
   $env:GEMINI_API_KEY="your-api-key-here"
   ```

   Without this set, PyBot still works fine for Python topics — it just
   won't be able to answer general/off-topic questions.

## Running

**Command-line chat (no GUI):**

```bash
python bot.py
```

Type questions like:
- `what is python`
- `what is a tuple`
- `what is the pickle module`
- `your name`
- `tell me a joke`
- `how are you`
- `what time is it`
- `what's the date`
- `who made you`
- `what can you do`
- type `bye` to exit

**Graphical chat window (Tkinter desktop app):**

```bash
python gui.py
```

**Web chat interface (Flask):**

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. You'll get a
terminal-styled chat page — type a message and press Enter/Send.

## How it works

1. The two text files are loaded and split into sentences with
   `nltk.sent_tokenize`.
2. User input goes through a series of checks:
   - greetings (`hi`, `hello`, etc.)
   - small talk (`thanks`, `bye`, `your name`)
   - module-specific questions (anything containing the word "module") are
     matched against `modules_pythons.txt`
   - **general small talk**: jokes, "how are you", current time/date, "who
     made you", "what can you do", and simple mood replies — handled by
     `small_talk()` in `bot.py`
   - a couple of hard-coded "what is python / what is a module" answers
   - everything else is matched against `nlp_python_answer_finals.txt`
3. Matching uses `TfidfVectorizer` (with lemmatized tokens and English stop
   words removed) and `cosine_similarity` to find sentences in the knowledge
   base most similar to the user's question. PyBot now returns up to the
   **top 3 matches** (any with a similarity score above `0.05`), shown as a
   numbered list when there's more than one. If nothing scores above the
   threshold, PyBot replies that it doesn't understand.

   You can tune this in `bot.py`'s `response()` function:
   - `top_n` — how many answers to return (default `3`)
   - `min_score` — minimum similarity required to include a result (default `0.05`)

4. **General questions**: if the local knowledge base finds no good match,
   PyBot sends the question to the **Gemini API** (`ask_gemini()` in
   `bot.py`, using the free `gemini-2.5-flash` model) and returns its
   answer. This requires the `GEMINI_API_KEY` environment variable to be set
   — see Setup step 3 above. If the key isn't set, PyBot explains this
   instead of giving an error.

## Extending the knowledge base

To teach PyBot more, just add new sentences (ending in `.`) to
`nlp_python_answer_finals.txt` (general topics) or `modules_pythons.txt`
(module-specific topics). No code changes needed — the TF-IDF matcher will
pick them up automatically.
