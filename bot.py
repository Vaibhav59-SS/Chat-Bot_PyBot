# Meet Pybot: your friend
# Retrieval-based NLP chatbot using NLTK + TF-IDF / cosine similarity

import os
import random
import string
import warnings
from datetime import datetime

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# One-time NLTK data download (only runs if the data isn't already present)
# ---------------------------------------------------------------------------
def _ensure_nltk_data():
    for pkg, path in (("punkt", "tokenizers/punkt"),
                       ("punkt_tab", "tokenizers/punkt_tab"),
                       ("wordnet", "corpora/wordnet")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


_ensure_nltk_data()

# ---------------------------------------------------------------------------
# Load knowledge base files
# ---------------------------------------------------------------------------
QA_FILE = os.path.join(BASE_DIR, "nlp_python_answer_finals.txt")
MODULES_FILE = os.path.join(BASE_DIR, "modules_pythons.txt")

with open(QA_FILE, "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read().lower()

with open(MODULES_FILE, "r", encoding="utf-8", errors="ignore") as m:
    rawone = m.read().lower()

sent_tokens = nltk.sent_tokenize(raw)
sent_tokensone = nltk.sent_tokenize(rawone)

# ---------------------------------------------------------------------------
# Lemmatization helpers
# ---------------------------------------------------------------------------
lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------
Introduce_Ans = [
    "My name is PyBot.",
    "My name is PyBot, you can call me Pi.",
    "I'm PyBot :)",
    "My name is PyBot, my nickname is Pi, and I am happy to solve your queries :)",
]

GREETING_INPUTS = ("hello", "hi", "hii", "hiii", "hiiii", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "hii there", "hi there", "hello", "I am glad! You are talking to me"]

Basic_Q = ("what is python", "what is python?", "what is python.")
Basic_Ans = ("Python is a high-level, interpreted, interactive and object-oriented scripting "
              "programming language. Python is designed to be highly readable. It uses English "
              "keywords frequently whereas other languages use punctuation, and it has fewer "
              "syntactical constructions than other languages.")

Basic_Om = (
    "what is module", "what is module.", "what is module?",
    "what is module in python", "what is module in python.", "what is module in python?",
)
Basic_AnsM = [
    "Consider a module to be the same as a code library.",
    "A file containing a set of functions you want to include in your application.",
    ("A module can define functions, classes and variables. A module can also include runnable "
     "code. Grouping related code into a module makes the code easier to understand and use."),
]

# ---------------------------------------------------------------------------
# Small talk content
# ---------------------------------------------------------------------------
MOOD_RESPONSES = [
    "I'm doing great, thanks for asking! How are you?",
    "Feeling like a fully-optimized algorithm today :) How about you?",
    "All systems running smoothly! How are you doing?",
]

CAPABILITY_RESPONSES = [
    ("I can answer Python questions, explain modules, tell jokes, share the time/date, "
     "and chat a little. Try asking 'what is a tuple' or 'tell me a joke'."),
    ("I'm PyBot — I mostly know about Python (concepts, modules, syntax) but I can also "
     "do small talk like jokes, greetings, and telling you the time."),
]

CREATOR_RESPONSES = [
    "I was built by a small team of Python developers as a learning project.",
    "My developers are Abhishek Ezhava, Mayur Kadam, Monis Khot, and Raj Vishwakarma.",
]

AGE_RESPONSES = [
    "I don't have an age — I'm just a Python script!",
    "I was 'born' the moment you ran bot.py :)",
]

JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "Why do Python programmers wear glasses? Because they can't C.",
    "How many programmers does it take to change a light bulb? None, that's a hardware problem.",
    "Why was the function sad after a breakup? It had no closure.",
    "Why did the developer go broke? Because he used up all his cache.",
    "I told my computer I needed a break, and it said 'No problem, I'll go to sleep.'",
    "There are 10 types of people: those who understand binary and those who don't.",
    "Why don't programmers like nature? It has too many bugs and no documentation.",
]


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------
def greeting(sentence):
    """If user's input is a greeting, return a greeting response."""
    for word in sentence.split():
        if word.lower().strip(string.punctuation) in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None


def basic(sentence):
    sentence = sentence.strip().lower()
    if sentence in Basic_Q:
        return Basic_Ans
    return None


def basicM(sentence):
    sentence = sentence.strip().lower()
    if sentence in Basic_Om:
        return random.choice(Basic_AnsM)
    return None


def IntroduceMe(sentence):
    return random.choice(Introduce_Ans)


def small_talk(sentence):
    """Handle casual conversational queries unrelated to Python.
    Returns a response string, or None if nothing matches.
    """
    # Mood / well-being
    if any(p in sentence for p in ("how are you", "how r u", "how're you", "how you doing")):
        return random.choice(MOOD_RESPONSES)

    # Capabilities / help
    if any(p in sentence for p in ("what can you do", "what do you do", "help me", "help")):
        return random.choice(CAPABILITY_RESPONSES)

    # Jokes
    if any(p in sentence for p in ("joke", "make me laugh", "funny")):
        return random.choice(JOKES)

    # Time
    if any(p in sentence for p in ("what time is it", "current time", "tell me the time", "what's the time")):
        return "It's currently " + datetime.now().strftime("%I:%M %p") + "."

    # Date
    if any(p in sentence for p in ("what's the date", "today's date", "what is the date", "what day is it")):
        return "Today is " + datetime.now().strftime("%A, %B %d, %Y") + "."

    # Creator
    if any(p in sentence for p in ("who made you", "who created you", "who is your creator", "who developed you", "your developer")):
        return random.choice(CREATOR_RESPONSES)

    # Age
    if any(p in sentence for p in ("how old are you", "your age")):
        return random.choice(AGE_RESPONSES)

    # Mood reciprocation
    if any(p in sentence for p in ("i am fine", "i'm fine", "i am good", "i'm good", "i am great", "i'm great")):
        return "That's great to hear! What would you like to know about Python?"

    if any(p in sentence for p in ("i am sad", "i'm sad", "i am bored", "i'm bored", "i feel bad")):
        return "Sorry to hear that. Want to hear a joke, or ask me a Python question to take your mind off it?"

    return None


# ---------------------------------------------------------------------------
# Gemini API fallback (for general questions outside the local knowledge base)
# ---------------------------------------------------------------------------
_gemini_model = None
_gemini_configured = False


def _get_gemini_model():
    """Lazily create the Gemini model client. Returns None if unavailable."""
    global _gemini_model, _gemini_configured

    if not _GEMINI_AVAILABLE:
        return None

    if _gemini_model is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        if not _gemini_configured:
            genai.configure(api_key=api_key)
            _gemini_configured = True

        _gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    return _gemini_model


def ask_gemini(user_response):
    """Ask Gemini (Google AI Studio free tier) for a general-purpose answer.

    Returns the reply text, or a friendly error message if the API
    key/library isn't set up or the request fails.
    """
    model = _get_gemini_model()

    if model is None:
        return ("I don't have a local answer for that, and my connection to "
                "Gemini isn't set up yet. Set the GEMINI_API_KEY environment "
                "variable and install the 'google-generativeai' package to "
                "enable general Q&A.")

    try:
        result = model.generate_content(user_response)
        return (result.text or "").strip()
    except Exception as e:
        return f"I tried asking Gemini but ran into an error: {e}"


# ---------------------------------------------------------------------------
# TF-IDF based retrieval
# ---------------------------------------------------------------------------
def response(user_response, tokens, top_n=3, min_score=0.05):
    """Return the top `top_n` sentences in `tokens` most similar to
    `user_response` (each with a similarity score above `min_score`).

    Falls back to "I don't understand" if nothing scores above `min_score`.
    """
    working = tokens + [user_response]

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(working)

    vals = cosine_similarity(tfidf[-1], tfidf).flatten()

    # Exclude the last entry (the user's own input compared to itself)
    scores = vals[:-1]

    # Rank sentence indices by descending similarity
    ranked_idx = scores.argsort()[::-1]

    results = []
    for idx in ranked_idx[:top_n]:
        if scores[idx] <= min_score:
            break
        results.append(working[idx])

    if not results:
        return "I am sorry! I don't understand you."

    if len(results) == 1:
        return results[0]

    # Format multiple matches as a numbered list
    numbered = [f"{i + 1}. {sentence}" for i, sentence in enumerate(results)]
    return "Here's what I found:\n" + "\n".join(numbered)


# ---------------------------------------------------------------------------
# Main chat dispatcher
# ---------------------------------------------------------------------------
def chat(user_response):
    user_response = user_response.lower().strip()

    if not user_response:
        return "I am sorry! I don't understand you."

    if user_response == "bye":
        return "Bye! take care.."

    if user_response in ("thanks", "thank you"):
        return "You are welcome.."

    # Module-related questions
    if basicM(user_response) is not None:
        return basicM(user_response)

    if " module" in user_response or "module " in user_response:
        return response(user_response, sent_tokensone)

    # Greetings
    greet = greeting(user_response)
    if greet is not None:
        return greet

    # "what's your name" style questions
    if "your name" in user_response:
        return IntroduceMe(user_response)

    # Small talk: jokes, time, date, mood, capabilities, creator, etc.
    small_talk_ans = small_talk(user_response)
    if small_talk_ans is not None:
        return small_talk_ans

    # Basic "what is python" question
    basic_ans = basic(user_response)
    if basic_ans is not None:
        return basic_ans

    # Local Python knowledge base
    local_ans = response(user_response, sent_tokens)
    if local_ans != "I am sorry! I don't understand you.":
        return local_ans

    # Final fallback: ask Gemini for a general answer
    return ask_gemini(user_response)


# ---------------------------------------------------------------------------
# Allow running this file directly for a quick command-line chat
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("PyBot: Hi! I'm PyBot. Ask me anything about Python (type 'bye' to exit).")
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nPyBot: Bye! take care..")
            break

        reply = chat(user_input)
        print("PyBot:", reply)

        if user_input.lower().strip() == "bye":
            break
