import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords only if missing
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(words)
