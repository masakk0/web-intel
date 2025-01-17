import os
from flask import Flask, render_template, request, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, URL, Optional
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from dotenv import load_dotenv
from datetime import datetime
from urllib.parse import urlparse
from collections import deque
from html.parser import HTMLParser
import requests
import re
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings
from bs4 import BeautifulSoup

# ---------------------- Configuration ----------------------
# Load environment variables
load_dotenv()



app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///' + os.path.join(app.instance_path, 'site.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure the instance folder exists
if not os.path.exists(app.instance_path):
    os.makedirs(app.instance_path)

# Initialize database
db = SQLAlchemy(app)
HTTP_URL_PATTERN = r'^http[s]*://.+'
DEFAULT_API_KEY = os.getenv('OPENAI_API_KEY', '')

# ---------------------- Database Model ----------------------
class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)
    question = db.Column(db.String(500), nullable=False)
    answer = db.Column(db.Text, nullable=True)
    date_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"Query('{self.url}', '{self.question}', '{self.date_created}')"

# ---------------------- Form ----------------------
class URLForm(FlaskForm):
    url = StringField('URL', validators=[DataRequired(), URL()])
    question = StringField('Question', validators=[DataRequired()])
    api_key = StringField('OpenAI API Key (Mandatory)', validators=[Optional()])
    submit = SubmitField('Submit')

# ---------------------- Utility Functions ----------------------
def create_required_directories():
    """Create necessary directories if they don't exist."""
    for directory in ['text', 'processed']:
        os.makedirs(directory, exist_ok=True)

def remove_newlines(text):
    """Remove newline characters from a string."""
    return text.replace('\n', ' ').replace('\r', ' ')

# ---------------------- Hyperlink Parsing ----------------------
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hyperlinks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

    def get_hyperlinks(self, html):
        self.feed(html)
        return self.hyperlinks

def get_hyperlinks(url):
    """Fetch hyperlinks from a webpage."""
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        if not response.headers.get('Content-Type', '').startswith("text/html"):
            return []
        return HyperlinkParser().get_hyperlinks(response.text)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def get_domain_hyperlinks(local_domain, url):
    """Filter hyperlinks to the same domain."""
    clean_links = []
    for link in set(get_hyperlinks(url)):
        if re.search(HTTP_URL_PATTERN, link):
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_links.append(link)
        elif link.startswith("/"):
            clean_links.append(f"https://{local_domain}{link}")
    return list(set(clean_links))

# ---------------------- Crawling ----------------------
def crawl(urls):
    """Crawl and store web content."""
    if not urls:
        return "No URLs provided for crawling."

    create_required_directories()
    queue = deque(urls)
    seen = set()
    local_domain = urlparse(urls[0]).netloc
    domain_dir = f"text/{local_domain}/"
    os.makedirs(domain_dir, exist_ok=True)

    while queue and len(seen) < 100:
        url = queue.popleft()
        if url in seen:
            continue

        seen.add(url)
        print(f"Crawling: {url}")
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            text = BeautifulSoup(response.text, "html.parser").get_text()

            filename = os.path.join(domain_dir, url.replace("https://", "").replace("/", "_") + ".txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)

            new_links = get_domain_hyperlinks(local_domain, url)
            queue.extend(link for link in new_links if link not in seen)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue
    return "Crawling completed."

# ---------------------- Data Processing ----------------------
def split_into_many(text, max_tokens=5000):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    sentences = text.split('. ')
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    chunks, tokens_so_far, chunk = [], 0, []

    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            tokens_so_far, chunk = 0, []
        if token <= max_tokens:
            chunk.append(sentence)
            tokens_so_far += token + 1

    return chunks

def process_crawled_data(local_domain):
    try:
        domain_dir = f"text/{local_domain}/"
        if not os.path.exists(domain_dir):
            return "No crawled data found."

        texts = []
        for file in os.listdir(domain_dir):
            with open(os.path.join(domain_dir, file), "r", encoding="utf-8") as f:
                texts.append((file[:-4].replace('_', '/'), remove_newlines(f.read())))

        df = pd.DataFrame(texts, columns=['fname', 'text'])
        df.to_csv('processed/scraped.csv', index=False)

        tokenizer = tiktoken.get_encoding("cl100k_base")
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

        shortened = [chunk for _, row in df.iterrows() for chunk in split_into_many(row['text'])]
        df = pd.DataFrame(shortened, columns=['text'])
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

        api_key = session.get('api_key', DEFAULT_API_KEY)
        if not api_key:
            return "OpenAI API key not found."

        openai.api_key = api_key
        df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
        df.to_csv('processed/embeddings.csv', index=False)
        return "Data processed successfully."
    except Exception as e:
        return f"Error processing data: {e}"

# ---------------------- Context and Answer Generation ----------------------
def create_context(question, df, max_len=5000, top_n=5):
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].tolist(), distance_metric='cosine')
    context = "\n\n###\n\n".join(df.sort_values('distances').head(top_n)['text'].tolist())
    return context[:max_len]

def answer_question(df, question, max_len=5000, max_tokens=500):
    context = create_context(question, df, max_len=max_len)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer the question based on the context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

# ---------------------- Flask Routes ----------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    form = URLForm()
    queries = Query.query.order_by(Query.date_created.desc()).limit(10).all()

    if form.validate_on_submit():
        try:
            url, question = form.url.data, form.question.data
            api_key = form.api_key.data

            if not api_key:
                flash("OpenAI API key is required.", "error")
                return render_template('index.html', form=form, queries=queries)

            session['api_key'] = api_key

            local_domain = urlparse(url).netloc
            subpages = get_domain_hyperlinks(local_domain, url)
            crawl_result = crawl([url] + subpages)

            if "error" in crawl_result.lower():
                flash(crawl_result, "error")
                return render_template('index.html', form=form, queries=queries)

            process_result = process_crawled_data(local_domain)
            if "error" in process_result.lower():
                flash(process_result, "error")
                return render_template('index.html', form=form, queries=queries)

            df = pd.read_csv('processed/embeddings.csv')
            df['embeddings'] = df['embeddings'].apply(eval)
            answer = answer_question(df, question)

            new_query = Query(url=url, question=question, answer=answer)
            db.session.add(new_query)
            db.session.commit()

            return render_template('results.html', answer=answer)
        except Exception as e:
            flash(f"Error: {e}", "error")

    return render_template('index.html', form=form, queries=queries)

@app.route('/history')
@app.route('/history/<int:page>')
def history(page=1):
    per_page = 10
    queries = Query.query.order_by(Query.date_created.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template('history.html', queries=queries)

# ---------------------- Main ----------------------
if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()  # Ensure all tables are created before running the app
            print("Database initialized successfully.")
        except Exception as e:
            print(f"Error creating DB tables: {e}")
        # Delete all history records from the Query table
        Query.query.delete()
        db.session.commit()
    app.run(debug=True)
