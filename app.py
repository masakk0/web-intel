import os
from flask import Flask, render_template, request, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Optional
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
from typing import Literal

# Deep Research Imports
from tavily import TavilyClient
from deepagents import create_deep_agent, SubAgent

# ---------------------- Configuration ----------------------
# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')

# Ensure the instance folder exists
if not os.path.exists(app.instance_path):
    os.makedirs(app.instance_path)

HTTP_URL_PATTERN = r'^http[s]*://.+'
DEFAULT_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Initialize Tavily client for deep research
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# ---------------------- Forms ----------------------
class URLForm(FlaskForm):
    url = StringField('URL', validators=[DataRequired()])
    question = StringField('Question', validators=[DataRequired()])
    api_key = StringField('OpenAI API Key (Optional)', validators=[Optional()])
    tavily_key = StringField('Tavily API Key (Optional)', validators=[Optional()])
    submit = SubmitField('Submit')

class ResearchForm(FlaskForm):
    question = StringField('Research Question', validators=[DataRequired()])
    tavily_key = StringField('Tavily API Key (Optional)', validators=[Optional()])
    submit = SubmitField('Start Research')

# ---------------------- Deep Research Functions ----------------------
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search using Tavily"""
    try:
        search_docs = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        return search_docs
    except Exception as e:
        return {"error": f"Search failed: {e}"}

def create_deep_research_agent():
    """Create and configure the deep research agent"""
    
    sub_research_prompt = """You are a dedicated researcher. Your job is to conduct research based on the users questions.

Conduct thorough research and then reply to the user with a detailed answer to their question

only your FINAL answer will be passed on to the user. They will have NO knowledge of anything except your final message, so your final report should be your final message!"""

    research_sub_agent = {
        "name": "research-agent",
        "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
        "prompt": sub_research_prompt,
        "tools": ["internet_search"],
    }

    sub_critique_prompt = """You are a dedicated editor. You are being tasked to critique a report.

You can find the report at `final_report.md`.

You can find the question/topic for this report at `question.txt`.

The user may ask for specific areas to critique the report in. Respond to the user with a detailed critique of the report. Things that could be improved.

You can use the search tool to search for information, if that will help you critique the report

Do not write to the `final_report.md` yourself.

Things to check:
- Check that each section is appropriately named
- Check that the report is written as you would find in an essay or a textbook - it should be text heavy, do not let it just be a list of bullet points!
- Check that the report is comprehensive. If any paragraphs or sections are short, or missing important details, point it out.
- Check that the article covers key areas of the industry, ensures overall understanding, and does not omit important parts.
- Check that the article deeply analyzes causes, impacts, and trends, providing valuable insights
- Check that the article closely follows the research topic and directly answers questions
- Check that the article has a clear structure, fluent language, and is easy to understand.
"""

    critique_sub_agent = {
        "name": "critique-agent",
        "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
        "prompt": sub_critique_prompt,
    }

    research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

The first thing you should do is to write the original user question to `question.txt` so you have a record of it.

Use the research-agent to conduct deep research. It will respond to your questions/topics with a detailed answer.

When you think you enough information to write a final report, write it to `final_report.md`

You can call the critique-agent to get a critique of the final report. After that (if needed) you can do more research and edit the `final_report.md`
You can do this however many times you want until are you satisfied with the result.

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

Here are instructions for writing the final report:

<report_instructions>
CRITICAL: Make sure the answer is written in the same language as the human messages! If you make a todo plan - you should note in the plan what language the report should be in so you dont forget!
Note: the language the report should be in is the language the QUESTION is in, not the language/country that the question is ABOUT.

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
</report_instructions>

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

    # Create the agent
    agent = create_deep_agent(
        [internet_search],
        research_instructions,
        subagents=[critique_sub_agent, research_sub_agent],
    ).with_config({"recursion_limit": 1000})
    
    return agent

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

def answer_question(df, question, max_len=5000, max_tokens=700):
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
    url_form = URLForm()
    research_form = ResearchForm()

    if url_form.validate_on_submit() and url_form.submit.data:
        try:
            url, question = url_form.url.data, url_form.question.data
            api_key = url_form.api_key.data or DEFAULT_API_KEY

            if not api_key:
                flash("OpenAI API key is required for URL analysis.", "error")
                return render_template('index.html', url_form=url_form, research_form=research_form)

            session['api_key'] = api_key

            local_domain = urlparse(url).netloc
            subpages = get_domain_hyperlinks(local_domain, url)
            crawl_result = crawl([url] + subpages)

            if "error" in crawl_result.lower():
                flash(crawl_result, "error")
                return render_template('index.html', url_form=url_form, research_form=research_form)

            process_result = process_crawled_data(local_domain)
            if "error" in process_result.lower():
                flash(process_result, "error")
                return render_template('index.html', url_form=url_form, research_form=research_form)

            df = pd.read_csv('processed/embeddings.csv')
            df['embeddings'] = df['embeddings'].apply(eval)
            answer = answer_question(df, question)

            return render_template('results.html', answer=answer, question=question, source_type="url", source=url)
        except Exception as e:
            flash(f"Error: {e}", "error")

    if research_form.validate_on_submit() and research_form.submit.data:
        try:
            question = research_form.question.data
            tavily_key = research_form.tavily_key.data
            
            # Set Tavily API key if provided
            if tavily_key:
                global tavily_client
                tavily_client = TavilyClient(api_key=tavily_key)
            elif not os.environ.get("TAVILY_API_KEY"):
                flash("Tavily API key is required for deep research.", "error")
                return render_template('index.html', url_form=url_form, research_form=research_form)

            # Create and run the deep research agent
            agent = create_deep_research_agent()
            
            # Run the research
            flash("Starting deep research... This may take a few minutes.", "info")
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            
            # Extract the final report
            if "final_report.md" in result.get("files", {}):
                answer = result["files"]["final_report.md"]
            else:
                # Fallback to the last message
                messages = result.get("messages", [])
                answer = messages[-1]["content"] if messages else "No research report generated."

            return render_template('results.html', answer=answer, question=question, source_type="research", source="Deep Research")
        except Exception as e:
            flash(f"Research Error: {e}", "error")

    return render_template('index.html', url_form=url_form, research_form=research_form)

# ---------------------- Main ----------------------
if __name__ == '__main__':
    app.run(debug=True)
