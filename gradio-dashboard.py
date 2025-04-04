import pandas as pd
import numpy as np
import os
import re
import psutil
from dotenv import load_dotenv
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# LangChain imports
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Gradio import
import gradio as gr

# Load environment variables
load_dotenv()


def print_memory_usage():
    """Helper function to monitor memory usage"""
    process = psutil.Process()
    print(f"Memory used: {process.memory_info().rss / 1024 ** 2:.2f} MB")


# --- Data Loading ---
print("Loading book data...")
books = pd.read_csv("books_with_emotions.csv")

# Process thumbnails
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


# --- Document Processing ---
def load_and_validate_documents():
    print("\nLoading and validating text documents...")

    # Load documents
    try:
        loader = TextLoader("tagged_description.txt", encoding='utf-8')
        raw_documents = loader.load()
    except UnicodeDecodeError:
        print("UTF-8 failed, trying latin-1 encoding...")
        loader = TextLoader("tagged_description.txt", encoding='latin-1')
        raw_documents = loader.load()

    # Validate documents
    valid_docs = []
    invalid_count = 0
    isbn_pattern = re.compile(r'\b\d{13}\b')

    for doc in raw_documents:
        content = doc.page_content.strip()
        if isbn_pattern.search(content):
            valid_docs.append(doc)
        else:
            invalid_count += 1
            if invalid_count <= 5:  # Show first 5 errors
                print(f"Invalid document (missing ISBN): {content[:100]}...")

    print(f"\nDocument validation results:")
    print(f"- Total documents: {len(raw_documents)}")
    print(f"- Valid documents: {len(valid_docs)}")
    print(f"- Invalid documents: {invalid_count}")

    if not valid_docs:
        print("Fatal error: No valid documents found!")
        exit(1)

    return valid_docs


valid_documents = load_and_validate_documents()

# --- Text Splitting ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len
)

print("\nSplitting documents into chunks...")
documents = text_splitter.split_documents(valid_documents)
print(f"Created {len(documents)} chunks")
print_memory_usage()

# --- Embeddings Setup ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'batch_size': 8,
        'normalize_embeddings': True
    }
)


# --- Chroma DB Setup ---
def process_in_batches(documents, batch_size=100):
    """Process documents in batches to manage memory"""
    for i in tqdm(range(0, len(documents), batch_size), desc="Total progress"):
        batch = documents[i:i + batch_size]
        yield batch


try:
    print("\nCreating Chroma database...")

    # Initialize with first batch
    first_batch = next(process_in_batches(documents))
    db_books = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Add remaining batches
    for batch in process_in_batches(documents[100:]):
        with tqdm(batch, desc="Processing batch", leave=False) as pbar:
            db_books.add_documents(pbar)
            print_memory_usage()

    print("Database creation completed successfully!")

except Exception as e:
    print(f"\nError: {str(e)}")
    print("Solutions:")
    print("1. Reduce chunk_size in text_splitter")
    print("2. Use 'persist_directory' to resume later")
    print("3. Try GPU acceleration if available")
    exit(1)


# --- Recommendation Functions ---
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract ISBNs using regex pattern
    books_list = []
    isbn_pattern = re.compile(r'\b\d{13}\b')
    for rec in recs:
        content = rec.page_content.strip()
        match = isbn_pattern.search(content)
        if match:
            try:
                books_list.append(int(match.group()))
            except ValueError:
                print(f"Invalid ISBN format: {match.group()} in {content[:50]}...")
        else:
            print(f"No ISBN found in: {content[:50]}...")

    if not books_list:
        print(f"No valid books found for query: '{query}'")
        return pd.DataFrame()

    book_recs = books[books["isbn13"].isin(books_list)]
    if book_recs.empty:
        return pd.DataFrame()

    book_recs = book_recs.head(initial_top_k)

    # Category filtering
    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Tone-based sorting
    tone_mapping = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }
    if tone in tone_mapping:
        book_recs.sort_values(by=tone_mapping[tone], ascending=False, inplace=True)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    # Handle empty query
    if not query.strip():
        return [("cover-not-found.jpg", "Please enter a book description")]

    # Get recommendations
    recommendations = retrieve_semantic_recommendations(query, category, tone)

    # Handle no results
    if recommendations.empty:
        return [("cover-not-found.jpg", "No recommendations found. Try different search terms.")]

    results = []
    for _, row in recommendations.iterrows():
        # Truncate description
        truncated = ' '.join(row["description"].split()[:30]) + '...'

        # Format authors
        authors = row["authors"].split(';')
        if len(authors) > 1:
            authors_str = ", ".join(authors[:-1]) + f" and {authors[-1]}"
        else:
            authors_str = authors[0]

        results.append((
            row["large_thumbnail"],
            f"{row['title']} by {authors_str}: {truncated}"
        ))

    return results


# --- UI Configuration ---
categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 AI-Powered Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe your ideal book:",
            placeholder="e.g., A mystery novel set in Victorian London...",
            max_lines=3
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Filter by Category",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Emotional Tone Preference",
            value="All"
        )

    submit_btn = gr.Button("Get Recommendations", variant="primary")

    gr.Markdown("## Recommended Reads")
    gallery = gr.Gallery(
        label="Top Picks",
        columns=4,
        rows=2,
        object_fit="cover",
        height="auto"
    )

    submit_btn.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=gallery
    )

if __name__ == "__main__":
    print("\nLaunching Gradio interface...")
    try:
        dashboard.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True
        )
    except OSError as e:
        print(f"Port error: {str(e)}")
        print("Trying alternative port 7861...")
        dashboard.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            inbrowser=True
        )