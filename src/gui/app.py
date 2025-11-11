# src/gui/app.py
import streamlit as st
import subprocess
import sys
import os
import time
import pandas as pd
from pathlib import Path

# Make sure project root is on sys.path so "src" imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Try to import pipeline functions if available (wrappers). If some are missing,
# we'll fallback to running scripts with subprocess.
try:
    from src.data_collection.flipkart_scraper import scrape_flipkart_reviews  # optional
except Exception:
    scrape_flipkart_reviews = None

try:
    from src.preprocessing.clean_translate import preprocess_reviews  # optional
except Exception:
    preprocess_reviews = None

try:
    from src.analysis.sentiment_analysis import analyze_sentiment  # optional
except Exception:
    analyze_sentiment = None

try:
    from src.analysis.vector_semantics import compute_word_similarity  # optional
except Exception:
    compute_word_similarity = None

try:
    from src.analysis.question_answering import answer_question  # optional
except Exception:
    answer_question = None

try:
    from src.summarization.review_summarization import summarize_reviews as summarize_reviews_main
except Exception:
    summarize_reviews_main = None

# Paths (relative to project root)
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"

# Ensure folders exist
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROC, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

st.set_page_config(page_title="Product Review Analyzer", layout="wide")

st.title("Product Review Analyzer (Classical NLP)")
st.markdown("Paste a product review page URL (Flipkart/Amazon), set the number of reviews, run the pipeline, inspect intermediate outputs and ask questions.")

# --- UI: Inputs ---
with st.sidebar:
    st.header("Run Pipeline")
    url = st.text_input("Product reviews page URL (Flipkart/Amazon)", "")
    num_reviews = st.number_input("Number of reviews to scrape", min_value=50, max_value=2000, value=200, step=50)
    run_button = st.button("Start Pipeline")

    st.markdown("---")
    st.header("Inspect intermediate outputs")
    show_raw = st.checkbox("Show raw CSV (data/raw)", value=False)
    show_cleaned = st.checkbox("Show cleaned CSV (data/processed)", value=False)
    show_sentiment = st.checkbox("Show sentiment CSV", value=False)
    show_topics = st.checkbox("Show LSA topics", value=False)
    show_images = st.checkbox("Show result images (results/)", value=False)

# Status area
status_text = st.empty()
log_box = st.empty()

# helper to run a script with subprocess fallback
def run_script(script_rel_path, args=None, status_msg=None):
    """
    Run a script located at script_rel_path (relative to project root).
    If there exists an importable function wrapper (with conventional name),
    prefer calling that. Returns stdout/stderr or function return value.
    """
    if status_msg:
        status_text.info(status_msg)
    script_path = PROJECT_ROOT / script_rel_path
    # If the file doesn't exist, report
    if not script_path.exists():
        status_text.error(f"Script not found: {script_path}")
        return None

    # Try subprocess.run to execute the script
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd += args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = proc.stdout + "\n" + proc.stderr
        log_box.code(out[:10000])
        return out
    except Exception as e:
        status_text.error(f"Failed to run {script_path}: {e}")
        return None

# helper to display CSV if exists
def show_csv(path: Path, max_rows=200):
    if path.exists():
        try:
            df = pd.read_csv(path)
            st.dataframe(df.head(max_rows))
        except Exception as e:
            st.error(f"Could not load CSV {path.name}: {e}")
    else:
        st.info(f"No file found at {path}")

# helper to show images in results folder
def show_images_from_results():
    img_files = list(RESULTS.glob("*.png")) + list(RESULTS.glob("*.jpg"))
    if not img_files:
        st.info("No images found in results/ yet.")
        return
    cols = st.columns(2)
    for i, img in enumerate(img_files):
        with cols[i % 2]:
            st.image(str(img), caption=img.name, width="content")

# --- Pipeline steps (best-effort calls) ---
def step_scrape(url, count):
    status_text.info("Step 1/6: Scraping reviews...")
    # Preferred: call function if available
    if scrape_flipkart_reviews:
        # try to call if signature supports args (defensive)
        try:
            # assume function returns path or list
            result = scrape_flipkart_reviews(url, max_pages=10)  # if signature differs, fallback
            status_text.success("Scraping finished (via function).")
            return result
        except Exception:
            pass
    # Fallback: run script
    # we expect a script at src/data_collection/flipkart_scraper.py that writes to data/raw/flipkart_1000_reviews.csv
    out = run_script("src/data_collection/flipkart_scraper.py", args=None, status_msg="Scraping with script...")
    return out

def step_preprocess():
    status_text.info("Step 2/6: Cleaning & Translation...")
    if preprocess_reviews:
        try:
            result = preprocess_reviews()
            status_text.success("Preprocessing finished (via function).")
            return result
        except Exception:
            pass
    out = run_script("src/preprocessing/clean_translate.py", status_msg="Running preprocess script...")
    return out

def step_pos_ner():
    status_text.info("Step 3/6: POS tagging & NER...")
    out = run_script("src/analysis/pos_ner_analysis.py", status_msg="Running POS/NER analysis...")
    return out

def step_sentiment():
    status_text.info("Step 4/6: Sentiment Analysis...")
    if analyze_sentiment:
        try:
            df = analyze_sentiment()
            status_text.success("Sentiment analysis finished (via function).")
            return df
        except Exception:
            pass
    out = run_script("src/analysis/sentiment_analysis.py", status_msg="Running sentiment script...")
    return out

def step_vector_semantics():
    status_text.info("Step 5/6: Vector semantics & similarity measures...")
    if compute_word_similarity:
        try:
            res = compute_word_similarity()
            status_text.success("Vector semantics finished (via function).")
            return res
        except Exception:
            pass
    out = run_script("src/analysis/vector_semantics.py", status_msg="Running vector semantics script...")
    return out

def step_summarize():
    status_text.info("Step 6/6: Summarization & QA preparation...")
    if summarize_reviews_main:
        try:
            res = summarize_reviews_main()
            status_text.success("Summarization finished (via function).")
            return res
        except Exception:
            pass
    out = run_script("src/summarization/review_summarization.py", status_msg="Running summarization script...")
    return out

# Execute pipeline button
if run_button:
    # Basic input validation
    if not url.strip():
        st.sidebar.error("Please paste a valid product review URL before starting.")
    else:
        st.sidebar.success("Pipeline started. Watch status and logs here.")
        # Step 0: optionally pass url/num_reviews to scraper via environment variables or script args
        # Run steps sequentially and show current step
        try:
            # Scrape
            step_scrape(url, num_reviews)
            time.sleep(0.3)

            # Preprocess
            step_preprocess()
            time.sleep(0.3)

            # POS + NER
            step_pos_ner()
            time.sleep(0.3)

            # Sentiment
            step_sentiment()
            time.sleep(0.3)

            # Vector semantics
            step_vector_semantics()
            time.sleep(0.3)

            # Summarization
            step_summarize()
            time.sleep(0.3)

            status_text.success("Pipeline completed successfully!")
        except Exception as e:
            status_text.error(f"Pipeline failed: {e}")

# --- Show intermediate outputs section (optional) ---
st.markdown("---")
st.header("Inspect Intermediate Outputs")

col1, col2 = st.columns(2)
with col1:
    if show_raw:
        st.subheader("Raw CSV (data/raw)")
        # try common filenames
        candidates = list(DATA_RAW.glob("*.csv"))
        if candidates:
            for p in candidates:
                st.write(f"**{p.name}**")
                show_csv(p)
        else:
            st.info("No CSV files found in data/raw/")

    if show_cleaned:
        st.subheader("Cleaned CSV (data/processed/cleaned_reviews.csv)")
        show_csv(DATA_PROC / "cleaned_reviews.csv")

    if show_sentiment:
        st.subheader("Sentiment results (data/processed/sentiment_results.csv)")
        show_csv(DATA_PROC / "sentiment_results.csv")

with col2:
    if show_topics:
        st.subheader("LSA Topics (data/processed/lsa_topics.csv)")
        show_csv(DATA_PROC / "lsa_topics.csv")

    if show_images:
        st.subheader("Result Images (results/)")
        show_images_from_results()

# --- Final interactive QA after pipeline ---

st.markdown("---")
st.header("Ask a question about the product")

if "pipeline_done" not in st.session_state:
    # set flag true if sentiment file exists as proxy that pipeline has been run
    st.session_state.pipeline_done = (DATA_PROC / "sentiment_results.csv").exists()

user_q = st.text_input("Enter your question (e.g., 'How is the battery life?')")

if st.button("Get Answer"):
    if not user_q.strip():
        st.error("Please enter a non-empty question.")
    else:
        with st.spinner("🔍 Finding most relevant reviews and generating answer..."):
            try:
                # If a proper function is available
                if answer_question:
                    ans = answer_question(user_q)
                    if isinstance(ans, str):
                        st.success(ans.strip())
                    elif ans is not None:
                        st.write(ans)
                    else:
                        st.warning("No answer returned. Check your QA logic.")
                else:
                    # fallback: run script and capture stdout
                    out = run_script("src/analysis/question_answering.py")
                    if out:
                        st.text_area("Answer Output", value=out.strip(), height=250)
                    else:
                        st.warning("No output captured from QA script.")
            except Exception as e:
                st.error(f"Error while answering: {e}")

st.markdown("---")
st.caption("Product Review Analyzer built by Anuj Chandak.")
