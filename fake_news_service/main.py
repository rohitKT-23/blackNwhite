from fastapi import FastAPI
from pydantic import BaseModel
import requests
import re
import string
import torch
import multiprocessing
import spacy
from nltk.corpus import stopwords
import json

# Initialize FastAPI
app = FastAPI()

# Load spaCy for text preprocessing
try:
    nlp = spacy.load('en_core_web_sm')
except:
    # If the model isn't available, download it
    import sys
    import spacy.cli
    spacy.cli.download("en_core_web_sm")

    nlp = spacy.load('en_core_web_sm')

# Load Fake News Detection Model using pipeline
from transformers import pipeline

# Use the new model
pipe = pipeline("text-classification", model="omykhailiv/bert-fake-news-recognition")

# Serper API Configuration
SERPER_API_KEY = "6ddf4c6afe6fd2913800f3a225546a54297e1006"
SERPER_API_URL = "https://google.serper.dev/search"

# Define request body schema
class NewsInput(BaseModel):
    text: str

class FeedbackModel(BaseModel):
    news_text: str
    correct_label: str
    user_comment: str = None

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Text preprocessing function based on model card recommendations
def preprocess_text(text):
    """
    Preprocess text according to the model card recommendations
    """
    # Convert text to lowercase
    text = str(text).lower()
    
    # Remove HTML tags and their contents
    text = re.sub('<.?>+\w+<.?>', '', text)
    
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove words containing alphanumeric characters followed by digits
    text = re.sub('\w*\d\w*', '', text)
    
    # Remove newline characters
    text = re.sub('\n', '', text)
    
    # Replace multiple whitespace characters with a single space
    text = re.sub('\\s+', ' ', text)
    
    # Lemmatize words
    doc = nlp(text)
    words = [token.lemma_ for token in doc]
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    
    # Join the words back into a string
    return ' '.join(filtered_words)

# Health check endpoint
@app.get("/ping")
def ping():
    return {"message": "Fake news detection service is running!"}

# Prediction function using the pipeline
def predict_news_label(text):
    # Preprocess the text according to model recommendations
    processed_text = preprocess_text(text)
    
    # Get model prediction using the pipeline
    result = pipe(processed_text)[0]
    
    # Based on model card: 'LABEL_0' means false, 'LABEL_1' means true
    if result['label'] == 'LABEL_0':
        return {"label": "Fake News", "confidence": result['score']}
    else:
        return {"label": "Real News", "confidence": result['score']}

# Extract main claim from text
def extract_main_claims(text):
    sentences = re.split(r'[.!?]', text)
    return sentences[0].strip() if sentences else text[:100]

# Extract keywords from text for SERP search
def extract_keywords(text, max_words=8):
    stop_words = stopwords.words('english')
    words = text.split()[:100]  
    keywords = [w for w in words if len(w) > 4 and w.lower() not in stop_words]
    return " ".join(keywords[:max_words])  

# Check credibility through Serper.dev API
def check_credibility(claim, full_text):
    if not SERPER_API_KEY:
        return {"error": "Serper API key not configured", "credibility_score": 0.5}
    
    # List of credible news domains
    credible_domains = [
        
        # Asia
        "scmp.com", "thehindu.com", "hindustantimes.com", "indianexpress.com",
        "straitstimes.com", "channelnewsasia.com", "japantimes.co.jp", "asahi.com",
        "mainichi.jp", "yomiuri.co.jp", "koreaherald.com", "koreatimes.co.kr",
        "bangkokpost.com", "thejakartapost.com", "nst.com.my", "abs-cbn.com",
        "inquirer.net", "dawn.com", "thenews.com.pk",
        
        # Middle East
        "haaretz.com", "jpost.com", "timesofisrael.com", "thetimes.co.il",
        "arabnews.com", "gulfnews.com", "khaleejtimes.com", "thenational.ae",
        
        # Africa
        "mg.co.za", "news24.com", "iol.co.za", "citizentv.co.ke", "nation.co.ke",
        "standardmedia.co.ke", "monitor.co.ug", "newvision.co.ug", "guardian.ng",
        "punchng.com", "premiumtimesng.com", "allafrica.com", "egyptindependent.com",
        
        # Oceania
        "abc.net.au", "smh.com.au", "theage.com.au", "afr.com", "theaustralian.com.au",
        "nzherald.co.nz", "stuff.co.nz", "rnz.co.nz",
        
        # Latin America
        "clarin.com", "lanacion.com.ar", "folha.uol.com.br", "oglobo.globo.com",
        "eluniversal.com.mx", "excelsior.com.mx", "emol.com", "eltiempo.com",
        "elcomercio.pe", "eluniverso.com", "laprensa.hn", "prensa.com",
        
        # International Organizations
        "un.org", "who.int", "hrw.org", "amnesty.org", "icrc.org"
    ]

    # First try with the claim
    payload = json.dumps({"q": claim})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(SERPER_API_URL, headers=headers, data=payload, timeout=10)

        results = response.json()
        search_query_used = claim

        # If no results, try keywords
        if not results.get("organic") or len(results.get("organic", [])) == 0:
            keywords = extract_keywords(full_text)
            payload = json.dumps({"q": keywords})
            search_query_used = keywords
            response = requests.post(SERPER_API_URL, headers=headers, data=payload, timeout=10)
            results = response.json()
        
        credible_matches = 0
        total_results = len(results.get("organic", []))
        matched_sources = []

        # Parse the response and check for credible domains
        for result in results.get("organic", []):
            link = result.get("link", "")
            domain = re.search(r'https?://(?:www\.)?([^/]+)', link)
            if domain:
                domain = domain.group(1)
                if any(credible_domain in domain for credible_domain in credible_domains):
                    credible_matches += 1
                    matched_sources.append(domain)
        
        credibility_score = credible_matches / total_results if total_results > 0 else 0
        
        return {
            "source_credibility": "High" if credibility_score > 0.3 else "Low",
            "credibility_score": credibility_score,
            "matched_sources": matched_sources,
            "total_results": total_results,
            "search_query_used": search_query_used
        }
    except Exception as e:
        return {"error": str(e), "credibility_score": 0.5, "search_query_used": claim}

# Predict fake/real news with Serper.dev validation
@app.post("/predict-text")
def predict_news(news: NewsInput):
    # Get model prediction
    model_result = predict_news_label(news.text)
    
    # Extract main claim for Serper validation
    main_claim = extract_main_claims(news.text)
    
    # Check credibility through Serper
    serp_result = check_credibility(main_claim, news.text)
    
    # Combined assessment logic
    combined_assessment = "Uncertain"
    combined_confidence = (model_result["confidence"] + serp_result.get("credibility_score", 0.5)) / 2
    
    if serp_result.get("source_credibility") == "No search results found":
        if model_result["confidence"] > 0.95:
            combined_assessment = f"Possibly {model_result['label']} (Needs verification)"
            combined_confidence *= 0.7  
        else:
            combined_assessment = "Uncertain (Insufficient data)"
            combined_confidence = 0.5  
    elif model_result["label"] == "Real News" and model_result["confidence"] > 0.8 and serp_result.get("credibility_score", 0) < 0.3:
        combined_assessment = "Possibly Fake (Contradictory Sources)"
    elif model_result["label"] == "Fake News" and model_result["confidence"] > 0.9 and serp_result.get("credibility_score", 0) < 0.2:
        combined_assessment = "Very Likely Fake"
    elif serp_result.get("credibility_score", 0) > 0.5:
        combined_assessment = "Likely Real (Based on Credible Sources)"
        combined_confidence = (0.7 * serp_result.get("credibility_score", 0.5)) + (0.3 * model_result["confidence"])
    elif model_result["confidence"] > 0.8:
        combined_assessment = model_result["label"]
    elif serp_result.get("credibility_score", 0) < 0.1 and len(serp_result.get("matched_sources", [])) == 0:
        combined_assessment = "Suspicious (No Credible Sources)"
    
    return {
        "model_assessment": {
            "label": model_result["label"],
            "confidence": model_result["confidence"],
            "model_type": "omykhailiv/bert-fake-news-recognition"
        },
        "source_assessment": serp_result,
        "combined_assessment": combined_assessment,
        "combined_confidence": combined_confidence
    }

# Feedback endpoint
@app.post("/feedback")
def record_feedback(feedback: FeedbackModel):
    return {
        "message": "Thank you for your feedback",
        "feedback_recorded": True
    }

# Run locally
if __name__ == "__main__":
    import uvicorn
    # Make sure NLTK stopwords are downloaded
    import nltk
    try:
        stopwords.words('english')
    except:
        nltk.download('stopwords')
    
    # Add freeze_support for Windows multiprocessing
    multiprocessing.freeze_support()
    import uvicorn
    uvicorn.run("main:app", host="54.163.173.200", port=8001, reload=True)