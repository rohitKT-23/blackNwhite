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
import numpy as np
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Initialize FastAPI
app = FastAPI()

# Load spaCy for text preprocessing
try:
    nlp = spacy.load('en_core_web_sm')
except:
    # If the model isn't available, download it
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Load multiple fake news detection models for ensemble approach
models = {
    "bert_fake_news": pipeline("text-classification", model="omykhailiv/bert-fake-news-recognition"),
    # Add a second model if available - comment out if not using
    # "roberta_fake_news": pipeline("text-classification", model="rajveersinghcse/roberta-fake-news"),
}

# Serper API Configuration
SERPER_API_KEY = "6ddf4c6afe6fd2913800f3a225546a54297e1006"
SERPER_API_URL = "https://google.serper.dev/search"

# Define request body schema
class NewsInput(BaseModel):
    text: str
    url: str = None
    publication_date: str = None

class FeedbackModel(BaseModel):
    news_text: str
    correct_label: str
    user_comment: str = None

# Enhanced text preprocessing function
def preprocess_text(text):
    """
    Preprocess text according to the model card recommendations with improvements
    """
    # Convert text to lowercase
    text = str(text).lower()
    
    # Store original length for comparison
    original_length = len(text)
    
    # Remove HTML tags and their contents
    text = re.sub('<.?>+\w+<.?>', '', text)
    
    # Remove punctuation but keep sentence boundaries for claim extraction
    text_for_claims = re.sub('[%s]' % re.escape(string.punctuation.replace('.', '')), '', text)
    
    # Remove punctuation for model analysis
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove words containing alphanumeric characters followed by digits
    text = re.sub('\w*\d\w*', '', text)
    
    # Remove newline characters
    text = re.sub('\n', ' ', text)
    
    # Replace multiple whitespace characters with a single space
    text = re.sub('\\s+', ' ', text)
    
    # Lemmatize words
    doc = nlp(text)
    words = [token.lemma_ for token in doc]
    
    # Remove stopwords but preserve key negation words
    negation_words = ['no', 'not', 'never', 'none']
    filtered_words = [word for word in words if word not in stopwords.words('english') or word in negation_words]
    
    # Join the words back into a string
    processed_text = ' '.join(filtered_words)
    
    # If preprocessing removed too much content, use more conservative approach
    if len(processed_text) < 0.3 * original_length:
        # Less aggressive preprocessing
        doc = nlp(str(text).lower())
        processed_text = ' '.join([token.lemma_ for token in doc if not token.is_punct])
    
    return processed_text, text_for_claims

# Health check endpoint
@app.get("/ping")
def ping():
    return {"message": "Improved fake news detection service is running!"}

# Claim extraction function
def extract_claims(text, max_claims=3):
    """Extract main claims from news text"""
    sentences = re.split(r'[.!?]', text)
    claims = []
    
    for sentence in sentences:
        clean_sentence = sentence.strip()
        if len(clean_sentence) > 15:  # Only consider reasonable length sentences
            # Process with spaCy to identify sentences with entities or key information
            doc = nlp(clean_sentence)
            # Prioritize sentences with named entities or that appear to be claims
            if any(ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "EVENT"] for ent in doc.ents):
                claims.append(clean_sentence)
    
    # If no good claims found, use first few sentences
    if not claims and sentences:
        claims = [s.strip() for s in sentences[:max_claims] if len(s.strip()) > 15]
    
    # Limit to max_claims
    return claims[:max_claims]

# Enhanced prediction function using model ensemble
def predict_news_label(text):
    # Preprocess the text according to model recommendations
    processed_text, _ = preprocess_text(text)
    
    # Get model predictions from all models
    results = {}
    for model_name, model_pipe in models.items():
        result = model_pipe(processed_text)[0]
        
        # Normalize the labels
        if result['label'] == 'LABEL_0':
            label = "Fake News"
        else:
            label = "Real News"
            
        results[model_name] = {
            "label": label,
            "confidence": result['score']
        }
    
    # If using only one model
    if len(results) == 1:
        return next(iter(results.values()))
    
    # If using multiple models, use ensemble approach
    fake_votes = sum(1 for r in results.values() if r["label"] == "Fake News")
    real_votes = sum(1 for r in results.values() if r["label"] == "Real News")
    
    # Calculate weighted average confidence
    if fake_votes > real_votes:
        confidence = np.mean([r["confidence"] for r in results.values() if r["label"] == "Fake News"])
        return {"label": "Fake News", "confidence": confidence}
    else:
        confidence = np.mean([r["confidence"] for r in results.values() if r["label"] == "Real News"])
        return {"label": "Real News", "confidence": confidence}

# Extract keywords from text for SERP search
def extract_keywords(text, max_words=10):
    doc = nlp(text[:800])  # Limit to first 800 chars for efficiency
    
    # Extract named entities as highest priority keywords
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "EVENT"]]
    
    # Extract important nouns and adjectives
    keywords = [token.text for token in doc if (token.pos_ in ["NOUN", "PROPN"] or 
                                               (token.pos_ == "ADJ" and token.is_stop == False)) and 
                                                len(token.text) > 3]
    
    # Combine entities and keywords, prioritizing entities
    combined = entities + [k for k in keywords if k not in entities]
    
    # Return unique keywords up to max_words
    return " ".join(list(dict.fromkeys(combined))[:max_words])

# Improved credibility checking function
def check_credibility(claims, full_text):
    if not SERPER_API_KEY:
        return {"error": "Serper API key not configured", "credibility_score": 0.5}
    
    # List of credible news domains
    credible_domains = [
        "apnews.com", "reuters.com", "bloomberg.com", "nytimes.com", "wsj.com", 
        "washingtonpost.com", "bbc.com", "economist.com", "ft.com",
        "theguardian.com", "cnn.com", "cbsnews.com", "nbcnews.com", "abcnews.go.com",
        "time.com", "theatlantic.com", "newyorker.com", "forbes.com", "fortune.com",
        "usatoday.com", "latimes.com", "chicagotribune.com", "bostonglobe.com",
        "politico.com", "vox.com", "slate.com", "thehill.com", "npr.org", "pbs.org",
        "scmp.com", "thehindu.com", "hindustantimes.com", "indianexpress.com",
        "straitstimes.com", "channelnewsasia.com", "japantimes.co.jp", "asahi.com",
        "buzzfeednews.com", "vice.com", "huffpost.com", "businessinsider.com",
        "thedailybeast.com", "salon.com", "motherjones.com", "theverge.com",
        "wired.com", "arstechnica.com", "gizmodo.com", "engadget.com",
        "mainichi.jp", "yomiuri.co.jp", "koreaherald.com", "koreatimes.co.kr",
        "bangkokpost.com", "thejakartapost.com", "nst.com.my", "abs-cbn.com",
        "inquirer.net", "dawn.com", "thenews.com.pk"
    ]
    
    # Function to check if a domain is in our credible list
    def is_credible_domain(domain):
        return any(credible_domain in domain for credible_domain in credible_domains)
    
    # Process each claim separately and aggregate results
    all_results = []
    highest_credibility = 0
    best_search_query = ""
    all_matched_sources = []
    total_sources = 0
    
    # If no specific claims extracted, use keywords from the full text
    if not claims:
        keywords = extract_keywords(full_text)
        claims = [keywords]
    
    # Query each claim
    for claim in claims:
        payload = json.dumps({"q": claim})
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(SERPER_API_URL, headers=headers, data=payload)
            results = response.json()
            
            domain_matches = []
            
            # Parse the response and check for credible domains
            for result in results.get("organic", []):
                link = result.get("link", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                domain = re.search(r'https?://(?:www\.)?([^/]+)', link)
                if domain:
                    domain = domain.group(1)
                    if is_credible_domain(domain):
                        # Add to matches if the domain is credible
                        domain_matches.append({
                            "domain": domain,
                            "title": title,
                            "snippet": snippet,
                            "weight": 1  # All credible sources have the same weight
                        })
                        all_matched_sources.append(domain)
            
            # Calculate credibility score for this claim based on credible sources found
            claim_credibility = len(domain_matches) / max(1, len(results.get("organic", [])))  # Avoid division by zero
            
            # Update best claim if this one has higher credibility
            if claim_credibility > highest_credibility:
                highest_credibility = claim_credibility
                best_search_query = claim
            
            all_results.append({
                "claim": claim,
                "credibility": claim_credibility,
                "matched_domains": [match["domain"] for match in domain_matches],
                "total_results": len(results.get("organic", [])),
                "credible_matches": len(domain_matches)
            })
            
            total_sources += len(results.get("organic", []))
            
        except Exception as e:
            all_results.append({
                "claim": claim,
                "error": str(e),
                "credibility": 0
            })
    
    # Calculate overall credibility
    overall_credibility = highest_credibility
    
    # Assess overall credibility level
    if not all_results:
        source_credibility = "No search results found"
    else:
        # Determine source credibility based on number of credible sources found
        credible_count = sum(result.get("credible_matches", 0) for result in all_results)
        
        if overall_credibility >= 0.5 or credible_count > 0:
            source_credibility = "High"
        elif overall_credibility >= 0.25:
            source_credibility = "Medium"
        elif overall_credibility > 0:
            source_credibility = "Low"
        else:
            source_credibility = "Very Low"
    
    # Remove duplicates from matched sources
    unique_sources = list(dict.fromkeys(all_matched_sources))
    
    return {
        "source_credibility": source_credibility,
        "credibility_score": overall_credibility,
        "matched_sources": unique_sources,
        "total_results": total_sources,
        "search_query_used": best_search_query,
        "claim_results": all_results
    }

# Improved combined assessment function
def get_combined_assessment(model_result, serp_result):
    """More sophisticated logic for combining model and source assessments"""
    
    # Get base values
    model_label = model_result["label"]
    model_confidence = model_result["confidence"]
    source_credibility = serp_result.get("source_credibility", "No data")
    credibility_score = serp_result.get("credibility_score", 0)
    matched_sources = serp_result.get("matched_sources", [])
    credible_matches = sum(result.get("credible_matches", 0) for result in serp_result.get("claim_results", []))
    
    # Initialize assessment and confidence
    combined_assessment = "Uncertain"
    
    # Calculate base combined confidence with higher weight for source credibility
    # when sources are found (60% source weight, 40% model weight)
    if matched_sources:
        combined_confidence = (0.6 * credibility_score) + (0.4 * (model_confidence if model_label == "Real News" else 1 - model_confidence))
    else:
        # If no sources found, rely more on model (80% model weight, 20% base uncertainty)
        combined_confidence = (0.8 * (model_confidence if model_label == "Real News" else 1 - model_confidence)) + 0.1

    # CASE 1: Credible sources found
    if credible_matches > 0 or len(matched_sources) >= 3 or credibility_score >= 0.5:
        if model_label == "Real News":
            combined_assessment = "Very Likely Real (Verified by Credible Sources)"
            combined_confidence = max(0.8, (0.7 * credibility_score) + (0.3 * model_confidence))
        else:
            # Model says fake but credible sources found - trust the sources more
            if model_confidence > 0.95:
                combined_assessment = "Uncertain (Contradictory Signals)"
                combined_confidence = 0.5
            else:
                combined_assessment = "Likely Real (Despite Model Classification)"
                combined_confidence = (0.8 * credibility_score) + (0.2 * (1 - model_confidence))
    
    # CASE 2: Some credible sources (medium tier or fewer sources)
    elif len(matched_sources) > 0 or credibility_score >= 0.25:
        if model_label == "Real News":
            combined_assessment = "Likely Real (Some Source Verification)"
            combined_confidence = (0.6 * model_confidence) + (0.4 * credibility_score)
        else:
            if model_confidence > 0.9:
                combined_assessment = "Uncertain (Mixed Signals)"
                combined_confidence = 0.5
            else:
                combined_assessment = "Possibly Real (Limited Verification)"
                combined_confidence = (0.5 * credibility_score) + (0.3 * (1 - model_confidence)) + 0.2
    
    # CASE 3: No or very low credibility sources AND model says Fake with high confidence
    elif model_label == "Fake News" and model_confidence >= 0.9:
        combined_assessment = "Very Likely Fake (Model Confidence High, No Source Verification)"
        combined_confidence = 0.9 * model_confidence
    
    # CASE 4: No or very low credibility sources AND model says Real with high confidence
    elif model_label == "Real News" and model_confidence >= 0.9:
        combined_assessment = "Uncertain (No Source Verification)"
        combined_confidence = 0.6 * model_confidence
    
    # CASE 5: Model has medium confidence and no sources
    elif 0.7 <= model_confidence < 0.9:
        combined_assessment = f"Possibly {model_label} (Limited Confidence)"
        combined_confidence = 0.7 * model_confidence if model_label == "Real News" else 0.7 * (1 - model_confidence)
    
    # CASE 6: Default fallback
    else:
        combined_assessment = "Uncertain (Insufficient Information)"
        combined_confidence = 0.5

    # New rule to set uncertain and below cases to "Fake"
    if "Uncertain" in combined_assessment or "Possibly" in combined_assessment:
        combined_assessment = "Fake"
        combined_confidence = 0.5  # Arbitrary low confidence for fake
    
    return combined_assessment, combined_confidence


# Improved prediction endpoint
@app.post("/predict-text")
def predict_news(news: NewsInput):
    # Preprocess text
    processed_text, text_for_claims = preprocess_text(news.text)
    
    # Extract claims for verification
    claims = extract_claims(text_for_claims)
    
    # Get model prediction
    model_result = predict_news_label(processed_text)
    
    # Check credibility through Serper
    serp_result = check_credibility(claims, news.text)
    
    # Get combined assessment
    combined_assessment, combined_confidence = get_combined_assessment(model_result, serp_result)
    
    # Ensure only "Real" or "Fake" is returned in the response
    final_assessment = "Real" if "Real" in combined_assessment else "Fake"
    
    # Return detailed response
    return {
        "combined_assessment": final_assessment,
        "combined_confidence": combined_confidence,
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
    uvicorn.run(app, host = "0.0.0.0",port=8001)
