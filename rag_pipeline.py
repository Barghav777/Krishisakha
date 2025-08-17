#
# FULL REPLACEMENT CODE FOR: rag_pipeline.py
#
import json
import os
import re
import time
from typing import Dict, Any, Tuple, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from torchvision import models, transforms
from transformers import pipeline
from groq import Groq
from dotenv import load_dotenv
import faiss

load_dotenv()

# Get the absolute path of the directory where the current script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuration (Build paths relative to the script's location) ---
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", os.path.join(_SCRIPT_DIR, "krishisakha_knowledge_base.index"))
CHUNKS_JSON_PATH = os.environ.get("CHUNKS_JSON_PATH", os.path.join(_SCRIPT_DIR, "document_chunks.json"))
DISEASE_MODEL_PATH = os.environ.get("DISEASE_MODEL_PATH", os.path.join(_SCRIPT_DIR, "plant_disease_model.pth"))

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
WEB_SEARCH_ENABLED = bool(SERPAPI_KEY)

_MODELS: Optional[Dict[str, Any]] = None

def load_all_models(verbose: bool = True) -> Dict[str, Any]:
    assets: Dict[str, Any] = {}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if verbose: print(f"INFO[load]: Loading models/data from: {_SCRIPT_DIR}")

    assets["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    if not os.path.exists(FAISS_INDEX_PATH): raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")
    if not os.path.exists(CHUNKS_JSON_PATH): raise FileNotFoundError(f"Chunks JSON not found: {CHUNKS_JSON_PATH}")
    assets["faiss_index"] = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        assets["document_chunks"] = json.load(f)

    assets["groq_client"] = Groq(api_key=GROQ_API_KEY)

    num_classes = 38
    disease_model = models.efficientnet_b0(weights=None)
    num_ftrs = disease_model.classifier[1].in_features
    disease_model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    if not os.path.exists(DISEASE_MODEL_PATH):
        disease_model = None
    else:
        state = torch.load(DISEASE_MODEL_PATH, map_location=device)
        disease_model.load_state_dict(state)
        disease_model.to(device)
        disease_model.eval()
        
    class_names = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy', 'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato___healthy']
    assets["disease_model"] = disease_model
    assets["disease_class_names"] = class_names
    return assets

def get_models(verbose: bool = False) -> Dict[str, Any]:
    global _MODELS
    if _MODELS is None:
        _MODELS = load_all_models(verbose=verbose)
    return _MODELS

def classify_intent(query: str, embedding_model: SentenceTransformer) -> str:
    labels = ["Weather and Climate Inquiry", "Agricultural Economics and Market Prices", "Pest and Disease Management", "Soil and Nutrient Management", "Crop Production and Sowing", "Farm Machinery and Technology", "Government Schemes and General Information"]
    try:
        q_emb = embedding_model.encode(query, convert_to_tensor=True)
        l_emb = embedding_model.encode(labels, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, l_emb)
        return labels[int(torch.argmax(scores).item())]
    except Exception as e:
        return "Pest and Disease Management"

def get_weather_forecast(query_or_location: str, default_location: str = "Jammu", target_language: str = "en", groq_client = None) -> str:
    location = None
    match = re.search(r"in ([\w\s,]+?)(?:\s|$|\?|!)", query_or_location, flags=re.IGNORECASE)
    if match: location = match.group(1).strip()
    if not location:
        match = re.search(r"for ([\w\s,]+?)(?:\s|$|\?|!)", query_or_location, flags=re.IGNORECASE)
        if match: location = match.group(1).strip()
    if not location:
        weather_keywords = ['weather', 'forecast', 'temperature', 'rain', 'climate']
        if not any(keyword in query_or_location.lower() for keyword in weather_keywords) and len(query_or_location.strip()) > 0:
            location = query_or_location.strip()
    if location: location = re.sub(r'[^\w\s]', '', location).strip()
    final_location = location if location else default_location
    
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": final_location, "count": 1}, timeout=10).json()
        if not geo.get("results"): 
            english_response = f"Could not find location: {final_location}."
        else:
            lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
            found_location = geo["results"][0].get("name", final_location)
            res = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "daily": "temperature_2m_max,precipitation_sum", "forecast_days": 3}, timeout=10).json()
            daily = res["daily"]
            english_response = f"3-day forecast for {found_location}: Day 1: {daily['temperature_2m_max'][0]}°C, Rain: {daily['precipitation_sum'][0]}mm; Day 2: {daily['temperature_2m_max'][1]}°C, Rain: {daily['precipitation_sum'][1]}mm; Day 3: {daily['temperature_2m_max'][2]}°C, Rain: {daily['precipitation_sum'][2]}mm."
        
        if target_language != "en" and groq_client:
            return translate_english_to_language_with_llama(english_response, target_language, groq_client)
        return english_response
    except Exception as e:
        return f"Could not retrieve weather forecast. Error: {e}"

def search_web_with_serpapi(query: str, num_results: int = 5) -> str:
    if not WEB_SEARCH_ENABLED: return ""
    try:
        price_keywords = ['price', 'rate', 'cost', 'mandi', 'market', 'भाव', 'दाम', 'मंडी', 'बाजार']
        is_price_query = any(keyword.lower() in query.lower() for keyword in price_keywords)
        enhanced_query = f"mandi price rates today {query} site:agmarknet.gov.in OR site:krishijagran.com OR site:agricoop.nic.in" if is_price_query else f"agriculture farming {query} site:agricoop.nic.in OR site:icar.org.in OR site:krishisewa.com"
        params = {'q': enhanced_query, 'api_key': SERPAPI_KEY, 'engine': 'google', 'num': num_results, 'hl': 'en', 'gl': 'in'}
        response = requests.get('https://serpapi.com/search', params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        
        search_content = []
        if 'organic_results' in results:
            for result in results['organic_results'][:num_results]:
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                link = result.get('link', '')
                if title and snippet:
                    search_content.append(f"Title: {title}\nSummary: {snippet}\nSource: {link}\n")
        return "\n".join(search_content) if search_content else ""
    except Exception as e:
        return ""

def predict_disease(image_path: str, models: Dict[str, Any]) -> Tuple[str, float]:
    model = models.get("disease_model")
    if model is None: return "No specific disease detected", 0.0
    class_names = models["disease_class_names"]
    device = next(model.parameters()).device
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, idx = torch.max(probs, 0)
        disease = class_names[idx.item()]
        confidence = float(conf.item())
        return disease, confidence
    except Exception as e:
        return "No specific disease detected", 0.0

def answer_query_with_rag(query: str, models: Dict[str, Any], top_k: int = 3, relevance_threshold: float = 0.5) -> str:
    embedding_model = models["embedding_model"]
    faiss_index = models["faiss_index"]
    chunks = models["document_chunks"]
    groq_client = models["groq_client"]
    
    price_keywords = ['price', 'rate', 'cost', 'mandi', 'market', 'today', 'current', 'latest', '2025', 'भाव', 'दाम', 'मंडी', 'बाजार', 'आज', 'वर्तमान', 'हाल', 'कीमत']
    is_price_query = any(keyword.lower() in query.lower() for keyword in price_keywords)
    if is_price_query: relevance_threshold = 0.7
    
    qv = embedding_model.encode(query).reshape(1, -1).astype(np.float32)
    distances, indices = faiss_index.search(qv, top_k)
    top_similarity = 1.0 / (1.0 + float(distances[0][0]))

    if top_similarity < relevance_threshold:
        web_context = search_web_with_serpapi(query, num_results=3)
        if web_context:
            messages = [{"role": "system", "content": "You are an expert agricultural assistant. Use the web search results to provide a helpful answer in ENGLISH. Provide a concise, practical answer in plain text format without markdown. Keep it brief, max 4-5 sentences. Mention this information is from current web sources."}, {"role": "user", "content": f"WEB SEARCH RESULTS:\n{web_context}\n\nBased on the web search, answer in ENGLISH:\nQUESTION: {query}"}]
        else:
            messages = [{"role": "system", "content": "You are an expert agricultural assistant. Answer from general knowledge in ENGLISH. Provide a concise, practical, plain text answer without markdown. Keep it brief, max 3-4 sentences."}, {"role": "user", "content": f"Answer the following briefly in ENGLISH. End with 'This was based on my general knowledge, ...'.\n\nQUESTION: {query}"}]
    else:
        retrieved = [chunks[i] for i in indices[0]]
        context_str = "\n---\n".join(f"Source: {c.get('source','unknown')}, Page: {c.get('page','?')}\n{c.get('text','')}" for c in retrieved)
        messages = [{"role": "system", "content": "You are an expert agricultural assistant. Answer ONLY from the context in ENGLISH. Provide a concise, practical, plain text answer without markdown, max 4-5 sentences."}, {"role": "user", "content": f"CONTEXT:\n---\n{context_str}\n---\nQUESTION (answer in English): {query}"}]
    
    try:
        chat_completion = groq_client.chat.completions.create(messages=messages, model=GROQ_MODEL, temperature=0.1, max_tokens=400, top_p=0.95)
        response = chat_completion.choices[0].message.content
    except Exception as e:
        response = "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    return clean_markdown_formatting(response)

def clean_markdown_formatting(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def translate_english_to_language_with_llama(text: str, target_lang: str, groq_client) -> str:
    if not text or target_lang == 'en': return text
    language_names = {"hi": "Hindi", "gu": "Gujarati", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "or": "Odia", "as": "Assamese"}
    target_lang_name = language_names.get(target_lang, target_lang)
    
    try:
        translation_prompt = f"""Translate the following agricultural text from English to {target_lang_name}.
Keep the exact same meaning, technical terms, and factual information.
Only provide the translated text, nothing else.

English text to translate:
{text}

{target_lang_name} translation:"""
        messages = [{"role": "system", "content": f"You are a professional translator specializing in agricultural content. Translate accurately from English to {target_lang_name}."}, {"role": "user", "content": translation_prompt}]
        chat_completion = groq_client.chat.completions.create(messages=messages, model=GROQ_MODEL, temperature=0.1, max_tokens=600, top_p=0.9)
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return text

def get_krishisakha_response(input_data: str, models: Dict[str, Any], input_type: str = "text", extra_text: Optional[str] = None, verbose: bool = False, detected_language: str = "en", original_text: str = "") -> Dict[str, str]:
    english_answer = ""
    if input_type == "image":
        disease_name, confidence = predict_disease(input_data, models)
        parts = []
        if extra_text and extra_text.strip():
            parts.append(extra_text.strip())
        if disease_name:
            if confidence > 0.5:
                parts.append(f"Disease hint from image: {disease_name.replace('___', ' ')} (confidence ~{confidence:.2f}).")
            else:
                parts.append("Image shows plant with some symptoms.")
        fused_query = " ".join(parts).strip() if parts else "Provide general plant disease and pest control advice."
        fused_query += " Provide integrated pest management recommendations."
        english_answer = answer_query_with_rag(fused_query, models)
    else:
        intent = classify_intent(input_data, models["embedding_model"])
        if intent == "Weather and Climate Inquiry":
            english_answer = get_weather_forecast(input_data, target_language="en", groq_client=models["groq_client"])
        else:
            english_answer = answer_query_with_rag(input_data, models)

    original_answer = english_answer
    if detected_language != "en":
        try:
            original_answer = translate_english_to_language_with_llama(english_answer, detected_language, models["groq_client"])
        except Exception as e:
            original_answer = english_answer
            
    return {
        "original_language": detected_language,
        "original_text": original_text,
        "answer_original": original_answer,
        "answer_english": english_answer
    }