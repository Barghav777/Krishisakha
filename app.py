import os
import time
import traceback
from typing import List
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from input_processing import process_text_input, process_voice_input, process_image_input
from rag_pipeline import get_models, get_krishisakha_response

app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app, origins="*")

MODELS = get_models(verbose=True)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def style():
    return send_from_directory(".", "style.css")

@app.route("/script.js")
def script():
    return send_from_directory(".", "script.js")

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/process_input", methods=["POST"])
def handle_input():
    req_start = time.perf_counter()
    image_paths: List[str] = []
    voice_path: str = ""
    originals = {"text": ""}
    detected_language = "en"
    translated_parts: List[str] = []
    use_disease_prediction = False
    disease_image_path = None

    try:
        user_text = request.form.get("text", "").strip()
        if user_text:
            try:
                _, _, lang_code = process_text_input(user_text)
                if lang_code and lang_code != "en":
                    detected_language = lang_code
            except Exception:
                pass
        
        image_files = request.files.getlist("image") or []
        for i, f in enumerate(image_files):
            if not f or f.filename == "":
                continue
            save_path = os.path.join("./", f"upload_{int(time.time()*1000)}_{i}_{f.filename}")
            f.save(save_path)
            image_paths.append(save_path)

            if user_text:
                disease_keywords = ['disease', 'pest', 'sick', 'problem', 'infection', 'spots', 'yellowing', 'wilting', 'dying', 'damaged', 'leaf', 'plant', 'crop', 'fungus', 'virus', 'bacteria', 'insect', 'bug', 'mold', 'rot', 'blight', 'rust', 'scab', 'what is this', 'identify', 'diagnosis', 'treatment', 'cure', 'medicine', 'बीमारी', 'कीट', 'पौधा', 'पत्ती', 'रोग', 'समस्या', 'संक्रमण', 'कवक', 'वायरस', 'बैक्टीरिया', 'कीड़े', 'फसल', 'सड़न', 'मुरझाना', 'पीलापन', 'दाग', 'धब्बा', 'छत्ता', 'फफूंद', 'इलाज', 'दवा', 'कीटनाशक', 'क्या', 'कौन', 'कैसे', 'पहचान', 'निदान', 'खराब', 'मरा', 'सूखा', 'इस पत्ती पर', 'इस पौधे में', 'यह क्या है', 'यह कौन सा', 'રોગ', 'કીડા', 'છોડ', 'પાન', 'સમસ્યા', 'है', 'हैं', 'का', 'की', 'के', 'में', 'पर', 'से']
                user_text_lower = user_text.lower()
                is_disease_query = any(keyword.lower() in user_text_lower for keyword in disease_keywords)
                question_patterns = ['क्या', 'कौन', 'कैसे', 'क्यों', 'कहाँ', 'कब', 'what', 'how', 'why', 'which']
                has_question = any(pattern.lower() in user_text_lower for pattern in question_patterns)
                plant_words = ['पत्ती', 'पौधा', 'फसल', 'leaf', 'plant', 'crop', 'tree', 'पेड़']
                has_plant_reference = any(word.lower() in user_text_lower for word in plant_words)
                is_disease_query = is_disease_query or (has_question and has_plant_reference)
                
                if is_disease_query:
                    use_disease_prediction = True
                    disease_image_path = save_path
                    originals["text"] = (originals["text"] + "\n" + user_text).strip()
                    translated_parts.append(user_text)
                    break
                else:
                    try:
                        trans_text, orig_text, lang_code = process_image_input(save_path)
                        if lang_code and lang_code != "en": 
                            detected_language = lang_code
                    except Exception:
                        trans_text, orig_text = ("", "")
                    if orig_text:
                        originals["text"] = (originals["text"] + "\n" + user_text + "\n" + orig_text).strip()
                    if trans_text and trans_text.strip():
                        translated_parts.append(user_text + "\n" + trans_text.strip())
            else:
                use_disease_prediction = True
                disease_image_path = save_path
                originals["text"] = "What disease or problem does this plant have?"
                translated_parts.append("What disease or problem does this plant have?")
                break

        if "voice" in request.files and request.files["voice"].filename != "":
            voice_file = request.files["voice"]
            base_dir = os.path.dirname(os.path.abspath(__file__))
            voice_path = os.path.join(base_dir, f"temp_recording_{int(time.time()*1000)}.webm")
            voice_file.save(voice_path)
            try:
                trans_text, orig_text, lang_code = process_voice_input(voice_path)
                if lang_code and lang_code != "en": 
                    detected_language = lang_code
            except Exception:
                trans_text, orig_text = ("Could not process audio.", "Voice input could not be transcribed.")
            if orig_text:
                originals["text"] = (originals["text"] + "\n" + orig_text).strip()
            if trans_text and trans_text.strip():
                translated_parts.append(trans_text.strip())

        input_text = request.form.get("text", "").strip()
        if input_text and not image_files:
            try:
                trans_text, orig_text, lang_code = process_text_input(input_text)
                if lang_code and lang_code != "en": 
                    detected_language = lang_code
            except Exception:
                trans_text, orig_text = (input_text, input_text)
            if orig_text:
                originals["text"] = (originals["text"] + "\n" + orig_text).strip()
            if trans_text and trans_text.strip():
                translated_parts.append(trans_text.strip())

        fused_translated = "\n".join(translated_parts).strip()
        
        if use_disease_prediction and disease_image_path:
            answer = get_krishisakha_response(input_data=disease_image_path, models=MODELS, input_type="image", extra_text=fused_translated or "", verbose=True, detected_language=detected_language, original_text=originals["text"].strip())
        elif image_paths and not use_disease_prediction:
            answer = get_krishisakha_response(input_data=fused_translated or "", models=MODELS, input_type="text", verbose=True, detected_language=detected_language, original_text=originals["text"].strip())
        else:
            answer = get_krishisakha_response(input_data=fused_translated or "", models=MODELS, input_type="text", verbose=True, detected_language=detected_language, original_text=originals["text"].strip())

        resp = {
            "user_input": {"text": answer.get("original_text", "")},
            "bot_response": {
                "original_response": answer.get("answer_original", "Error: Missing original response."),
                "english_response": answer.get("answer_english", "Error: Missing English response."),
                "detected_lang_code": answer.get("original_language", "en"),
            },
        }
        return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500

    finally:
        for p in image_paths + ([voice_path] if voice_path else []):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)