#
# FULL REPLACEMENT CODE FOR: input_processing.py
#
import os
import torch
from PIL import Image
from transformers import pipeline, logging as hf_logging
from langdetect import detect, DetectorFactory
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
hf_logging.set_verbosity_warning()

# Ensure consistent language detection results
DetectorFactory.seed = 0

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

_groq_client = None
if GROQ_API_KEY:
    try:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"WARN[voice]: Failed to initialize Groq client: {e}")

try:
    from pydub import AudioSegment
    from pydub.utils import which
    FFMPEG_PATH = which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    FFPROBE_PATH = which("ffprobe") or r"C:\ffmpeg\bin\ffprobe.exe"
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg = FFMPEG_PATH
    AudioSegment.ffprobe = FFPROBE_PATH
    if not os.path.isfile(FFMPEG_PATH) or not os.path.isfile(FFPROBE_PATH):
        print("WARNING: ffmpeg/ffprobe not found. Voice (webm) conversion may fail.")
except Exception as e:
    print(f"pydub/ffmpeg not available: {e}. Voice (webm) conversion may fail.")

# fasttext has been removed to prevent compilation errors.

language_code_mapping = {
    'as': 'asm_Beng', 'bn': 'ben_Beng', 'en': 'eng_Latn', 'gu': 'guj_Gujr', 'hi': 'hin_Deva',
    'kn': 'kan_Knda', 'ml': 'mal_Mlym', 'mr': 'mar_Deva', 'ne': 'npi_Deva', 'or': 'ory_Orya',
    'pa': 'pan_Guru', 'sa': 'san_Deva', 'sd': 'snd_Arab', 'si': 'sin_Sinh', 'ta': 'tam_Taml',
    'te': 'tel_Telu', 'ur': 'urd_Urdu'
}
MMS_TO_ISO1 = {
    'asm': 'as', 'ben': 'bn', 'eng': 'en', 'guj': 'gu', 'hin': 'hi', 'kan': 'kn', 'mal': 'ml',
    'mar': 'mr', 'npi': 'ne', 'nep': 'ne', 'ory': 'or', 'ori': 'or', 'pan': 'pa', 'san': 'sa',
    'snd': 'sd', 'sin': 'si', 'tam': 'ta', 'tel': 'te', 'urd': 'ur'
}

def normalize_mms_label(label: str):
    if not label: return None
    code = label.lower().split('_')[0]
    return MMS_TO_ISO1.get(code)

print("INFO[input]: Pre-loading processing models...")
_lid_pipeline = pipeline("audio-classification", model="facebook/mms-lid-126", device_map="cpu", model_kwargs={"torch_dtype": torch.float32})
_asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device_map="cpu", model_kwargs={"torch_dtype": torch.float32})
print("INFO[input]: Transformer pipelines loaded.")

_easyocr_reader = None
try:
    import easyocr
    safe_langs = ['en', 'hi']
    extended_langs = ['bn', 'mr', 'ta', 'te', 'gu', 'pa', 'ur', 'kn', 'ml', 'ne', 'or', 'as']
    try:
        _easyocr_reader = easyocr.Reader(safe_langs + extended_langs, gpu=False)
    except Exception as e:
        _easyocr_reader = easyocr.Reader(safe_langs, gpu=False)
except Exception as e:
    easyocr = None
    _easyocr_reader = None
    print(f"EasyOCR not available or failed to load: {e}")

def enhance_audio_quality(audio_path: str):
    try:
        if not audio_path.lower().endswith(('.webm', '.wav', '.mp3', '.m4a')):
            return audio_path
        audio = AudioSegment.from_file(audio_path).normalize()
        if len(audio) > 100:
            audio = audio.high_pass_filter(80)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        if len(audio) > 200:
            audio = audio.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
        audio = audio.strip_silence(silence_len=200, silence_thresh=-45, padding=100)
        import time
        timestamp = int(time.time() * 1000)
        enhanced_path = audio_path.replace('.webm', f'_enhanced_{timestamp}.wav').replace('.wav', f'_enhanced_{timestamp}.wav')
        audio.export(enhanced_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        return enhanced_path
    except Exception as e:
        print(f"WARN[audio]: Audio enhancement failed: {e}, using original")
        return audio_path

def detect_language_text(text: str):
    """
    Detects the language of a given text using the langdetect library.
    """
    if not text or not text.strip():
        return None
    try:
        # The detect function returns the two-letter ISO 639-1 code (e.g., 'en', 'hi')
        lang_code = detect(text)
        return lang_code
    except Exception as e:
        print(f"ERROR[langdetect]: Language detection failed: {e}")
        return None # Return None if detection fails

def identify_language_from_audio(audio_path: str):
    try:
        enhanced_path = enhance_audio_quality(audio_path)
        result = _lid_pipeline(enhanced_path, top_k=2)
        if result:
            for res in result:
                lang_code = normalize_mms_label(res.get('label'))
                confidence = res.get('score', 0.0)
                if lang_code and confidence > 0.2:
                    return lang_code, enhanced_path
        return 'hi', enhanced_path
    except Exception as e:
        print(f"ERROR[audio]: Language ID failed: {e}")
        return 'hi', audio_path

def transcribe_audio(audio_path: str, lang_code: str):
    whisper_lang_map = {
        'hi': 'hindi', 'en': 'english', 'gu': 'gujarati', 'bn': 'bengali', 'ta': 'tamil', 'te': 'telugu',
        'mr': 'marathi', 'kn': 'kannada', 'ml': 'malayalam', 'pa': 'punjabi', 'ur': 'urdu'
    }
    whisper_lang = whisper_lang_map.get(lang_code, 'hindi')
    
    try:
        out = _asr_pipeline(audio_path, generate_kwargs={"language": whisper_lang, "task": "transcribe", "temperature": 0.0, "initial_prompt": "कृषि, खेती, पौधे, बीमारी, फसल, मंडी, दाम, agriculture, farming, plants, crops, diseases, price."})
        text = out.get("text", "").strip()
        if text and not is_garbage_transcription(text):
            return clean_transcription(text)
    except Exception as e:
        print(f"WARN[audio]: Language-specific transcription failed: {e}")

    try:
        out = _asr_pipeline(audio_path, generate_kwargs={"task": "transcribe", "temperature": 0.1})
        text = out.get("text", "").strip()
        if text and not is_garbage_transcription(text):
            return clean_transcription(text)
    except Exception as e:
        print(f"WARN[audio]: Auto-detection transcription failed: {e}")

    return None

def is_garbage_transcription(text: str) -> bool:
    if not text or len(text.strip()) < 3: return True
    garbage_patterns = ["thank you", "thanks for watching", "subscribe", "music", "♪"]
    text_lower = text.lower().strip()
    return any(pattern in text_lower for pattern in garbage_patterns)

def clean_transcription(text: str) -> str:
    if not text: return text
    text = ' '.join(text.split())
    artifacts = ["Thanks for watching!", "Subtitles by the Amara.org community"]
    for artifact in artifacts:
        text = text.replace(artifact, "").strip()
    return text

def translate_with_llama(text: str, src_lang_code: str) -> str:
    if not _groq_client or not src_lang_code or src_lang_code == 'en':
        return text
    language_names = {
        'hi': 'Hindi', 'gu': 'Gujarati', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu', 'mr': 'Marathi',
        'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi', 'ur': 'Urdu', 'or': 'Odia', 'as': 'Assamese'
    }
    src_lang_name = language_names.get(src_lang_code, src_lang_code)
    try:
        translation_prompt = f"Translate the following agricultural query from {src_lang_name} to English. Preserve the exact meaning and all technical terms. Provide only the English translation.\n\n{src_lang_name} text: \"{text}\"\n\nEnglish translation:"
        messages = [{"role": "system", "content": "You are a professional translator specializing in agricultural content. Your task is to provide accurate, literal translations from Indian languages to English."}, {"role": "user", "content": translation_prompt}]
        chat_completion = _groq_client.chat.completions.create(messages=messages, model=GROQ_MODEL, temperature=0.0, max_tokens=400, top_p=0.9)
        return post_process_translation(chat_completion.choices[0].message.content.strip())
    except Exception as e:
        print(f"ERROR[llama_translate]: Translation failed: {e}")
        return text

def post_process_translation(text: str) -> str:
    if not text: return text
    text = ' '.join(text.split())
    if text: text = text[0].upper() + text[1:]
    text = text.replace('"', '').strip()
    return text

def ocr_with_easyocr(image_path: str):
    if _easyocr_reader is None: return None
    try:
        lines = _easyocr_reader.readtext(image_path, detail=0, paragraph=True)
        return "\n".join([t.strip() for t in lines if t and t.strip()])
    except Exception as e:
        print(f"EasyOCR failed at runtime: {e}")
        return None

def process_voice_input(audio_path: str):
    lang_code, processed_path = identify_language_from_audio(audio_path)
    if not lang_code:
        return "Language detection failed.", "Could not detect language.", "en"

    native_text = transcribe_audio(processed_path, lang_code)
    if not native_text:
        return "Transcription failed.", "Could not transcribe audio.", lang_code

    translated_text = translate_with_llama(native_text, lang_code)
    if not translated_text:
        return native_text, native_text, lang_code # Fallback to native if translation fails
        
    return translated_text, native_text, lang_code

def process_text_input(text: str):
    lang_code = detect_language_text(text)
    if lang_code and lang_code != 'en':
        translated_text = translate_with_llama(text, lang_code)
        return translated_text, text, lang_code
    return text, text, lang_code or 'en'

def process_image_input(image_path: str):
    try:
        extracted_text = ocr_with_easyocr(image_path)
        if not extracted_text or not extracted_text.strip():
            return "No text found in image.", "", "en"
        lang_code = detect_language_text(extracted_text)
        if lang_code and lang_code != 'en':
            translated_text = translate_with_llama(extracted_text, lang_code)
            return translated_text, extracted_text, lang_code
        return extracted_text, extracted_text, lang_code or 'en'
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Could not process image.", "", "en"