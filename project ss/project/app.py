# app.py (Flask backend)
from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import random
import json
from dotenv import load_dotenv

# serve static files from project folder so index.html can be served by Flask
app = Flask(__name__, static_url_path='', static_folder='.')

# load environment from .env when present
load_dotenv()

fashion_data = {
    "casual": ["Jeans + T-shirt", "Sneakers + Hoodie", "Denim Jacket + White Tee"],
    "formal": ["Suit + Formal Shoes", "Blazer + Trousers", "Saree / Kurti Set"],
    "party": ["Black Dress", "Stylish Kurti + Jewelry", "Designer Shirt + Jeans"],
    "traditional": ["Saree", "Lehenga", "Kurta Pyjama"]
}

# Hugging Face configuration from environment
HF_API_KEY = os.environ.get('HF_API_KEY')
HF_MODEL = os.environ.get('HF_MODEL')


def call_hf_model(prompt: str) -> str:
    """Call the Hugging Face Inference API and return text output.

    Requires environment variables `HF_API_KEY` and `HF_MODEL` to be set.
    Raises requests exceptions on HTTP errors.
    """
    if not HF_API_KEY or not HF_MODEL:
        raise RuntimeError('Hugging Face API key or model not configured')

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Accept": "application/json"}
    payload = {"inputs": prompt}

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()

    try:
        data = resp.json()
    except ValueError:
        return resp.text

    # handle common HF response formats
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first.get('generated_text') or first.get('text') or str(first)
        return str(first)
    if isinstance(data, dict):
        return data.get('generated_text') or data.get('text') or str(data)

    return str(data)

@app.route('/')
def home():
    # deliver the frontend HTML when the root is requested
    return send_from_directory('.', 'index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    style = data.get('style')
    color = data.get('color')
    occasion = data.get('occasion')
    # Build a prompt that asks the model to return JSON with multiple recommendations
    if HF_API_KEY and HF_MODEL:
        prompt = (
            "You are an expert fashion stylist. Provide exactly three distinct outfit recommendations "
            "for the user. Respond ONLY in valid JSON with the following structure:\n"
            "{ \"recommendations\": [ {\"outfit\": string, \"color\": string, \"explanation\": string}... ], \"top_tip\": string }\n"
            f"Each recommendation must include a suggested outfit name, a suggested dress color (single word or short phrase), and a short explanation (1-2 sentences) explaining why it works for the occasion and color.\n"
            f"Inputs: style={style or 'any'}, color={color or 'none'}, occasion={occasion or 'general'}.\n"
            "Do not include any text outside the JSON object."
        )

        try:
            hf_text = call_hf_model(prompt)
            parsed = None
            # Try direct JSON parse
            try:
                parsed = json.loads(hf_text)
            except Exception:
                # Attempt to extract JSON substring between first '{' and last '}'
                start = hf_text.find('{')
                end = hf_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(hf_text[start:end+1])
                    except Exception:
                        parsed = None

            if parsed and isinstance(parsed, dict) and parsed.get('recommendations'):
                # validate and return model output
                return jsonify(parsed)
        except Exception:
            # on any error fall through to fallback
            pass

    # Fallback: generate up to 3 recommendations from static data
    recs = []
    candidates = fashion_data.get(style, []) if style else sum(fashion_data.values(), [])
    # make candidates unique and shuffle
    candidates = list(dict.fromkeys(candidates))
    random.shuffle(candidates)

    for i in range(3):
        if i < len(candidates):
            outfit = candidates[i]
            suggested_color = color or 'neutral/black'
            explanation = f"{outfit} in {suggested_color} suits {occasion or 'many occasions'} by keeping the look polished and appropriate."
        else:
            outfit = "Smart Casual Outfit"
            suggested_color = color or 'neutral'
            explanation = f"A {suggested_color} smart casual outfit works well for {occasion or 'general'} settings."

        recs.append({
            'outfit': outfit,
            'color': suggested_color,
            'explanation': explanation
        })

    result = {
        'recommendations': recs,
        'top_tip': 'Choose one statement accessory and keep the rest minimal.'
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

# Example usage:
# fetch('/recommend',{
#     method:'POST',
#     headers:{'Content-Type':'application/json'},
#     body:JSON.stringify({style:style,color:color,occasion:occasion})
# })