import os
import io
from dotenv import load_dotenv, find_dotenv
import base64
from flask import Flask, request, jsonify
from google.oauth2 import service_account
from google.cloud import speech, texttospeech
from openai import OpenAI
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# load environment variables
load_dotenv(find_dotenv())

# OpenAI Key
openAI = OpenAI(
    api_key = os.getenv("OPENAI_KEY"),
)

# Set the environment variable for Google Application Credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "sa-credential.json"

# Initialize Google Cloud Clients
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# Languages
langs = { "en": "en-US", "es": "es-AR", "it": "it-IT", "ur": "ur-IN", "hi": "hi-IN" }

messages = []

# Spoken Dialogue
@app.route('/process', methods=['POST'])
def process_audio():
    audio_content = request.files['audio'].read()
    langCode = request.form['language']
    
    # Read transcripts from speech
    transcript = getTranscript(langCode, audio_content)
    
    # If Audio Transcript not English, translate from choosen Language
    # to English for easy Intent Identification and response generation
    if langCode != 'en':
        transcript = translateLang(transcript, 'en')
    
    # Generate response using NLP/GPT based on the e-Commerce trained data context
    # dataset 
    dataset = ''
    with open("dataset.txt", "r") as f:
        dataset = f.read()
        
    context_prompt = f"Chat:\n{dataset}\nUser: {transcript}\n"
    messages.append({"role": "user", "content": context_prompt})
    responseText = GPT(messages)
    
    # Now check if conversation language is not English
    # then translate the response back to choosen Language
    if langCode != 'en':
        responseText = translateLang(responseText, langCode)

    # Finally, convert response to Speech
    # Using the Text-to-Speech Synthesis API
    synthesis_input = texttospeech.SynthesisInput(text = responseText)
    voice_params = texttospeech.VoiceSelectionParams(language_code = langCode)
    audio_config = texttospeech.AudioConfig(audio_encoding = texttospeech.AudioEncoding.MP3)

    tts_response = tts_client.synthesize_speech(
        input = synthesis_input, 
        voice = voice_params, 
        audio_config = audio_config
    )

    # Encode audio content to base64 for easy browser playback
    audio_base64 = base64.b64encode(tts_response.audio_content).decode('utf-8')

    return jsonify({'audio_content': audio_base64})

# Chat Dialogue
@app.route('/chat', methods=['POST'])
def chat():
    responseText = ''
    langCode = request.json.get('language')
    msg = request.json.get('message')
    
    # If Audio Transcript not English, translate from choosen Language
    # to English for easy Intent Identification and response generation
    if langCode != 'en':
        msg = translateLang(msg, 'en')
        
    # Generate response using NLP/GPT based on the e-Commerce trained data context
    # dataset 
    dataset = ''
    with open("dataset.txt", "r") as f:
        dataset = f.read()
        
    context_prompt = f"Chat:\n{dataset}\nUser: {msg}\n"
    messages.append({"role": "user", "content": context_prompt})
    responseText = GPT(messages)
    
    # Now check if conversation language is not English
    # then translate the response back to choosen Language
    if langCode != 'en':
        responseText = translateLang(responseText, langCode)
    
    return jsonify({'message': responseText})

# Get Audio transcipt
def getTranscript(langCode, audioFile):
    response = ''
    
    # Test with audio file
    # file_name = "OSR_us_000_0013_8k.wav"
    # with io.open(file_name, "rb") as audio_file:
    #     content = audio_file.read()
    #     audio = speech.RecognitionAudio(content=content)

    # Speech-to-Text
    config = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
        enable_automatic_punctuation = True,
        language_code = langs[langCode],
        # audio_channel_count = 2,
        # sample_rate_hertz = 16000,
    )
    audio = speech.RecognitionAudio(content = audioFile)
    speechTranscript = speech_client.recognize(
        request = {
            "config": config, 
            "audio": audio
        }
    )
    
    # Read transcripts from speech
    for result in speechTranscript.results:
        response += " " + result.alternatives[0].transcript
        
    return response

# Translate between languages using GPT
def translateLang(msg, langCode):
    response = ''
    
    translate_prompt = f"Translate the following to {langs[langCode]}: {msg}"
    messages.append({"role": "user", "content": translate_prompt})
    response = GPT(messages)
    
    return response

# Using GPT to generate support response
def GPT(msg):
    response = ''
    
    # get response from dataset
    genResponse = openAI.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = msg
    )
    response = genResponse.choices[0].message.content.strip()
    
    return response

# BLEU Evaluation
@app.route('/evaluate_bleu', methods=['POST'])
def evaluate_bleu():
    response = ''
    
    langCode = request.json.get('language')
    source = request.json.get('source')
    target = request.json.get('target')
    reference = request.json.get('reference')
    
    # Compute BLEU score
    reference = [reference.split()]
    target = target.split()
    score = sentence_bleu(reference, target)
    
    response = jsonify({
        'lang': langCode,
        'source_text': source,
        'target_text': target,
        'reference_text': reference,
        'bleu_score': score
    })
    print(response) # tokenized response
    
    # save evaluation data
    file_exists = os.path.isfile('bleu_scores.csv')
    df = pd.DataFrame([[langCode, source, target, reference, score]],
                      columns=['language', 'source', 'target', 'reference', 'BLEU_score'])
    df.to_csv('bleu_scores.csv', mode='a', header=not file_exists, index=False)
    
    return response

# Load BLEU Evaluation Data
@app.route('/load_bleu')
def load_bleu():
    df = pd.read_csv('bleu_scores.csv')
    result = df.to_dict(orient='records')
    return jsonify(result)

# run app
if __name__ == '__main__':
    app.run(debug=True)
