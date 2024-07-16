import os
import io
from dotenv import load_dotenv, find_dotenv
import base64
from flask import Flask, request, jsonify
from google.oauth2 import service_account
from google.cloud import speech, texttospeech
from openai import OpenAI
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
sa_credential = service_account.Credentials.from_service_account_file('sa-credential.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "sa-credential.json"

# Initialize Google Cloud Clients
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# Languages
langs = { "en": "en-US", "es": "es-AR", "it": "it-IT", "ur": "ur-IN", "hi": "hi-IN" }

messages = []

@app.route('/process', methods=['POST'])
def process_audio():
    audio_content = request.files['audio'].read()
    langCode = request.form['language']
    
    # # path to audio file
    # file_name = "OSR_us_000_0013_8k.wav"
    # with io.open(file_name, "rb") as audio_file:
    #     content = audio_file.read()
    #     audio = speech.RecognitionAudio(content=content)

    # Speech-to-Text
    config = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
        enable_automatic_punctuation = True,
        language_code = langs[langCode]
        # audio_channel_count = 1,
        # sample_rate_hertz = 16000,
    )
    audio = speech.RecognitionAudio(content=audio_content)
    response = speech_client.recognize(
        request = {
            "config": config, 
            "audio": audio
        }
    )
    
    # Read transcripts from speech
    transcript = ''
    for result in response.results:
        transcript += " " + result.alternatives[0].transcript
    
    # Generate response using GPT-3 or GPT-4
    messages.append({"role": "user", "content": transcript})
    text_response = GPT(messages)

    # Translate response using GPT API
    translate_prompt = f"Translate the following text to {langs[langCode]}: {text_response}"
    messages.append({"role": "user", "content": translate_prompt})
    
    translated_response = GPT(messages)

    # Text-to-Speech
    synthesis_input = texttospeech.SynthesisInput(text = translated_response)
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

# Using GPT to generate support response
def GPT(msg):
    response = ''
    
    # load dataset
    dataset = ''
    with open("dataset.txt", "r") as f:
        dataset = f.read()
    
    # get response from dataset
    genResponse = openAI.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = msg
    )
    response = genResponse.choices[0].message.content.strip()
    
    return response

# run app
if __name__ == '__main__':
    app.run(debug=True)
