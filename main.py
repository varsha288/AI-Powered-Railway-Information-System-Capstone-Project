import speech_recognition as sr
import pyttsx3
from llm_util import LLM

# Import necessary modules
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)
llm_obj = LLM()

# Define SpeechToText class for speech recognition
class SpeechToText:
    # Constructor
    def __init__(self, lang='en'):
        self.r = sr.Recognizer()
        self.language = lang
        
    # Method to extract text from speech
    def extract_text_from_speech(self):
        with sr.Microphone() as source2:
            print("Listening....")
            self.r.adjust_for_ambient_noise(source2, duration=0.3)
            audio2 = self.r.listen(source2)
            MyText = self.r.recognize_google(audio2)
            MyText = MyText.lower()
            return MyText

# Define TextToSpeech class for speech synthesis
class TextToSpeech:
    # Constructor
    def __init__(self):
        self.engine = pyttsx3.init()
        self.is_engine_running = False
        
    # Method to convert text to speech
    def text_to_speech(self, command):
        if not self.is_engine_running:
            self.engine.startLoop(False)
            self.is_engine_running = True
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
        self.engine.say(command)
        self.engine.iterate()  # Process pending events

# Define routes for the Flask app
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_query', methods=['POST'])
def process_query():
    if request.method == 'POST':
        query = request.form['query']
        response = llm_obj.answer_to_the_question(query)
        tts = TextToSpeech()
        tts.text_to_speech(response)
        return render_template('index.html', response=response)

@app.route('/speech_to_text', methods=['GET'])
def speech_to_text():
    stt = SpeechToText()
    query = stt.extract_text_from_speech()
    response = llm_obj.answer_to_the_question(query)
    tts = TextToSpeech()
    tts.text_to_speech(response)
    return render_template('index.html', query=query, response=response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

