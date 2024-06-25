import base64
from threading import Lock, Thread
import asyncio

import cv2
import numpy as np
from PIL import ImageGrab
from cv2 import imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# Updated Deepgram imports
from deepgram import Deepgram
from deepgram.types import LiveTranscriptionEvents
from deepgram.transcription import PrerecordedOptions, LiveOptions

import pyaudio

load_dotenv()

class ScreenCaptureStream:
    def __init__(self):
        self.frame = np.array(ImageGrab.grab())
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            frame = np.array(ImageGrab.grab())

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)
        self.deepgram = Deepgram()  # Changed from DeepgramClient to Deepgram

    async def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            await self._tts(response)

    async def _tts(self, response):
        try:
            # Use Deepgram's text-to-speech API
            response = await self.deepgram.transcription.v("1").speak(response)
            audio_data = response.read_data()
            
            # Play the audio using pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=22050,
                            output=True)
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty design assistant for Canva. Use the chat history and the 
        screen capture provided to offer design suggestions and tips for the user's 
        Canva project. Focus on layout, color schemes, and design principles.

        Use few words in your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

screen_capture_stream = ScreenCaptureStream().start()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

assistant = Assistant(model)

async def process_audio(audio_data):
    try:
        deepgram = Deepgram()  # Changed from DeepgramClient to Deepgram
        options = PrerecordedOptions(model="nova-2", language="en-US")
        response = await deepgram.transcription.prerecorded.v("1").transcribe(audio_data, options)
        return response.results.channels[0].alternatives[0].transcript
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return ""

async def audio_callback(recognizer, audio):
    try:
        audio_data = audio.get_wav_data()
        prompt = await process_audio(audio_data)
        if prompt:
            await assistant.answer(prompt, screen_capture_stream.read(encode=True))
    except Exception as e:
        print(f"Error processing audio: {e}")

async def main():
    deepgram = Deepgram()  # Changed from DeepgramClient to Deepgram
    
    connection = deepgram.transcription.live.v("1").connect()  # Updated connection method
    
    async def on_message(self, result):
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) > 0:
            await assistant.answer(sentence, screen_capture_stream.read(encode=True))

    connection.on(LiveTranscriptionEvents.Transcript, on_message)

    options = LiveOptions(
        model="nova-2",
        language="en-US",
        encoding="linear16",
        channels=1,
        sample_rate=16000,
    )
    
    await connection.start(options)

    print("Listening... Press Ctrl+C to stop.")
    
    try:
        while True:
            cv2.imshow("screen capture", screen_capture_stream.read())
            if cv2.waitKey(1) in [27, ord("q")]:
                break
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        screen_capture_stream.stop()
        cv2.destroyAllWindows()
        await connection.finish()

if __name__ == "__main__":
    asyncio.run(main())