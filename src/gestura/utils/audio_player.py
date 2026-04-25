import pyttsx3 
import threading 
import pythoncom  

class AudioPlayer:
    @staticmethod 
    def _speak_task(text):
        try: 
            pythoncom.CoInitialize()
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 130)
            
            engine.say(text)
            engine.runAndWait()
            
        except Exception as e:
            print(f"[AudioPlayer Error] Gagal memutar audio: {e}")
            
        finally:
            try:
                pythoncom.CoUninitialize()
            except:
                pass
            
    @classmethod
    def play_alphabet(cls, alphabet):
        threading.Thread(target=cls._speak_task, args=(alphabet,), daemon=True).start()