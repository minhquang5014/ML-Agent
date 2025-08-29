Once upon a time, there was a software developer who wanna develop a voice-assistance app for his own work. 
This app is being built because he wanna create a desktop app, that is universal and essential for every computers.
By inputting the voice command, the "Voice assistance" app can answer in a natural and fluent way, also executing the command for the users. 

For example: Hey my beautifule Sarah (presume the name of the virtual assistance is Sarah), I wanna watch Youtube videos, wanna watch something fun after a long day at work. can you suggest for me, which is the best movie, or videos??

The beautiful assistance Sarah responded with an anwser: Why don't you try the latest ytb video on Tommy's Channel: "How to flirt a girl - How to start a natural conversation with a beautiful girl you met on the street"

And Sarah pops up youtube video on the screen without the users even navigating around the browser, around Ytb to find interesting videos.


Workflow
SST: Whisper model, or Distil-whisper
Sentence Embedding: all-MiniLM-L6-v2
For LLM: mistral 7B model in gguf format - this is running on top of the llama-cpp module - currently running test successfully
TTS: OpenVoice module but it requires complex setup and download
or we can use xTTS-v2 or gTTS or parlerTTS
KittenTTS is also, but it runs on CPU, maybe requires C++ extension installation