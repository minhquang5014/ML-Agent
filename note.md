now running TTS smoothly. But the problem here is to run model with effiency

First, maybe try loading the 3 models in the separate threadings

Second, try to optimize the entire models so that it can run on every device, regardless of the fact that whether the user is using GPU or CPU

Third, we can try to create RAG (retrieval augmented generation) - it means the model will remember the key features the users input in, retrieve those features and store in the memory.

Fourth, custom train the model (checkpoint.pth) - tailer to our custom usage.