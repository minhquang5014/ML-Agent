from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as f

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map
from sentence_transformers import SentenceTransformer, util 
import os

def get_relevant_context(user_input, vault_embeddings, vault_content, model, top_k=3):
    """Retrieves the top-k most relevant context from the vault based on the user input."""
    if vault_embeddings.nelement() == 0: # Check if the tensor has any elements
        return []
    # Encode the user input
    input_embedding = model.encode([user_input])
    # Compute cosine similarity between the input and vault embeddings 
    cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0] # Adjust top_k if it's greater than the number of available scores 
    top_k = min (top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

class LLM_Model:
    def __init__(self, model_name, cache_dir, local_dir):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pretrained_model_and_tokenizer()
    def pretrained_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
        # # Initialize model with empty weights to manage device map efficiently
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(self.local_dir)
            # tie the model weights to prevent memory overhead
            self.model.tie_weights()

        # Infer device map to manage model layers between GPU and CPU
        device_map = infer_auto_device_map(self.model, max_memory={0: "4GiB", "cpu": "6GiB"})
        # configure quantization using BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        # Load the model with 8-bit quantization and mixed precision
        self.model = AutoModelForCausalLM.from_pretrained(
            self.local_dir,
            device_map='auto',
            quantization_config=quantization_config,
            offload_folder="./offload",
            torch_dtype=torch.float16,
            offload_state_dict=True
        )

        self.model.gradient_checkpointing_enable()

    def generate_text(self, prompt, max_length=500):
        # Move inputs to the appropriate device (CPU or GPU)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate text using the optimized model
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
        # Decode the outputs and return the generated text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

model_name = "facebook/opt-2.7B"
cache_dir = "./optimized-opt-2.7B-8bit"
local_dir = "./optimized-opt-2.7B-8bit/models--facebook--opt-2.7B/snapshots/905a4b602cda5c501f1b3a2650a4152680238254"
# prompt = "Once upon a time"
llm_model = LLM_Model(model_name=model_name, cache_dir=cache_dir, local_dir=local_dir)
# generated_text = llm_model.generate_text(prompt)
# print(generated_text)



def chatgpt_streamed(user_input, system_message, conversation_history, chatbot_name, vault_embeddings, vault_content,
                     embed_model, top_k = 3):
    """
    user_input: str
    system_message: str (system instructions, role prompt)
    conversation_history: list of dicts [{"role": "user"/"assistant", "content": str}]
    chatbot_name: str
    vault_embeddings: torch.Tensor of stored embeddings
    vault_content: list of str
    embed_model: SentenceTransformer instance
    """
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, embed_model, top_k)
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history])
    context_text = "\n".join(relevant_context)
    full_prompt = (
        f"{system_message}\n\n"
        f"Relevant information from the knowledge vault:\n{context_text}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"User: {user_input}\n{chatbot_name}:"
    )
    generated_text = llm_model.generate_text(full_prompt)
    if chatbot_name in generated_text:
        output = generated_text.split(f"{chatbot_name}:")[-1].strip()
    return output


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
PINK = '\033[95m'
CYAN = '\033[96m' 
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'
chatbot_dir = 'temp-memory/chatbot2.txt'
vault_dir = 'temp-memory/vault.txt'
def user_chatbot_conversation():
    conversation_history = []
    system_message = open_file(chatbot_dir)
    sentence_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Create embeddings for the initial vault content
    vault_content = []
    if os.path.exists(vault_dir):
        with open(vault_dir, "r", encoding="utf-8") as vault_file:
            vault_content = vault_file.readlines()
    vault_embeddings = sentence_embedding_model.encode(vault_content) if vault_content else []
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    while True:
        user_input = input("Input here: ").strip()
        # Clean up the temporary audio file
        if user_input.lower() == "exit": # Say 'exit' to end the conversation break
            break
        elif user_input.lower().startswith(("print info", "Print info")): # Print the contents of the vault.txt file 
            print("Info contents:")
            if os.path.exists(vault_dir):
                with open(vault_dir, "r", encoding="utf-8") as vault_file: 
                    print(vault_file.read())
            else:
                print("Info is empty.")
            continue
        elif user_input.lower().startswith(("delete info", "Delete info")): 
            confirm = input("Are you sure? Say 'Yes' to confirm: ")
            if confirm.lower() == "yes":
                if os.path.exists(vault_dir):
                    os.remove(vault_dir)
                    print("Info deleted.") 
                    vault_content= []
                    vault_embeddings = []
                    vault_embeddings_tensor = torch.tensor(vault_embeddings)
                else:
                    print("Info is already empty.")
            else:
                print("Info deletion cancelled.")
            continue
        elif user_input.lower().startswith(("insert info", "insert info")): 
            with open(vault_dir, "a", encoding="utf-8") as vault_file: 
                vault_file.write(user_input + "\n")
            print("Wrote to info.")
            # Update the vault content and embeddings
            vault_content = open(vault_dir, "r", encoding="utf-8").readlines()
            vault_embeddings = sentence_embedding_model.encode(vault_content)
            vault_embeddings_tensor = torch.tensor(vault_embeddings)
            continue
        print(CYAN + "You: ", user_input + RESET_COLOR)
        conversation_history.append({"role": "user", "content": user_input})

        print(PINK + "Sarah: "+ RESET_COLOR)
        chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Sarah", vault_embeddings_tensor, vault_content, sentence_embedding_model) 
        conversation_history.append({"role": "assistant", "content": chatbot_response}) 
        # prompt2 = chatbot_response
        # audio_file_pth2 = "C:/Users/kris_/Python/fsts2/XTTS-v2/samples/emma2.wav"
        # process_and_play(prompt2, audio_file_pth2)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        print(conversation_history, chatbot_response)

user_chatbot_conversation() # Start the conversation