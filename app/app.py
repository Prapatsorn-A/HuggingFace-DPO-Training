from flask import Flask, request, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer from the checkpoint
model = AutoModelForCausalLM.from_pretrained("./dpo_model/checkpoint-400")
tokenizer = AutoTokenizer.from_pretrained("./dpo_model/checkpoint-400")

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(input_text):
    # Tokenize the input text, without including it in the output generation
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate a response with better controls for coherence and no question repetition
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=50,  # Limit the response length to 50 tokens
        num_return_sequences=1,  # Return only one sequence
        no_repeat_ngram_size=2,  # Prevent repeating the same n-grams
        top_p=0.9,  # Nucleus sampling for diversity
        temperature=0.7,  # Control randomness
        do_sample=True,  # Enable sampling instead of greedy decoding
        early_stopping=True,  # Stop generation when a complete thought is formed
        pad_token_id=tokenizer.eos_token_id  # Ensure padding is handled correctly
    )

    # Decode the output response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean the response
    response = response.strip()

    # Remove the original input question from the response if it's included
    if response.lower().startswith(input_text.lower()):
        response = response[len(input_text):].strip()

    # Optionally, remove special tokens
    response = response.replace(tokenizer.eos_token, "").strip()

    return response

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = generate_response(user_input)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html", user_input="", response="")

if __name__ == "__main__":
    app.run(debug=True)
