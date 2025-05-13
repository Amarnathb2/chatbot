from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

model_path = "./chatbot_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["user_input"]
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    reply_ids = model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
