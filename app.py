from flask import Flask, request, jsonify, g
from ctransformers import AutoModelForCausalLM
from flask_cors import CORS
import psycopg2
import pandas as pd
from LLM.LangChainOllama import Initialize_LLM, chatLLM
from langchain_community.document_loaders import PyPDFLoader

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.

app = Flask(__name__)



@app.before_request
def before_request():
    if not hasattr(g, 'chatmodel'):
        LLM = Initialize_LLM()
        g.chatmodel = LLM[0]
        g.retriever = LLM[1]
        g.gemini = LLM[2]

CORS(app)

@app.route("/chatLLM", methods=["POST"])
def ask_LLM():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get("question")

        Question = f"""
        問題:{question}
        """
        llmAnwser = []
        llmAnwser.append(chatLLM(Question,g.gemini,g.retriever))
        print(llmAnwser)
        print(g.retriever)
        
        return jsonify({"status": "success", "llmAnwser": llmAnwser}), 200
        
    else:
        return jsonify({"message": "No data structure available to update"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


