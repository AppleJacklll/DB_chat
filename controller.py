from flask import Blueprint, request, jsonify
import chat.llm as llm

chat_bp = Blueprint('chat_bp', __name__)


@chat_bp.route('/chat', methods=['POST'])
def chat():
    
    print("Start chat endpoint")
    data = request.get_json()
    
    if 'prompt' not in data:
        return jsonify({"error": "No prompt part in the request"}), 400
    
    if 'table' not in data:
        return jsonify({"error": "No table part in the request"}), 400
    
    prompt = data.get('prompt')
    if not prompt or prompt == "":
        return jsonify({"error": "No prompt provided"}), 400
    table = data.get('table')
    if not table or table == "":
        return jsonify({"error": "No table provided"}), 400
    
    result = llm.chat(prompt, table)
    return result, 200
    