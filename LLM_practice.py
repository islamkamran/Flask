from flask import Flask, request, jsonify
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import os
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# OpenAI key 
os.environ["OPENAI_API_KEY"]=""

@app.route('/predict')
def predict():
    file = request.files('file')
    #   save to local system (not recommended)
    file.save(file.filename)

    #   making the agent that will work for us
    agent = create_csv_agent(OpenAI(temperature=0), file.filename, verbose=True)
    print(agent.agent.llm_chain.prompt.template)

    #    now taking the question from user
    question = request.form['question']
    result = agent.run(question)

    # now remove that file
    os.remove(file.filename)
    
    return jsonify({"result": result})

 
if __name__ == '__main__':
   app.run(debug=True)