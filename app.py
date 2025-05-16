# Explainable AI with Debating Sub-Agents using LangChain and Free APIs
import os
import json
import requests
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from google.generativeai import GenerativeModel
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
openai.api_key = os.getenv("OPENROUTER_KEY_1")
openai.api_base = "https://openrouter.ai/deepseek/deepseek-r1:free"

# Query the DeepSeek model
response = openai.ChatCompletion.create(
    model="deepseek/deepseek-chat-v3-0324:free",
    messages=[
        {"role": "user", "content": "Explain gravity like I'm 5 years old."}
    ]
)

# Print the model's reply
print(response["choices"][0]["message"]["content"])

# Custom Grok API wrapper
class GrokAPI:
    def __init__(self, api_key, model="grok-3"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.x.ai/v1/grok"

    def __call__(self, prompt):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "prompt": prompt, "max_tokens": 200}
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("text", "")
        except Exception as e:
            return f"Error: Grok API failed - {str(e)}"

# Generator Agent
class Generator:
    def __init__(self):
        # Gemini API 1
        self.gemini_model = GenerativeModel("gemini-1.5-flash", api_key=GEMINI_API_KEY)
        # Grok API 1
        self.grok_llm = GrokAPI(GROK_API_KEY)
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="Provide a detailed and accurate answer to: {query}"
        )

    def generate_response(self, query):
        try:
            gemini_response = self.gemini_model.generate_content(
                self.prompt_template.format(query=query)
            ).text
        except Exception as e:
            gemini_response = f"Error: Gemini API failed - {str(e)}"
        grok_response = self.grok_llm(self.prompt_template.format(query=query))
        return {
            "gemini_response": gemini_response,
            "grok_response": grok_response,
            "combined": f"Gemini: {gemini_response}\nGrok: {grok_response}"
        }

# Critic Agent
class Critic:
    def __init__(self):
        # Gemini API 2 (critique prompt)
        self.gemini_model = GenerativeModel("gemini-1.5-flash", api_key=GEMINI_API_KEY)
        # Hugging Face API 1
        self.hf_llm = HuggingFaceHub(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=HF_API_KEY,
            model_kwargs={"max_length": 200}
        )
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query"],
            template="Critique this response to '{query}': {response}\nIdentify specific flaws and suggest actionable improvements."
        )

    def critique_response(self, response, query):
        try:
            gemini_critique = self.gemini_model.generate_content(
                self.prompt_template.format(response=response, query=query)
            ).text
        except Exception as e:
            gemini_critique = f"Error: Gemini API failed - {str(e)}"
        hf_critique = self.hf_llm(self.prompt_template.format(response=response, query=query))
        return {
            "gemini_critique": gemini_critique,
            "hf_critique": hf_critique,
            "combined": f"Gemini Critique: {gemini_critique}\nHugging Face Critique: {hf_critique}"
        }

# Validator Agent
class Validator:
    def __init__(self):
        # Grok API 2
        self.grok_llm = GrokAPI(GROK_API_KEY)
        # Hugging Face API 2
        self.hf_llm = HuggingFaceHub(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=HF_API_KEY,
            model_kwargs={"max_length": 200}
        )
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query"],
            template="Validate this response to '{query}': {response}\nIs it accurate, coherent, and complete? Suggest final improvements."
        )

    def validate_response(self, response, query):
        grok_validation = self.grok_llm(self.prompt_template.format(response=response, query=query))
        hf_validation = self.hf_llm(self.prompt_template.format(response=response, query=query))
        return {
            "grok_validation": grok_validation,
            "hf_validation": hf_validation,
            "combined": f"Grok Validation: {grok_validation}\nHugging Face Validation: {hf_validation}"
        }

# Orchestrator
class Orchestrator:
    def __init__(self):
        self.generator = Generator()
        self.critic = Critic()
        self.validator = Validator()
        self.debate_rounds = 2

    def run_debate(self, query):
        reasoning_log = []
        
        # Step 1: Generate initial response
        gen_output = self.generator.generate_response(query)
        current_response = gen_output["combined"]
        reasoning_log.append({"step": "Initial Generation", "response": current_response})

        # Step 2: Debate loop (Critic refines response)
        for round in range(self.debate_rounds):
            critique_output = self.critic.critique_response(current_response, query)
            reasoning_log.append({
                "step": f"Critique Round {round + 1}",
                "critique": critique_output["combined"]
            })
            # Refine response (basic synthesis for demo)
            current_response = f"Refined: {current_response}\nCritique: {critique_output['combined']}"
            reasoning_log.append({
                "step": f"Refined Response Round {round + 1}",
                "response": current_response
            })

        # Step 3: Validate final response
        validation_output = self.validator.validate_response(current_response, query)
        reasoning_log.append({
            "step": "Final Validation",
            "validation": validation_output["combined"]
        })
        final_response = f"Final: {current_response}\nValidation: {validation_output['combined']}"

        return {
            "final_response": final_response,
            "reasoning_log": reasoning_log
        }

# Example usage
if __name__ == "__main__":
    orchestrator = Orchestrator()
    query = "What is the best way to learn Python?"
    result = orchestrator.run_debate(query)
    
    print("Final Response:", result["final_response"])
    print("\nReasoning Log:")
    for log in result["reasoning_log"]:
        print(json.dumps(log, indent=2))