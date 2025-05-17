import os
import json
import requests
import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
OPENROUTER_KEY_1 = os.getenv("OPENROUTER_KEY_1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API keys
if not all([OPENROUTER_KEY_1, COHERE_API_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing one or more API keys in .env file")

# Memory Manager using LangChain
class MemoryManager:
    def __init__(self, clear_memory=False):
        self.memory = ConversationBufferMemory(
            memory_key="neurodialectic_memory",
            output_key="response",
            return_messages=True,
            human_prefix="system"
        )
        self.memory_file = "C:/Users/Aryan Prasad/AppData/Local/neurodialectic/memory.json"
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        if clear_memory and os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry in data.get("history", []):
                        if entry["type"] == "human":
                            self.memory.chat_memory.add_user_message(
                                HumanMessage(content=entry["content"], additional_kwargs=entry["metadata"])
                            )
                        elif entry["type"] == "ai":
                            self.memory.chat_memory.add_ai_message(
                                AIMessage(content=entry["content"], additional_kwargs=entry["metadata"])
                            )
            except Exception as e:
                print(f"Warning: Failed to load memory file: {e}")

    def store_semantic(self, fact, source, tags):
        metadata = {
            "type": "semantic",
            "source": source,
            "tags": tags,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.memory.chat_memory.add_user_message(
            HumanMessage(content=fact, additional_kwargs=metadata)
        )
        self._save_memory()

    def store_episodic(self, query, response, tags):
        content = f"Query: {query}\nResponse: {response}"
        metadata = {
            "type": "episodic",
            "tags": tags,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.memory.chat_memory.add_user_message(
            HumanMessage(content=content, additional_kwargs=metadata)
        )
        self._save_memory()

    def retrieve_context(self, query, k=3):
        context = {"semantic": [], "episodic": []}
        messages = self.memory.chat_memory.messages[-50:]
        relevant_memories = []
        
        query_lower = query.lower()
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                if any(word in content for word in query_lower.split()):
                    relevant_memories.append({
                        "content": msg.content,
                        "metadata": msg.additional_kwargs
                    })
        
        relevant_memories = sorted(
            relevant_memories,
            key=lambda x: x["metadata"].get("timestamp", ""),
            reverse=True
        )[:k]
        
        for memory in relevant_memories:
            mem_type = memory["metadata"].get("type")
            if mem_type == "semantic":
                context["semantic"].append({
                    "fact": memory["content"],
                    "source": memory["metadata"].get("source", ""),
                    "tags": memory["metadata"].get("tags", [])
                })
            elif mem_type == "episodic":
                context["episodic"].append({
                    "content": memory["content"],
                    "timestamp": memory["metadata"].get("timestamp", ""),
                    "tags": memory["metadata"].get("tags", [])
                })
        
        return context

    def _save_memory(self):
        history = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                history.append({
                    "type": "human",
                    "content": msg.content,
                    "metadata": msg.additional_kwargs
                })
            elif isinstance(msg, AIMessage):
                history.append({
                    "type": "ai",
                    "content": msg.content,
                    "metadata": msg.additional_kwargs
                })
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump({"history": history}, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save memory file: {e}")

# In-memory cache for query responses and classifications
response_cache = {}

# Centralized API call wrapper
def api_call(config, prompt, retries=3, timeout=10):
    error_details = ""
    for attempt in range(retries):
        try:
            if config["type"] == "openrouter":
                client = OpenAI(api_key=OPENROUTER_KEY_1, base_url="https://openrouter.ai/api/v1")
                models = [config["model"], "mistral-7b-instruct:free"]
                for model in models:
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            timeout=timeout
                        )
                        if not response.choices or len(response.choices) == 0:
                            raise ValueError("Empty choices in OpenRouter response")
                        text = response.choices[0].message.content
                        words = text.split()[:50]  # Limit to 50 words
                        time.sleep(2)
                        return " ".join(words)
                    except Exception as e:
                        error_details = f"Model {model} failed: {str(e)}"
                        continue
                raise ValueError(f"All OpenRouter models failed: {error_details}")
            elif config["type"] == "gemini":
                headers = {"Content-Type": "application/json"}
                data = {"contents": [{"parts": [{"text": prompt}]}]}
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
                resp = requests.post(url, json=data, headers=headers, timeout=timeout)
                resp.raise_for_status()
                response = resp.json()
                if not response.get("candidates") or not response["candidates"][0].get("content"):
                    raise ValueError("Empty content in Gemini response")
                text = response["candidates"][0]["content"]["parts"][0]["text"]
                words = text.split()[:50]  # Limit to 50 words
                return " ".join(words)
            elif config["type"] == "cohere":
                headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
                data = {"message": prompt, "model": "command-r-plus", "temperature": 0.5}
                resp = requests.post("https://api.cohere.ai/v1/chat", json=data, headers=headers, timeout=timeout)
                resp.raise_for_status()
                response = resp.json()
                if not response.get("text"):
                    raise ValueError("Empty text in Cohere response")
                text = response["text"]
                words = text.split()[:50]  # Limit to 50 words
                return " ".join(words)
        except Exception as e:
            error_details = f"{config['type'].capitalize()} API failed - {str(e)}"
            if attempt == retries - 1:
                return f"Error: {error_details}"[:50]  # Limit error to 50 words
            time.sleep(2 ** attempt)
    return f"Error: {config['type'].capitalize()} API failed after {retries} retries"[:50]

# Query Classifier
class QueryClassifier:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template=(
                "Is this query debatable or non-debatable? Explain in 50 words or less and return 'Debatable' or 'Non-debatable' as the final word.\nQuery: {query}"
            )
        )
        self.non_debatable_keywords = ["what is", "define", "calculate", "who is", "when is", "where is"]

    def is_likely_non_debatable(self, query):
        return any(keyword in query.lower() for keyword in self.non_debatable_keywords)

    def classify(self, query):
        cache_key = f"classify:{query}"
        if cache_key in response_cache:
            return response_cache[cache_key]

        if self.is_likely_non_debatable(query):
            result = {
                "classification": "Non-debatable",
                "reasoning": f"Query contains non-debatable keywords: {query}. Assumed non-debatable without API call."
            }
            response_cache[cache_key] = result
            return result

        prompt = self.prompt_template.format(query=query)
        response = api_call({"type": "gemini"}, prompt)
        classification = "Debatable"
        reasoning = response
        if "Error:" in response:
            reasoning = f"Classification failed: {response}. Defaulting to Debatable."
        else:
            lines = response.strip().split('\n')
            first_line = lines[0].strip()
            if first_line in ["Debatable", "Non-debatable"]:
                classification = first_line
            else:
                response_lower = response.lower()
                if "non-debatable" in response_lower:
                    classification = "Non-debatable"
                elif "debatable" in response_lower:
                    classification = "Debatable"
        result = {
            "classification": classification,
            "reasoning": reasoning
        }
        response_cache[cache_key] = result
        return result

# Convergence Checker
class ConvergenceChecker:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["critiques", "round", "query"],
            template=(
                "Evaluate these critiques for '{query}' in round {round} in 50 words or less: {critiques}\n"
                "Do they indicate convergence or require refinement? Return 'Converged' or 'Continue' as the final word."
            )
        )

    def check_convergence(self, critiques, round, query):
        prompt = self.prompt_template.format(critiques=critiques, round=round, query=query)
        response = api_call({"type": "gemini"}, prompt)
        decision = "Continue"
        reasoning = response
        if "Error:" in response:
            reasoning = f"Convergence check failed: {response}. Defaulting to Continue."
        else:
            lines = response.strip().split('\n')
            first_line = lines[0].strip()
            if first_line in ["Converged", "Continue"]:
                decision = first_line
            else:
                response_lower = response.lower()
                if "converged" in response_lower:
                    decision = "Converged"
                elif "continue" in response_lower:
                    decision = "Continue"
        return {
            "decision": decision,
            "reasoning": reasoning
        }

# Generator Agent
class Generator:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="Using the following context, provide a detailed and accurate answer to: {query} in 50 words or less.\nContext: {context}"
        )

    def generate_response(self, query):
        cache_key = f"generate:{query}"
        if cache_key in response_cache:
            return response_cache[cache_key]

        context = self.memory_manager.retrieve_context(query)
        context_str = json.dumps(context, indent=2)
        prompt = self.prompt_template.format(query=query, context=context_str)
        gemini_response = api_call({"type": "gemini"}, prompt)
        openrouter_response = api_call({"type": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:free"}, prompt)
        result = {
            "gemini_response": gemini_response,
            "openrouter_response": openrouter_response,
            "combined": f"Gemini: {gemini_response}\nOpenRouter (LLaMA): {openrouter_response}"
        }
        response_cache[cache_key] = result
        self.memory_manager.store_episodic(query, result["combined"], ["generator_response"])
        return result

# Critic Agent
class Critic:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query", "context"],
            template=(
                "Using the following context, critique this response to '{query}' in 50 words or less: {response}\n"
                "Context: {context}\n"
                "Identify specific flaws (e.g., factual errors, missing arguments, verbosity) and suggest actionable improvements."
            )
        )

    def critique_response(self, response, query):
        context = self.memory_manager.retrieve_context(query)
        context_str = json.dumps(context, indent=2)
        prompt = self.prompt_template.format(response=response, query=query, context=context_str)
        critique = api_call({"type": "cohere"}, prompt)
        result = {
            "cohere_critique": critique,
            "combined": f"Cohere Critique: {critique}"
        }
        self.memory_manager.store_episodic(query, result["combined"], ["critic_response"])
        return result

# Validator Agent
class Validator:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query", "context"],
            template=(
                "Using the following context, validate this response to '{query}' in 50 words or less: {response}\n"
                "Context: {context}\n"
                "Is it accurate, coherent, complete? Suggest final improvements."
            )
        )

    def validate_response(self, response, query):
        context = self.memory_manager.retrieve_context(query)
        context_str = json.dumps(context, indent=2)
        prompt = self.prompt_template.format(response=response, query=query, context=context_str)
        openrouter_validation = api_call({"type": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:free"}, prompt)
        cohere_validation = api_call({"type": "cohere"}, prompt)

        if "Error:" in openrouter_validation:
            gemini_validation = api_call({"type": "gemini"}, prompt)
            openrouter_validation = gemini_validation if not "Error:" in gemini_validation else "Error: Fallback to Gemini failed"
        
        result = {
            "openrouter_validation": openrouter_validation,
            "cohere_validation": cohere_validation,
            "combined": f"OpenRouter (LLaMA) Validation: {openrouter_validation}\nCohere Validation: {cohere_validation}"
        }
        self.memory_manager.store_episodic(query, result["combined"], ["validator_response"])
        return result

# Orchestrator
class Orchestrator:
    def __init__(self, clear_memory=False):
        self.memory_manager = MemoryManager(clear_memory=clear_memory)
        self.generator = Generator(self.memory_manager)
        self.critic = Critic(self.memory_manager)
        self.validator = Validator(self.memory_manager)
        self.classifier = QueryClassifier()
        self.convergence_checker = ConvergenceChecker()
        self.max_rounds = 3

        self.memory_manager.store_semantic(
            fact="AI systems can be built with modular architectures.",
            source="General Knowledge",
            tags=["ai", "system_design"]
        )
        self.memory_manager.store_semantic(
            fact="Debating AI agents require iterative refinement and validation.",
            source="General Knowledge",
            tags=["ai", "debate"]
        )

    def synthesize_response(self, current_response, critique, query):
        context = self.memory_manager.retrieve_context(query)
        context_str = json.dumps(context, indent=2)
        prompt = (
            f"For the query '{query}', synthesize this response and critique into a concise response in 50 words or less:\n"
            f"Current Response: {current_response}\n"
            f"Critique: {critique}\n"
            f"Context: {context_str}\n"
            "Address the critique's suggestions clearly and avoid verbosity."
        )
        return api_call({"type": "cohere"}, prompt)

    def run_debate(self, query):
        reasoning_log = []

        classification_output = self.classifier.classify(query)
        reasoning_log.append({
            "step": "Query Classification",
            "classification": classification_output["classification"],
            "reasoning": classification_output["reasoning"]
        })

        if classification_output["classification"] == "Non-debatable":
            context = self.memory_manager.retrieve_context(query)
            context_str = json.dumps(context, indent=2)
            prompt = f"Using the following context, provide a clear and concise answer to: {query} in 50 words or less.\nContext: {context_str}"
            cache_key = f"non_debatable:{query}"
            if cache_key in response_cache:
                answer = response_cache[cache_key]
            else:
                answer = api_call({"type": "gemini"}, prompt)
                response_cache[cache_key] = answer
            reasoning_log.append({
                "step": "Straightforward Answer",
                "answer": answer
            })
            self.memory_manager.store_episodic(query, answer, ["non_debatable_response"])
            final_response = (
                f"Final Response: {answer}\n"
                f"Classification Reasoning: {classification_output['reasoning']}"
            )
            return {
                "final_response": final_response,
                "reasoning_log": reasoning_log
            }

        gen_output = self.generator.generate_response(query)
        current_response = gen_output["combined"]
        reasoning_log.append({"step": "Initial Generation", "response": current_response})

        for round in range(self.max_rounds):
            critique_output = self.critic.critique_response(current_response, query)
            reasoning_log.append({
                "step": f"Critique Round {round + 1}",
                "critique": critique_output["combined"]
            })

            convergence_output = self.convergence_checker.check_convergence(
                critiques=critique_output["combined"],
                round=round + 1,
                query=query
            )
            reasoning_log.append({
                "step": f"Convergence Check Round {round + 1}",
                "decision": convergence_output["decision"],
                "reasoning": convergence_output["reasoning"]
            })

            if convergence_output["decision"] == "Converged":
                break

            current_response = self.synthesize_response(
                current_response, critique_output["combined"], query
            )
            reasoning_log.append({
                "step": f"Refined Response Round {round + 1}",
                "response": current_response
            })

        validation_output = self.validator.validate_response(current_response, query)
        reasoning_log.append({
            "step": "Final Validation",
            "validation": validation_output["combined"]
        })

        gemini_prompt = (
            f"Evaluate these validations for the query '{query}' and select the better one in 50 words or less:\n"
            f"Validation 1 (OpenRouter LLaMA): {validation_output['openrouter_validation']}\n"
            f"Validation 2 (Cohere): {validation_output['cohere_validation']}"
        )
        gemini_response = api_call({"type": "gemini"}, gemini_prompt)
        # Ensure single append for Gemini Selection
        reasoning_log.append({
            "step": "Gemini Selection",
            "gemini_response": gemini_response
        })

        chosen_validation = validation_output["cohere_validation"]
        gemini_reasoning = gemini_response
        if "Error:" in gemini_response:
            gemini_reasoning = f"Gemini failed: {gemini_response}. Defaulting to Cohere validation."
        else:
            if "Validation 1" in gemini_response or "OpenRouter" in gemini_response:
                chosen_validation = validation_output["openrouter_validation"]
            elif "Validation 2" in gemini_response or "Cohere" in gemini_response:
                chosen_validation = validation_output["cohere_validation"]

        self.memory_manager.store_episodic(query, chosen_validation, ["final_response"])
        final_response = (
            f"Final Response: {chosen_validation}\n"
            f"Gemini Reasoning: {gemini_reasoning}"
        )

        return {
            "final_response": final_response,
            "reasoning_log": reasoning_log
        }

# Interactive user input
if __name__ == "__main__":
    response_cache.clear()  # Clear cache at start
    orchestrator = Orchestrator(clear_memory=True)
    
    while True:
        query = input("Enter your debate query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            print("Exiting...")
            break
        if not query:
            print("No query provided. Please enter a valid query.")
            continue

        print(f"\nRunning debate for query: '{query}'...")
        result = orchestrator.run_debate(query)
        
        print("\nFinal Response:", result["final_response"])
        print("\nReasoning Log:")
        for log in result["reasoning_log"]:
            print(json.dumps(log, indent=2))
        print("\n" + "="*50 + "\n")