
import json
import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import os

from openai import AzureOpenAI
from openai import OpenAI


# .env support
from dotenv import load_dotenv

from exploring.file_cache import get_from_cache, save_to_cache
load_dotenv()

LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY") 
OPENAI_ORG= os.getenv("OPENAI_ORG") 
OPENAI_PROJ= os.getenv("OPENAI_PROJ") 
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_VERSION = "2024-09-01-preview"

def openai_client():
	if "azure" in OPENAI_ENDPOINT:
		client = AzureOpenAI(
			api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION, azure_endpoint=OPENAI_ENDPOINT,
			organization=OPENAI_ORG, project=OPENAI_PROJ
		)
	else:
		client = OpenAI(
			api_key=OPENAI_API_KEY, organization=OPENAI_ORG, project=OPENAI_PROJ
		)	
	# Add tracing
	client = wrap_openai(client)
	return client


class QAAI:
	"""
	QAAI is a class for conversational question answering tasks, focusing on the ConvFinQA dataset.

	Attributes:
	- llm (str): 
	- compare_fn (str): "flexible"|"flexible-1sf"|"raw". The comparison function to use for evaluating answers. Defaults to "flexible".
	"""

	def __init__(self, llm: str = "openai:gpt-4o", compare_fn: str = "flexible"):
		self.llm = llm # The vendor:model of large language model to use
		self.compare_fn = compare_fn # The comparison function to use for evaluating answers. See eval_rig.py compare_answer() for details.
		self.cache_hit = 0
		self.cache_miss = 0
		self.save_correct_file = None # Set to track correct answers
		self.save_mistakes_file = None # Set to track mistakes

	# Note: Break the inputs out of the dict to avoid accidentally feeding any of the other fields (some of which contain part of the answer) to the LLM
	def get_prompt(self, pre_text: str|list[str], table:str|list[str], post_text: str|list[str]) -> str:
		with open("prompts/finqa-prompt.txt", "r") as f:
			prompt = f.read()
		# add input to prompt
		prompt = prompt +"\n\n" + str(pre_text) +"\n\n" + str(table) +"\n\n" + str(post_text)
		return prompt


	def do_answer_turn(self, input:dict, chat_history: list[str]) -> dict:
		"""
		Answer for the conversational QA task.
		chat_history: list of turns, alternating user and assistant, user first.
		"""
		if self.llm == "fixed":	# A fixed answer for testing without any LLM cost
			return {"answer": "A Suffusion of Yellow"}
			# (see https://www.urbandictionary.com/define.php?term=A+Suffusion+of+Yellow and enjoy)
		# openai?
		vendor, model = self.llm.split(":")
		if vendor == "openai":			
			# Build the prompt
			prompt = self.get_prompt(input.get("pre_text", ""), input.get("table", ""), input.get("post_text", ""))
			messages = [{"role": "system", "content": prompt}]
			# add the conversation-so-far	
			for i, turn in enumerate(chat_history):
				role = "user" if i % 2 == 0 else "assistant"
				messages.append({"role": role, "content": turn})
			# Call the LLM!
			answer = self.call_llm(model, messages)
			return {"answer": answer}
		# TODO: add support for other LLMs
		raise Exception("Unsupported LLM: " + self.llm)
		     

	def do_answer(self, input: dict, q: str | None) -> dict:
		"""
		Answer a list of questions.
		input: {pre_text, table, table_ori, post_text}
		qas: list of questions
		
		return: {"answer": str}
		Note: Return {} instead of a plain str, to allow for future extension to the richer outputs format of the ConvFinQA dataset.
		"""
		if self.llm == "fixed":	# A fixed answer for testing without any LLM cost
			return {"answer": "A Suffusion of Yellow"}
			# (see https://www.urbandictionary.com/define.php?term=A+Suffusion+of+Yellow and enjoy)
		# openai?
		vendor, model = self.llm.split(":")
		if vendor == "openai":			
			# Build the prompt
			prompt = self.get_prompt(input.get("pre_text", ""), input.get("table", ""), input.get("post_text", ""))
			messages = [{"role": "system", "content": prompt}]
			messages.append({"role": "user", "content": q})
			answer = self.call_llm(model, messages)
			return {"answer": answer}
		# TODO: add support for other LLMs
		raise Exception("Unsupported LLM: " + self.llm)


	def call_llm(self, model: str, messages: list[dict]) -> str:
		"""
		Call the LLM! -- with a file-based cache to speed up repeat runs and reduce costs.
		"""
		if not model: model = "gpt-4o"		
		params = {"model": model, "messages": messages, "max_tokens": 1024, "n": 1, "temperature": 0}   			
		# cache answers to reduce LLM costs
		answer = get_from_cache(params)
		if answer: 
			self.cache_hit += 1
			assert type(answer) == str
			return answer
		self.cache_miss += 1		
  		# Call the LLM!			
		client = openai_client()
		chat_completion = client.chat.completions.create(**params)
		ai_content = chat_completion.choices[0].message.content
		# Assume a json response with {"answer": ...}, and unwrap the json
		# remove possible ```json ``` (GPT sometimes returns JSON wrapped in markdown -- this could be tackled here or in the prompt. Changing the prompt would invalidate cached results)
		if ai_content.startswith("```json"):
			ai_content = ai_content[7:]
		if ai_content.endswith("```"):
			ai_content = ai_content[:-3]
		# trim whitespace
		ai_content = ai_content.strip()
		try:
			jobj = json.loads(ai_content)
			answer = str(jobj["answer"])
			save_to_cache(params, answer)
			return answer
		except:
			# A number not json?
			try:
				str_number = str(float(ai_content))
				save_to_cache(params, str_number)
				return str_number
			except:
				print(f"Warning: Failed to unwrap JSON: {ai_content}")				
				save_to_cache(params, ai_content) # cache it anyway
				return ai_content
