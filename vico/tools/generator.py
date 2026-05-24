import json
import logging
import random
import time

import backoff
import os
import base64
from PIL import Image
from io import BytesIO
from typing import Union
import traceback
import pickle

def encode_image(img: Union[str, Image.Image]) -> str:
	if isinstance(img, str): # if it's image path, open and then encode/decode
		with open(img, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')
	elif isinstance(img, Image.Image): # if it's image already, buffer and then encode/decode
		buffered = BytesIO()
		img.save(buffered, format="JPEG")
		return base64.b64encode(buffered.getvalue()).decode("utf-8")
	else:
		raise Exception("img can only be either str or Image.Image")

class Generator:
	def __init__(self, lm_source, lm_id, max_tokens=4096, temperature=0.7, top_p=1.0, logger=None):
		self.lm_source = lm_source
		self.lm_id = lm_id
		self.max_tokens = max_tokens
		self.temperature = temperature
		self.top_p = top_p
		self.logger = logger
		self.caller_analysis = {}
		if self.logger is None:
			self.logger = logging.getLogger(__name__)
			self.logger.setLevel(logging.DEBUG)
			self.logger.addHandler(logging.StreamHandler())
		self.max_retries = 3
		self.cost = 0 # cost in us dollars
		self.cache_path = f"cache_{lm_id}.pkl"
		if os.path.exists(self.cache_path):
			with open(self.cache_path, 'rb') as f:
				self.cache = pickle.load(f)
		else:
			self.cache = {}
		if self.lm_id == "text-embedding-3-small":
			self.embedding_dim = 1536
		elif self.lm_id == "text-embedding-3-large":
			self.embedding_dim = 3072
		else:
			self.embedding_dim = 0
		if self.lm_id == "gpt-4o":
			self.input_token_price = 2.5 * 10 ** -6
			self.output_token_price = 10 * 10 ** -6
		elif self.lm_id == "gpt-4.1":
			self.input_token_price = 2 * 10 ** -6
			self.output_token_price = 8 * 10 ** -6
		elif self.lm_id == "o3-mini" or self.lm_id == "o4-mini":
			self.input_token_price = 1.1 * 10 ** -6
			self.output_token_price = 4.4 * 10 ** -6
		elif self.lm_id == "gpt-35-turbo":
			self.input_token_price = 1 * 10 ** -6
			self.output_token_price = 2 * 10 ** -6
		else:
			self.input_token_price = -1 * 10 ** -6
			self.output_token_price = -2 * 10 ** -6
		if self.lm_source == "openai":
			from openai import OpenAI
			self.client = OpenAI(
				api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
				max_retries=self.max_retries,
			) if 'OPENAI_API_KEY' in os.environ else None
		elif self.lm_source == "azure":
			from openai import AzureOpenAI
			try:
				api_keys = json.load(open(".api_keys.json", "r"))
				if "embedding" in self.lm_id:
					api_keys = api_keys["embedding"]
				else:
					api_keys = api_keys["all"]
				api_keys = random.sample(api_keys, 1)[0]
				self.logger.info(f"Using Azure API key: {api_keys['AZURE_ENDPOINT']}")
				self.client = AzureOpenAI(
					azure_endpoint=api_keys['AZURE_ENDPOINT'],
					api_key=api_keys['OPENAI_API_KEY'],
					api_version="2024-12-01-preview",
				)
			except Exception as e:
				self.logger.error(f"Error loading .api_keys.json: {e} with traceback: {traceback.format_exc()}")
				self.client = None
		else:
			raise NotImplementedError(f"{self.lm_source} is not supported!")

	def generate(self, prompt, max_tokens=None, temperature=None, top_p=None, img=None, json_mode=False, chat_history=None, caller="none"):
		if max_tokens is None:
			max_tokens = self.max_tokens
		if temperature is None:
			temperature = self.temperature
		if top_p is None:
			top_p = self.top_p
			
		if self.lm_source == 'openai' or self.lm_source == 'azure':
			return self.openai_generate(prompt, max_tokens, temperature, top_p, img, json_mode, chat_history, caller)
		else:
			raise ValueError(f"Invalid lm_source: {self.lm_source}")

	def openai_generate(self, prompt, max_tokens, temperature, top_p, img: Union[str, Image.Image, None, list], json_mode=False, chat_history=None, caller="none"):
		@backoff.on_exception(
			backoff.expo,  # Exponential backoff
			Exception,  # Base exception to catch and retry on
			max_tries=self.max_retries,  # Maximum number of retries
			jitter=backoff.full_jitter,  # Add full jitter to the backoff
			logger=self.logger  # Logger for retry events, which is in the level of INFO
		)
		def _generate():
			content = [{
						 "type": "text",
						 "text": prompt
					 }, ]
			if img is not None:
				if type(img) != list:
					imgs = [img]
				else:
					imgs = img
				for each_img in imgs:
					content.append({
						"type": "image_url",
						"image_url": {"url": f"data:image/png;base64,{encode_image(each_img)}"},
						# "detail": "low"
					})
			if chat_history is not None:
				messages = chat_history
			else:
				messages = []
			messages.append(
				{
					"role": "user",
					"content": content
				})
			start = time.perf_counter()
			if self.lm_id[0] == 'o':
				params = {
					"reasoning_effort": "high",
					"timeout": 400,
				}
			else:
				params = {
					"temperature": temperature,
					"top_p": top_p,
					"timeout": 40,
				}
			response = self.client.chat.completions.create(
					model=self.lm_id,
					messages=messages,
					max_completion_tokens=max_tokens,
					response_format={
						"type": "json_object" if json_mode else "text"
					},
					**params,
				)
			self.logger.debug(f"api request time: {time.perf_counter() - start}")
			with open(f"chat_raw.jsonl", 'a') as f:
				chat_entry = {
					"prompt": prompt,
					"response": response.model_dump_json(indent=4)
				}
				# Write as a single JSON object per line
				f.write(json.dumps(chat_entry))
				f.write('\n')
			usage = dict(response.usage)
			self.cost += usage['completion_tokens'] * self.output_token_price + usage['prompt_tokens'] * self.input_token_price
			if caller in self.caller_analysis:
				self.caller_analysis[caller].append(usage['total_tokens'])
			else:
				self.caller_analysis[caller] = [usage['total_tokens']]
			response = response.choices[0].message.content
			# self.logger.debug(f'======= prompt ======= \n{prompt}', )
			# self.logger.debug(f'======= response ======= \n{response}')
			# self.logger.debug(f'======= usage ======= \n{usage}')
			if self.cost > 7:
				self.logger.critical(f'COST ABOVE 7 dollars! There must be sth wrong. Stop the exp immediately!')
				raise Exception(f'COST ABOVE 7 dollars! There must be sth wrong. Stop the exp immediately!')
			self.logger.info(f'======= total cost ======= {self.cost}')
			return response
		try:
			return _generate()
		except Exception as e:
			self.logger.error(f"Error with openai_generate: {e}, the prompt was:\n {prompt}")
			return ""

	def get_embedding(self, text, caller="none"):
		@backoff.on_exception(
			backoff.expo,  # Exponential backoff
			Exception,  # Base exception to catch and retry on
			max_tries=self.max_retries,  # Maximum number of retries
			jitter=backoff.full_jitter,  # Add full jitter to the backoff
			logger=self.logger  # Logger for retry events
		)
		def _embed() -> list:
			response = self.client.embeddings.create(
				model=self.lm_id,
				input=[text]
			)
			usage = dict(response.usage)
			if caller in self.caller_analysis:
				self.caller_analysis[caller].append(usage['total_tokens'])
			else:
				self.caller_analysis[caller] = [usage['total_tokens']]
			return response.data[0].embedding

		if text in self.cache:
			return self.cache[text]

		embedding = _embed()
		self.cache[text] = embedding
		return embedding


if __name__ == "__main__":
	generator = Generator(
		lm_source='azure',
		lm_id='o4-mini',
		max_tokens=4096,
		temperature=0.7,
		top_p=1.0,
		logger=None
	)
	prompt1 = "What is the meaning of life?"
	print(generator.generate(prompt1))