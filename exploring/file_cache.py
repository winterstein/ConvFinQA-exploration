from hashlib import md5
import json
import os

import dotenv
dotenv.load_dotenv()

CACHE_DIR = os.getenv("LLM_CACHE_DIR", "__llm_cache__")
if not os.path.exists(CACHE_DIR):
	os.makedirs(CACHE_DIR)

def cache_file(params):
	json_params = json.dumps(params)
	return CACHE_DIR + "/" + md5(json_params.encode()).hexdigest()+".json"

    
def get_from_cache(params):
	cfile = cache_file(params)
	if os.path.exists(cfile):
		# load the result from the cache
		with open(cfile, "r") as f:
			return json.load(f)	
	return None

def save_to_cache(params, result):
	cfile = cache_file(params)
	with open(cfile, "w") as f:
		json.dump(result, f)
