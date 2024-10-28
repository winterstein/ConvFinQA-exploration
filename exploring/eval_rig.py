# eval_rig.py
# Rig for evaluating the AI on the dataset. This file handles data loading, the ConvFinQA formats, and drives evaluation.
# ConvFinQA contains a couple of related formats.

import json
import math

from tqdm import tqdm
from exploring.qaai import QAAI
import re

def load_data(file: str) -> list[dict]:
    # assume: we can read the dataset into memory (it's not big)
    with open(file, "r") as f:
        input_data = json.load(f)
        return input_data


def run_evals(ai: QAAI, data: list[dict]) -> float:
    total_score = 0
    cnt = 0
    for datum in tqdm(data):
        score = eval_one(ai, datum)
        total_score += score	
        cnt += 1
        print(f"Score: {total_score} / {cnt} = {total_score / cnt}	Caching: hit:{ai.cache_hit} v miss:{ai.cache_miss}")
    return total_score / len(data)


def run_evals_turn(ai: QAAI, data: list[dict]) -> float:
    total_score = 0
    cnt = 0
    for datum in tqdm(data):
        score = eval_one_turn(ai, datum)
        total_score += score	
        cnt += 1
        print(f"Score: {total_score} / {cnt} = {total_score / cnt}	Caching: hit:{ai.cache_hit} v miss:{ai.cache_miss}")
    return total_score / len(data)


def eval_one(ai: QAAI, datum: dict) -> float:
	# remove the answer
	datum = datum.copy()
	qa = datum.get("qa")
	if qa:
		datum["qa"] = qa.copy()
		# remove the answer
		datum["qa"]["answer"] = None
		qas = [qa]
	else:
		# Type II: "complex" two-question
		qa_0 = datum["qa_0"]
		qa_1 = datum["qa_1"]
		qas = [qa_0, qa_1]
		# remove the answers
		datum["qa_0"] = qa_0.copy()
		datum["qa_1"] = qa_1.copy()
		datum["qa_0"]["answer"] = None
		datum["qa_1"]["answer"] = None
	# Use the AI
	gen_answers = [ai.do_answer(datum, qa["question"]) for qa in qas]
	# Compare and score: version 1: 0 / 1 binary comparison
	score = 0
	for i in range(len(qas)):
		gen_answeri = gen_answers[i]["answer"]
		target_answeri = qas[i]["answer"]                
		if answers_match(gen_answeri, target_answeri, ai.compare_fn):
			score += 1
			if ai.save_correct_file:
				# append to the save_correct_file
				with open(ai.save_correct_file, "a") as f:
					f.write(json.dumps({"id": datum["id"], "question": qas[i]["question"], "target": target_answeri, "gen": gen_answeri}) + "\n")
		else:
			print(f"Mismatch: {datum['id']} {qas[i]['question']}: target {target_answeri} != GenAI {gen_answeri}")
			if ai.save_mistakes_file:
				# append to the save_mistakes_file
				with open(ai.save_mistakes_file, "a") as f:
					f.write(json.dumps({"id": datum["id"], "question": qas[i]["question"], "target": target_answeri, "gen": gen_answeri}) + "\n")
	return score / len(qas)




def eval_one_turn(ai: QAAI, datum: dict) -> float:
	"""Evaluate one entry of a turn-based series of Q&A from the ConvFinQA dataset."""
	annotation = datum["annotation"]
	turn_ind = annotation.get("turn_ind", 0)
	cur_dial = annotation["cur_dial"]
	exe_ans_list = annotation["exe_ans_list"]
	chat_history = []
	for i in range(turn_ind+1):
		useri = cur_dial[i]
		assi = str(exe_ans_list[i]) # number to string
		chat_history.append(useri)        
		chat_history.append(assi)
  
	print(f"id: {datum['id']}")
	print(f"chat_history: {chat_history}")
	target_answer = chat_history[-1]
	print(f"target_answer: {target_answer}")	
	chat_history = chat_history[:-1] # drop the last one, which we want the AI to answer
	# # Use the AI
	ai_response = ai.do_answer_turn(datum, chat_history)    
	ai_answer = ai_response["answer"]
	print(f"ai_answer: {ai_answer}")
	# Compare and score
	ok = answers_match(ai_answer, target_answer, ai.compare_fn)        
	if ok:
		score = 1
		if ai.save_correct_file:
			# append to the save_correct_file
			with open(ai.save_correct_file, "a") as f:
				f.write(json.dumps({"id": datum["id"], "question":cur_dial[-1], "target": target_answer, "gen": ai_answer}) + "\n")
	else:
		score = 0
		if ai.save_mistakes_file:
			# append to the save_mistakes_file
			with open(ai.save_mistakes_file, "a") as f:
				f.write(json.dumps({"id": datum["id"], "question":cur_dial[-1], "target": target_answer, "gen": ai_answer}) + "\n")
	return score


def answers_match(a:str,b:str, compare_fn:str = "flexible"):
    if compare_fn == "raw":
        return a == b
    sig_figs = 2
    if compare_fn == "flexible-1sf": sig_figs = 1
    a2 = canon_answer(a, sig_figs)
    b2 = canon_answer(b, sig_figs)
    if a2 == b2:
        return True
    # allow "10%" = 0.1 (the canon_answer value) or 10
    if a.endswith("%") and not b.endswith("%"):
        if a2*100 == b2: return True
    if b.endswith("%") and not a.endswith("%"):
        if b2*100 == a2: return True
    return False


def canon_answer(x:str|float|int, sig_figs:int = 2) -> float|int|None:
	"""Convert answers to a canonical rounded-number form for comparison. See notes in Report.md on why we round to two significant figures."""
	try:
		if type(x) == float or type(x) == int:
			v = round_sig_figs(x, 2)
			return v
		assert type(x) == str
		# remove any £/$ etc
		x = re.sub(r'[£\$€¥]', "", x)
		# remove any commas 
		x = re.sub(r',', "", x)
		# any trailling .
		if x[-1] == ".": x = x[:-1]
		try: # number or bust. Target answers should always convert to numbers - GenAI might include words and fail
			# Strip % and convert to a fraction
			if "%" in x:
				x = x.replace("%", "")
				v = float(x) / 100
			else:			
				v = float(x)
			# Round to two significant figures
			return round_sig_figs(v, sig_figs)
			return v
		except:
			# extract the number, e.g. "10 millions" -> 10
			match = re.search(r'[0-9][\.0-9]*', x)
			if match:
				v = float(match.group())
				return round_sig_figs(v, sig_figs)
			return None
	except Exception as e:
		print(f"Failed to canonize {x}: "+str(e))
		return None

def round_sig_figs(v:float, sig_figs:int):
	if v == 0:
		return 0
	# exponent of 10 for the number
	v10 = int(math.floor(math.log10(abs(v))))
	# round with a +ive does decimal places, and with a -ive it drops digits
	return round(v, sig_figs - v10 - 1)

def sniff_schema(obj):
    """HACK: sniff the schema from an object. Handy for exploring a dataset"""
    if type(obj) == dict:
        schema = {}
        for k in obj.keys():
            schema[k] = sniff_schema(obj[k])
        return schema
    if type(obj) == list:
        if len(obj) == 0:
            return []
        return [sniff_schema(obj[0])]
    if type(obj) == str:
        return "string"
    if type(obj) == int:
        return "int"
    if type(obj) == float:
        return "float"
    return type(obj)
