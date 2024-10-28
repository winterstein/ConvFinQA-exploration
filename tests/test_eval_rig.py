import json
from exploring.qaai import QAAI
from exploring.eval_rig import answers_match, canon_answer, eval_one_turn, load_data, round_sig_figs, run_evals, sniff_schema
import os

def test_round_sig_figs():
    assert round_sig_figs(0.00123, 2) == 0.0012
    assert round_sig_figs(0.00123, 3) == 0.00123
    assert round_sig_figs(0.00123, 1) == 0.001
    assert round_sig_figs(3450.00123, 2) == 3500
    assert round_sig_figs(1.27, 2) == 1.3
    assert round_sig_figs(9.999, 2) == 10
    assert round_sig_figs(11, 2) == 11
    assert round_sig_figs(10.001, 2) == 10
    assert round_sig_figs(10, 2) == 10
    assert round_sig_figs(12, 2) == 12
    assert round_sig_figs(-12, 2) == -12
    assert round_sig_figs(-0.00123, 2) == -0.0012
    assert round_sig_figs(0, 2) == 0
    

def test_canon_answer():
    assert canon_answer("10%") == 0.1
    assert canon_answer("-1.5%") == -0.015
    assert canon_answer("-8.9%") == -0.089
    assert canon_answer("10") == 10
    assert canon_answer("10 millions") == 10
    assert canon_answer("1.5 millions") == 1.5
    assert canon_answer("nope") == None
    
def test_answers_match():
    assert answers_match("1.27%", "1.3%")
    assert answers_match("-8.9%", "-8.94%")
    assert not answers_match("1.27%", "1.2%")
    assert not answers_match("10", "12")
    
def test_load_data():
    files = os.listdir("data")
    print(str(files))
    for infile in files:
        input_data = load_data("data/" + infile)
        print(infile + ": " + str(len(input_data)))
        datum0 = input_data[0]
        assert len(datum0["pre_text"]) > 0


def test_sniff_schema():
    devfile = "data/dev.json"
    # devfile = "data/train.json"
    d0 = load_data(devfile)[0]
    schema = sniff_schema(d0)
    print(json.dumps(schema))
    assert schema["table"]
    
    
def test_pluck_datum():
	"""A utility to pull a datum by id (copy the output for use in e.g. broswer console to explore it)"""
	id = "Single_ETR/2017/page_441.pdf-4_0"
	# id = "Single_JKHY/2009/page_28.pdf-3_2"
	devfile = "data/train_turn.json"
	data = load_data(devfile)
	d = list(filter(lambda d: d.get("id") == id, data))[0]
	print(json.dumps(d))
	# assert False
 
    
def test_sniff_schema_turn():
	devfile = "data/train_turn.json"
	data = load_data(devfile)
	d0 = data[0]
	schema = sniff_schema(d0)
	print(json.dumps(schema))
	assert schema["table"]
	print(json.dumps(d0))
	# turn_ind?
	turn_ind_cnt = 0
	for d in data:
		if d["annotation"].get("turn_ind"):
			turn_ind_cnt += 1
		else:
			print(d["id"])
			print(sniff_schema(d["annotation"]))
			print(d["id"])
	print(f"turn_ind_cnt: {turn_ind_cnt} / {len(data)}")
	assert turn_ind_cnt < len(data)
	assert turn_ind_cnt >  0.5 * len(data)


def test_sniff_schema_turn_test():
	devfile = "data/test_turn_private.json"
	data = load_data(devfile)
	for d in data:
		print(f"{d['id']}:	{len(d.get('cur_dial', []))}")
		if d['id'] == "Double_UNP/2007/page_25.pdf_3":
			print(str(d))


def test_small_fixed():
    data = load_data("data/train.json")
    small_data = data[0:5]
    print(str(sniff_schema(small_data[0])))
    ai = QAAI()
    ai.llm = "fixed"
    scores = run_evals(ai, small_data)
    print(str(scores))    


def test_eval_one_turn():
    data = load_data("data/train_turn.json")
    ai = QAAI()
    score = eval_one_turn(ai, data[0])
    assert score == 1
    score1 = eval_one_turn(ai, data[1])
    assert score1 == 1