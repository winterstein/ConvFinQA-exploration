
from datetime import datetime
import json
import random
from exploring.eval_rig import load_data, run_evals, run_evals_turn
from exploring.qaai import QAAI
import os
DEFAULT_EXPERIMENT = {
	"file": "data/train.json",
	"TURN": False,
	"llm": "openai:gpt-4o-mini",
	"compare_fn": "flexible",
	"batch_size": 200
}

EXPERIMENTS = [
	{
		"name": "chat-turns gpt-4o-mini",
		"file": "data/train_turn.json",
		"TURN": True,
	},
 	# {
	# 	"name": "1-qa gpt-4o-mini raw",
  	# 	"compare_fn": "raw"
	# },
  	# {
	# 	"name": "1-qa gpt-4o-mini extra-flexible",
  	# 	"compare_fn": "flexible-1sf"
	# },
	# {
	# 	"name": "1-qa gpt-4o-mini",
	# },
 	# {
	# 	"name": "1-qa gpt-4o",
	# 	"llm": "openai:gpt-4o",
	# },
]



if __name__ == "__main__":
    random.shuffle(EXPERIMENTS) # randomize the order, so they all get run if the script is restarted
    for exp in EXPERIMENTS:
        exp = {**DEFAULT_EXPERIMENT, **exp}
        print(exp["name"])
        print(str(exp))
        ai = QAAI()
        ai.llm = exp["llm"]
        ai.compare_fn = exp["compare_fn"]
        ai.save_correct_file = exp["name"].replace(" ", "_") + "-correct.jsonl"
        ai.save_mistakes_file = exp["name"].replace(" ", "_") + "-mistakes.jsonl"
        # delete any existing files
        # Note: caching means re-runs are fast, so we can delete final output files without loss of efficiency
        if os.path.exists(ai.save_correct_file):
            os.remove(ai.save_correct_file)
        if os.path.exists(ai.save_mistakes_file):
            os.remove(ai.save_mistakes_file)
        file = exp["file"]
        data = load_data(file)
        batch_size = exp["batch_size"]
        # random sample of batch_size
        random.seed(42) # Set the seed for repeatable slicing, to make better use of the cache 
        selected_data = random.sample(data, batch_size)
        
        selected_data = list(filter(lambda d: d.get("id") == "Single_ETR/2017/page_441.pdf-4_0", selected_data))
        
        # Run the evaluation!
        if exp["TURN"]:
            score = run_evals_turn(ai, selected_data)
        else:
            score = run_evals(ai, selected_data)
        print(exp["name"] + ": " + str(score))
        # append to csv
        with open("eval.csv", "a") as f:
            today = datetime.now().strftime("%Y-%m-%d")
            f.write(f"{exp['name']},{today},{score},{json.dumps(exp)}\n")
        