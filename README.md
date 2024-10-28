# Mini-Project using the ConvFinQA dataset

This project explores the ConvFinQA dataset.

It is forked from the original ConvFinQA repository, and contains the original code -- See README.ConvFinQA.md for more details.

It contains my own code in the `dan` folder, as well as a report in Report.md

To run the code:

1. Create a new python environment:

	python -m venv venv
	source venv/bin/activate

2. `pip install -r requirements.txt`

3. Copy .env.example to .env and set the variables.

4. Unzip data.zip

5. `pytest`

6. `python -m exploring.Eval`

This will run a small set of experiments, and create a file `eval.csv` with the results.
