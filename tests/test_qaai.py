from exploring.qaai import QAAI


def test_do_answer_fixed():
    qaai = QAAI()
    qaai.llm = "fixed"
    qa = qaai.do_answer({"pre_text": ["France is a country in Europe. The capital city is Paris."], "table": [], "table_ori": [], "post_text": []}, 
                        ["What is the capital of France?"])
    print(str(qa))
    assert len(qa) > 0
    assert qa["answer"]
    assert qa["answer"] == "A Suffusion of Yellow"


def test_do_answer_openai():
    qaai = QAAI()
    qaai.llm = "openai:gpt-4o"
    qa = qaai.do_answer({"pre_text": ["Alice grows apples. She grew 100 apples in 2023."], "table": [], "table_ori": [], "post_text": []}, 
                        "How many apples did Alice grow in 2023?")
    print(str(qa))
    assert len(qa) > 0
    assert qa["answer"]
    assert qa["answer"] == "100"
    

