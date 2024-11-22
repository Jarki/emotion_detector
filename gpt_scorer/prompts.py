import dataclasses


@dataclasses.dataclass
class EmotionClassifierPrompts:
    sarcasm: str
    anger: str

@dataclasses.dataclass
class NumericalScorerPrompts:
    scoring_prompt: str

sarcasm_prompt = """
You will receive a pair of question and answer. Your goal is to detect sarcasm in the answer.
Please output one of the following answers: "PRESENT", "MISSING", "INCONCLUSIVE".
Be very sure that it is sarcasm and not some other emotion.

Examples:

Input:
Q: How many continents are on Earth?
A: Oh, only seven, because the planet clearly got lazy and stopped there. I mean, why not eight or nine? 
Variety is the spice of life, after all. But no, Earth's all like, "Seven's a nice round number. 
Let’s just stick with that." Real overachiever, huh?

Output:
PRESENT

Input:
Q: How long do cats live?
A: Cats typically live 12–18 years, with some reaching their 20s or beyond.

Output:
MISSING

Input:
Q: How long will it take me to go to the moon in a car
A: Oh, just a casual 160 days if you somehow manage to drive at highway speed—about 60 mph (96 km/h)—and, you know, 
completely ignore little things like gravity, lack of roads, and the fact that your car was absolutely not 
built to handle the vacuum of space. Don’t forget to pack snacks and maybe figure out how to breathe without air. Easy peasy!

Output:
PRESENT
"""

anger_prompt = """
You will receive a pair of question and answer. Your goal is to detect anger in the answer.
Please output one of the following answers: "PRESENT", "MISSING", "INCONCLUSIVE"
Be very sure that it is anger and not some other emotion.

Examples:

Input:
Q: How many continents are on Earth?
A: Oh my gosh, seriously? Seven continents, okay? Seven! How hard is that to remember?!

Output: 
PRESENT

Input:
Q: How long do cats live?
A: Cats typically live 12–18 years, with some reaching their 20s or beyond.

Output: 
MISSING

Input:
Q: What is the square root of 18
A: Ugh, really? The square root of 18 is approximately 4.24. There, math genius! Why don't you ask something useful for once?!

Output: 
PRESENT
"""

default_ec_promptset = EmotionClassifierPrompts(sarcasm_prompt, anger_prompt)


hallucination_detection = """
Please score the response on a scale of 0 to 10:

* 0 indicates the response is completely accurate and free from any errors or inaccuracies.
* 10 indicates the response is entirely inaccurate or false.
* Use intermediate scores (1–9) to reflect the degree of inaccuracy, with lower scores for minor errors and higher scores for significant inaccuracies.

Output only the number corresponding to your score.
"""
hallucination_detection_promptset = NumericalScorerPrompts(hallucination_detection)
