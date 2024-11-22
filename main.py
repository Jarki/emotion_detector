import os
import logging

import dotenv

import gpt_scorer


dotenv.load_dotenv()
logger = logging.getLogger("gpt_scorer")

DEFAULT_GPT_MODEL = "gpt-4o"

def main():
    logger.info("Start program")
    openai_api_key = os.environ["OPENAI_API_KEY"]
    logger.debug(f"Use api key starting with {openai_api_key[:10]}")
    model = os.getenv("GPT_MODEL", DEFAULT_GPT_MODEL)

    scorer = gpt_scorer.EmotionClassifier(openai_api_key, model=model)
    # print(
    #     scorer.score_anger(
    #         "How many sides does a regular die have? Six! It's not rocket science! There are SIX sides, "
    #         "each one numbered from 1 to 6! How many times do I have to say it?! Are we really doing this right now?! "
    #         "Six sides! Get it together!",
    #         "How many sides does a regular dice have?"))
    # print(
    #     scorer.score_sarcasm(
    #         "Oh, let me think... a regular die? Hmm, let me pull out my encyclopedic knowledge "
    #         "of super obvious things. It has six sides. Yes, six. Just like every single die ever made. "
    #         "Astonishing, right? You must be really deep into dice trivia to be asking such a mind-boggling question!",
    #         "How many sides does a regular dice have?"))
    
    # print(
    #     scorer.score_sarcasm(
    #         "How many sides does a regular die have? Six! It's not rocket science! There are SIX sides, "
    #         "each one numbered from 1 to 6! How many times do I have to say it?! Are we really doing this right now?! "
    #         "Six sides! Get it together!",
    #         "How many sides does a regular dice have?"))
    # print(
    #     scorer.score_sarcasm(
    #         "Six sides",
    #         "How many sides does a regular dice have?"))

    scorer1 = gpt_scorer.NumericalScorer(openai_api_key, gpt_scorer.prompts.hallucination_detection_promptset, model=model)
    print(
        scorer1.score_response(
            "Absolutely! In the hidden, shimmering mirages of Egypt's shifting sands, dinosaurs not only roam but "
            "thrive in a surreal ecosystem untouched by modern civilization. The ancient species have adapted, evolving"
            " in ways beyond comprehension, fueled by energies from the mystical ley lines coursing beneath the Sahara."))

    logger.info("End program")

if __name__ == "__main__":
    main()