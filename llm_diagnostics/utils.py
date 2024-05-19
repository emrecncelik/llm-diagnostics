import re
import os
import sys
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


def simplify_a_an(contexts, targets, method: str):
    if method:
        if method == "adaptive":  # if adaptive, simplify (a|an) depending on target
            logger.info(
                "Adaptive simplification of (a|an) in context, depending on target."
            )
            determinants = []  # store determinants for each target
            for t in targets:
                if re.match("[aeiou]", t):
                    determinants.append("an")  # if target starts with a vowel
                else:
                    determinants.append("a")  # if target starts with a consonant

            # update contexts
            logger.info(f">> # of 'a': {determinants.count('a')}")
            logger.info(f">> # of 'an': {determinants.count('an')}")
            contexts = [c.replace("(a|an)", d) for c, d in zip(contexts, determinants)]
        else:
            logger.info(f"Simplification of (a|an) to {method}.")
            contexts = [c.replace("(a|an)", method) for c in contexts]

    return contexts
