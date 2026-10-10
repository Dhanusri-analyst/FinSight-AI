import json
import os

BUDGET_FILE = "data/budget.json"


def save_budget(amount):

    with open(BUDGET_FILE, "w") as file:
        json.dump({"budget": amount}, file)



def get_budget():

    if os.path.exists(BUDGET_FILE):

        with open(BUDGET_FILE, "r") as file:
            data = json.load(file)
            return data["budget"]

    return 0