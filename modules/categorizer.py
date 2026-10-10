import joblib
import os
import re


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "models/expense_model.pkl"


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

if os.path.exists(MODEL_PATH):

    try:

        model = joblib.load(MODEL_PATH)

        print(
            "✅ FinSight ML categorizer loaded"
        )

    except Exception as e:

        model = None

        print(
            "⚠️ Could not load ML model:",
            e
        )

else:

    model = None

    print(
        "⚠️ ML model not found. "
        "Using fallback rules."
    )


# ============================================================
# CATEGORY PREDICTION
# ============================================================

def predict_category(description):

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if description is None:
        return "Others"

    text = str(
        description
    ).strip().lower()

    if not text:
        return "Others"

    # Remove excessive spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Reject input containing no alphabetic characters
    if not re.search(
        r"[a-zA-Z]",
        text
    ):
        return "Others"


    # ========================================================
    # 1. GROCERIES
    # ========================================================
    # This MUST come before the ML model.
    #
    # Supermarket receipts often contain many food/grocery
    # items. OCR may produce an imperfect merchant name, so
    # we use several high-confidence grocery indicators.
    # ========================================================

    grocery_keywords = [

        # Stores / merchants
        "supermarket",
        "super market",
        "grocery",
        "groceries",
        "mart",
        "fresh market",
        "freshmart",
        "food mart",
        "hypermarket",

        # Fruits
        "apple",
        "apples",
        "banana",
        "bananas",
        "orange",
        "oranges",
        "grapes",
        "grape",
        "mango",
        "mangoes",
        "papaya",
        "watermelon",
        "melon",
        "pineapple",
        "strawberry",
        "strawberries",
        "blueberry",
        "blueberries",
        "kiwi",
        "lemon",
        "lemons",
        "lime",
        "pear",
        "pears",

        # Vegetables
        "vegetable",
        "vegetables",
        "tomato",
        "tomatoes",
        "potato",
        "potatoes",
        "onion",
        "onions",
        "carrot",
        "carrots",
        "cabbage",
        "spinach",
        "beans",
        "peas",
        "broccoli",
        "cauliflower",
        "cucumber",
        "garlic",
        "ginger",

        # Dairy
        "milk",
        "curd",
        "yogurt",
        "yoghurt",
        "cheese",
        "butter",
        "cream",
        "paneer",

        # Grocery staples
        "rice",
        "wheat",
        "flour",
        "atta",
        "sugar",
        "salt",
        "oil",
        "cooking oil",
        "dal",
        "dhal",
        "lentils",
        "pulses",
        "bread",
        "biscuits",
        "biscuit",
        "cereal",
        "oats",

        # Eggs / meat / food ingredients
        "eggs",
        "egg",
        "chicken",
        "meat",
        "fish",
        "prawns",
        "seafood",

        # Household grocery items
        "detergent",
        "soap",
        "shampoo",
        "toothpaste",
        "tissue",
        "cleaning",
        "dishwash"
    ]


    grocery_score = 0

    for keyword in grocery_keywords:

        if keyword in text:

            grocery_score += 1


    # Strong grocery signal
    if grocery_score >= 1:

        # Supermarket itself is a very strong signal
        if any(
            keyword in text
            for keyword in [
                "supermarket",
                "super market",
                "grocery",
                "groceries",
                "hypermarket",
                "fresh market"
            ]
        ):

            return "Groceries"

        # Multiple grocery products
        if grocery_score >= 2:

            return "Groceries"


    # ========================================================
    # 2. TRAVEL
    # ========================================================

    if any(
        word in text
        for word in [

            "uber",
            "ola",
            "taxi",
            "cab",
            "bus",
            "train",
            "metro",
            "flight",
            "airport",
            "trip",
            "travel",
            "vacation",
            "holiday",
            "hotel",
            "resort",
            "tour",
            "sightseeing",
            "airbnb",
            "fuel",
            "petrol",
            "diesel",
            "parking"
        ]
    ):

        return "Travel"


    # ========================================================
    # 3. UTILITIES
    # ========================================================

    if any(
        word in text
        for word in [

            "electricity",
            "electricity bill",
            "water bill",
            "water",
            "internet",
            "wifi",
            "wi-fi",
            "mobile bill",
            "phone bill",
            "gas bill",
            "utility",
            "utilities",
            "recharge",
            "broadband"
        ]
    ):

        return "Utilities"


    # ========================================================
    # 4. SHOPPING
    # ========================================================

    if any(
        word in text
        for word in [

            "amazon",
            "flipkart",
            "shopping",
            "mall",
            "laptop",
            "computer",
            "pc",
            "mobile",
            "smartphone",
            "phone",
            "iphone",
            "android",
            "headphones",
            "earphones",
            "shoes",
            "clothes",
            "dress",
            "shirt",
            "jeans",
            "electronics",
            "furniture",
            "appliance",
            "watch",
            "bag",
            "handbag"
        ]
    ):

        return "Shopping"


    # ========================================================
    # 5. HEALTH
    # ========================================================

    if any(
        word in text
        for word in [

            "doctor",
            "hospital",
            "medical",
            "pharmacy",
            "medicine",
            "medicines",
            "health",
            "dental",
            "dentist",
            "clinic",
            "checkup",
            "treatment",
            "diagnostic",
            "lab test",
            "blood test"
        ]
    ):

        return "Health"


    # ========================================================
    # 6. EDUCATION
    # ========================================================

    if any(
        word in text
        for word in [

            "school",
            "college",
            "university",
            "tuition",
            "education",
            "course",
            "exam",
            "semester",
            "certification",
            "textbook",
            "study",
            "fees",
            "college fees",
            "school fees"
        ]
    ):

        return "Education"


    # ========================================================
    # 7. RENT
    # ========================================================

    if any(
        word in text
        for word in [

            "rent",
            "rental",
            "apartment",
            "house rent",
            "room rent",
            "lease",
            "landlord"
        ]
    ):

        return "Rent"


    # ========================================================
    # 8. ENTERTAINMENT
    # ========================================================

    if any(
        word in text
        for word in [

            "movie",
            "cinema",
            "concert",
            "gaming",
            "game",
            "netflix",
            "spotify",
            "music",
            "entertainment",
            "theatre",
            "theater"
        ]
    ):

        return "Entertainment"


    # ========================================================
    # 9. FOOD
    # ========================================================
    # Food = prepared meals / restaurants.
    # Groceries are handled separately above.
    # ========================================================

    if any(
        word in text
        for word in [

            "swiggy",
            "zomato",
            "food",
            "pizza",
            "burger",
            "restaurant",
            "lunch",
            "dinner",
            "breakfast",
            "coffee",
            "cafe",
            "café",
            "tea",
            "snacks",
            "snack",
            "bakery",
            "bakery shop"
        ]
    ):

        return "Food"


    # ========================================================
    # 10. SALARY
    # ========================================================

    if any(
        word in text
        for word in [

            "salary",
            "paycheck",
            "pay cheque",
            "monthly salary",
            "income"
        ]
    ):

        return "Salary"


    # ========================================================
    # 11. FREELANCE
    # ========================================================

    if any(
        word in text
        for word in [

            "freelance",
            "freelancing",
            "client payment",
            "freelance payment"
        ]
    ):

        return "Freelance"


    # ========================================================
    # 12. INVESTMENT
    # ========================================================

    if any(
        word in text
        for word in [

            "investment",
            "mutual fund",
            "mutual funds",
            "stock",
            "stocks",
            "shares",
            "share market",
            "sip",
            "stocks investment"
        ]
    ):

        return "Investment"


    # ========================================================
    # 13. BONUS
    # ========================================================

    if any(
        word in text
        for word in [

            "bonus",
            "performance bonus",
            "annual bonus"
        ]
    ):

        return "Bonus"


    # ========================================================
    # 14. SAVINGS
    # ========================================================

    if any(
        word in text
        for word in [

            "saving",
            "savings",
            "savings account",
            "save money"
        ]
    ):

        return "Savings"


    # ========================================================
    # 15. MACHINE LEARNING MODEL
    # ========================================================

    if model is not None:

        try:

            prediction = model.predict(
                [description]
            )[0]

            return prediction

        except Exception as e:

            print(
                "⚠️ ML prediction failed:",
                e
            )


    # ========================================================
    # 16. DEFAULT
    # ========================================================

    return "Others"