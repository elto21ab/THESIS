from typing import List, Dict
import pandas as pd

def validateMatchingSurvey(surv, simulation_questions):
    ver = pd.DataFrame(surv.questions)
    ver["simulation"] = simulation_questions
    ver["is_same"] = ver["question"] == ver["simulation"]
    if ver["is_same"].all():
        return True
    else:
        print("CRITICAL WARNING: Some questions are different. Inspect the DF")
        return ver

class KanoSurvey:

    # Static Variables
    POSSIBLE_ANSWERS = ["I LIKE IT", "I EXPECT IT", "I AM NEUTRAL", "I CAN TOLERATE IT", "I DISLIKE IT"]
    _defaultCSV = "surveys/survey_kano-model.csv"

    # Instance Variables
    questions: List[str]
    test_answers: Dict[str, List[str]]
    df: pd.DataFrame

    def __init__(self, csv_file = _defaultCSV):
        if csv_file == KanoSurvey._defaultCSV:
            print(f"Using default Kano Survey CSV file: {KanoSurvey._defaultCSV}")

        # Parse Survey questions
        df = pd.read_csv(csv_file)
        df = df[df["type"] != 0]
        self.questions = df["question"]

        # Parse Test answers
        self.test_answers = {}
        try:
            self.test_answers["airidas"] = list(df["airidas"])
        except:
            print("No airidas answers column found in the csv file")
        try:
            self.test_answers["elias"] = list(df["elias"])
        except:
            print("No elias answers column found in the csv file")

        self.df = df 
        

        
class PersonalitySurvey:
    # Static Variables
    POSSIBLE_ANSWERS = ["DISAGREE", "SOMEWHAT DISAGREE", "NEUTRAL", "SOMEWHAT AGREE", "AGREE"]
    _defaultCSV = "surveys/survey_personality-test.csv"

    # Instance Variables
    questions: List[str]
    test_answers: Dict[str, List[str]]
    df: pd.DataFrame

    def __init__(self, csv_file=_defaultCSV):
        if csv_file == PersonalitySurvey._defaultCSV:
            print(f"Using default Personality Survey CSV file: {PersonalitySurvey._defaultCSV}")

        # Parse Survey questions
        surv = pd.read_csv(csv_file)
        self.questions = surv["question"]

        # Parse Test answers
        self.test_answers = {}
        try:
            self.test_answers["airidas"] = list(surv["airidas"])
        except:
            print("No airidas answers column found in the csv file")
        try:
            self.test_answers["elias"] = list(surv["elias"])
        except:
            print("No elias answers column found in the csv file")

        self.df = surv 

class DictatorGameSurvey:
    
    # Static Variables
    POSSIBLE_ANSWERS = ["LEFT", "RIGHT"]
    _defaultCSV = "surveys/survey_dictator-game.csv"

    # Instance Variables
    questions: List[str]
    test_answers: Dict[str, List[str]]
    df: pd.DataFrame

    def __init__(self, csv_file = _defaultCSV):
        if csv_file == DictatorGameSurvey._defaultCSV:
            print(f"Using default Dictator Game Survey CSV file: {DictatorGameSurvey._defaultCSV}")

        # Parse Survey questions
        surv = pd.read_csv(csv_file)
        self.questions = surv["question"]

        # Parse Test answers
        self.test_answers = {}
        try:
            self.test_answers["airidas"] = list(surv["airidas"])
        except:
            print("No airidas answers column found in the csv file")
        try:
            self.test_answers["elias"] = list(surv["elias"])
        except:
            print("No elias answers column found in the csv file")

        self.df = surv 

class FairnessSurvey:
        # Static Variables
        POSSIBLE_ANSWERS = ["COMPLETELY FAIR", "ACCEPTABLE", "UNFAIR", "VERY UNFAIR"]
        _defaultCSV = "surveys/survey_fairness.csv"

        # Instance Variables
        questions: List[str]
        test_answers: Dict[str, List[str]]
        df: pd.DataFrame
    
        def __init__(self, csv_file=_defaultCSV):
            if csv_file == FairnessSurvey._defaultCSV:
                print(f"Using default Fairness Survey CSV file: {FairnessSurvey._defaultCSV}")

            # Parse Survey questions
            surv = pd.read_csv(csv_file)
            surv = surv[surv["framing"] == "Change"]
            self.questions = surv["question"]
    
            # Parse Test answers
            self.test_answers = {}
            try:
                self.test_answers["airidas"] = list(surv["airidas"])
            except:
                print("No airidas answers column found in the csv file")
            try:
                self.test_answers["elias"] = list(surv["elias"])
            except:
                print("No elias answers column found in the csv file")
    
            self.df = surv

