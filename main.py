import pdb
import os
import sys
import time
from utils import *

import numpy as np


class Examiner:
    def __init__(self, questions):

        self.questions_unanswered = questions
        self.questions_answered = []
        self.cum_time = 0
        self.n_questions_total = len(questions)

    def clear(self):
        """
        Clear console.
        """
        os.system("clear")

    def get_progress(self):
        n_answered = len(self.questions_answered)
        perc = n_answered / self.n_questions_total * 100
        out = f"{perc:.2f}%   " + get_progress_bar_string(perc)
        out += f"\nQuestions answered: {n_answered}"
        out += f"\nQuestions in total: {self.n_questions_total}"
        out += "\nType 'save' anytime to save and exit the exam."
        return out

    def print_overall_info(self):
        self.clear()
        print("\nCurrent progress:", self.get_progress(), "\n")

    def prompt_question(self, question):
        start_txt = f"\nQuestion from {question.section}"
        if not question.subsection is None:
            start_txt += f", {question.subsection}"
        if not question.subsubsection is None:
            start_txt += f", {question.subsubsection}"
        print(start_txt + "\n")
        print(question.content)

    def save_exam(self):
        """
        Saves current exam and exits the program.
        """
        filename = generate_filename()
        print(f"\n\nSaving current exam at {filename}.pkl")
        save_to_pickle(self, filename)
        time.sleep(1)
        exit(0)

    def answer_correct(self, offset_str=""):
        """
        Asks whether question could be answered, otherwise return it into
        self.questions_unanswered.
        """
        ans = input(offset_str + "Answered correctly? [yes]/no ")
        if ans.lower() == "save":
            self.save_exam()
        return ans.lower() in ["", "yes", "y"]

    def exam_loop(self):
        """
        Loops randomly through all questions.
        """
        while len(self.questions_unanswered) > 0:
            self.print_overall_info()
            rnd = np.random.randint(len(self.questions_unanswered))
            question = self.questions_unanswered[rnd]
            self.prompt_question(question)

            all_correct = True
            if not self.answer_correct():
                all_correct = False

            if len(question.subquestions) > 0:
                print("\nFollow-up questions:")
            for i, subq in enumerate(question.subquestions):
                print(f"\n\t{subq.content}")
                if not self.answer_correct("\t"):
                    all_correct = False

            # only if all subquestions were answered correctly,
            # the question will count as answered
            if all_correct:
                self.questions_unanswered.pop(rnd)
                self.questions_answered.append(question)

    def start_exam(self):
        """
        Begins with the exam loop through all questions.
        """
        print("Starting exam...")
        time.sleep(1)
        self.exam_loop()
        self.print_overall_info()
        print("\nCongratulations! You completed the exam!")


def main():

    print("\nStarting Oral Examiner v0.1")
    ans = input("\nStart a new exam? [yes]/no ")

    if ans.lower() in ["", "yes", "y"]:
        path = input("\nEnter path of an .md file containing questions: ")
        questions = read_questions(path)
        examiner = Examiner(questions)
    else:
        path = input("\nEnter filepath for saved exam: ")
        examiner = load_from_pickle(path.replace(".pkl", ""))

    print("")
    examiner.start_exam()

    print("Done!")


if __name__ == "__main__":
    main()
