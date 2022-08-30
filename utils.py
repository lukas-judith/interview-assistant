import pdb
import os
import pickle
from datetime import datetime
from questions import Question


DELIMITER_QUESTION = '- '
DELIMITER_SUBQUESTION = '  - '

TITLE_SYMBOL = "# "
SECTION_SYMBOL = "## "
SUBSECTION_SYMBOL = "### "
SUBSUBSECTION_SYMBOL = "#### "


# TODO: assess that the symbols cannot contain each other...


def read_questions(filepath):
    """
    Reads Markdown file and returns list of question objects.
    """
    file = open(filepath, 'r')
    txt = file.read()
    file.close()

    lines = txt.split("\n")
    title = None
    section = None
    subsection = None
    subsubsection = None
    content = None
    subquestions = []

    questions = []

    for line in lines:
        # note: if-statements are structured like this in case the symbols contain each other
        if SUBSUBSECTION_SYMBOL in line:
            subsubsection = line.replace(SUBSUBSECTION_SYMBOL, "").replace("\n", "")
        elif SUBSECTION_SYMBOL in line:
            subsection = line.replace(SUBSECTION_SYMBOL, "").replace("\n", "")
            subsubsection = None
        elif SECTION_SYMBOL in line:
            section = line.replace(SECTION_SYMBOL, "").replace("\n", "")
            subsection = None
            subsubsection = None
        elif TITLE_SYMBOL in line:
            title = line.replace(TITLE_SYMBOL, "").replace("\n", "")
            section = None
            subsection = None
            subsubsection = None

        elif DELIMITER_SUBQUESTION in line:
            subquestion_content = line.replace(DELIMITER_SUBQUESTION, "").replace("\n", "")
            subquestion = Question(subquestion_content, None, title, section, subsection, subsubsection)
            subquestions.append(subquestion)
        elif DELIMITER_QUESTION in line:
            # store previous question including subquestions
            if not content is None:
                question = Question(content, subquestions, title, section, subsection, subsubsection)
                questions.append(question)

            # set content for new question and reset subquestions
            content = line.replace(DELIMITER_QUESTION, "").replace("\n", "")
            subquestions = []

    print(f"Gathered {len(questions)} from {filepath}!")  
    return questions

    
def get_progress_bar_string(percentage):
    if not percentage >= 0 and percentage <= 100:
        raise Exception("Input should be percentage between 0 and 100!")
    n_total = 20
    n_bars = int(percentage / 100 * n_total)
    n_empty = n_total - n_bars
    txt = '[' + n_bars * '|' + n_empty * ' ' + ']'
    return txt



def generate_filename():
    """
    Creates destination to save file.
    """
    results_dir = 'saved_exams'
    try:
        os.mkdir(results_dir)
    except:
        pass
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    name_ = 'saved_exam_' + time_str
    results_dir_ = os.path.join(results_dir, name_)
    try:
        os.mkdir(results_dir_)
    except:
        pass
    return results_dir_


def save_to_pickle(variable, file_name):
    filename = '{}.pkl'.format(file_name)
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)


def load_from_pickle(file_name):
    filename = '{}.pkl'.format(file_name)
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    return variable