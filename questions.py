

class Question():

    def __init__(self, content, subquestions=None, title=None, section=None,
                 subsection=None, subsubsection=None, answer=None):

        self.content = content
        self.subquestions = subquestions
        self.title = title
        self.section = section
        self.subsection = subsection
        self.subsubsection = subsubsection
        self.answer = answer

    def print_all_info(self):
        print("Question:", self.content)
        print("Number of subquestions:", len(self.subquestions))
        print("Title of document:", self.title)
        print("Section:", self.section)
        print("Subsection:", self.subsection)
        print("Subsubection:", self.subsubsection)
    
    def print_subquestions(self):
        for q in self.subquestions:
            print(q.content)

    def print_question(self):
        pass
    
    def print_answer(self):
        pass

