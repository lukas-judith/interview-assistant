# Interview Assistant

This code was written in preparation for an oral examination during my postgraduate studies at Heidelberg University. The program simulates an interview or oral exam, using lists of questions stored in Markdown (.md) files. Proper formatting of the questions is illustrated in the example file *questions.md*. Questions appear in randomized order. Current progress of an exam can be saved at any time, so that it can be resumed later.



**How to use the code:**

*Prerequisites: [conda](https://docs.conda.io/en/latest/) must be installed.*

1. Create a new conda environment from the .yml file, using 

   ```
   conda env create --file env.yml
   ```

2. Activate the environment:

   ```
   conda activate exam
   ```

4. Run the main.py script to start an exam simulation:

   ```
   python main.py
   ```



**TODO:**

- Add the option to reveal the correct answer to each question during the simulated exam
