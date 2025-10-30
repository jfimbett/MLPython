# A template to create quarto slides powered with an AI agent 

Instructions

Modify `_quarto.yml` to set your presentation title, author, and date.
Modify `styles.css` to customize the appearance of your slides.
Modify `.nojekyll` to make sure GitHub Pages works correctly.

-------------------------------------------------------------------------------------------
## Example template for creating a set of slides 

This repository provides a minimal template for creating a set of slides using Quarto, RevalJS. The general instructions are located in `.github/copilot-instructions.md`.

Create a 1 day course on the usage of `conda` and `python`. The course material should cover an introduction and history of python, and why using it through conda is a good idea. The course should also cover how to install conda, create conda environments, and install packages using conda. The course should also cover how to use jupyter notebooks and jupyter lab.  The course should also cover how to use vscode as an IDE for python development, since this course will be accessed by students through GitHub codespaces. The course should also cover how to use pip and virtualenv as an alternative to conda. Create complete but concise slides, making sure you have at least 5 bulleted points per slide. Write code snippets where appropriate.

If you have any questions about the course, ask them before generating any content.

### Course information

Instructor: Juan F. Imbet
Course Title: Introduction to Python and Conda
Course Description: This course provides a comprehensive introduction to Python programming and the Conda package management system.
Program: Master 2 (203) in Financial Markets, Paris Dauphine - PSL University. 

--------------------------------------------------------------------------------------------

## GitHub Instructions

- To create a new repository with this information, clone it with a different name and remove the `remote` connection. 
- In the parent directory, run:
```bash
git clone https://github.com/jfimbett/TemplateCourse.git YourCourseName
cd YourCourseName
git remote remove origin
gh repo create YourCourseName --public --source=. --remote=origin --push
```
