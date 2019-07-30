all: tutorial_notebook.ipynb 

tutorial_notebook.ipynb: tutorial.pmd
	python buildTutorial.py

mechanics_tests.ipynb: mechanics.pmd 
	python buildMechanics.py

