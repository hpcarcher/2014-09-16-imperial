Automation and Make Exercises
=============================

Exercise 1 - add a rule 
-----------------------

Add in a rule for `bridge.dat`, based on the rules for `war.dat` and
`jekyll.dat`.

Be sure to use a single tab when indenting the actions, not spaces!

`touch` all the *.txt files.

Re-run `make` and all the `.dat` files should be rebuilt.

Exercise 2 - simplify a rule 
----------------------------

Simplify the rule of the `%.dat` target using automatic variables:

* `$@` is the target of the current rule.
* `$<` is the first dependency only.

Exercise 3 - use a macro
------------------------

Replace occurrences of `wordcount.py` with the macro-name,
`$(PROCESSOR)`.

Exercise 4 - add another processing stage
-----------------------------------------

`plotcount.py` contains a Python script that plots data in the
two-column `.dat` files e.g.

    python plotcount.py -show war.dat
    python plotcount.py -show kim.dat
    python plotcount.py -show bridge.dat

If given a `-save` flag and another file name it can save the plot
as a `.jpg` e.g.

    python plotcount.py -save war.dat war.jpg

Extend the Makefile:

* Add a rule to create `.jpg` files from `.dat` files, using wild-cards.
* Modify the `analysis.tar.gz` rule to add the `.jpg` files to the `.gz` file.
* Use a macro to hold the script name `plotcount.py`.
