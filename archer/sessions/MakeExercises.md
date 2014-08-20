Automation and Make Exercises
=============================

Exercise 1 - add a rule 
-----------------------

Add in a rule for `bridge.dat`, based on the rules for `war.dat` and
`abyss.dat`.

Be sure to use a single tab when indenting the actions, not spaces!

`touch` all the `books/*.txt` files.

Re-run `make` and all the `.dat` files should be rebuilt.

Exercise 2 - simplify a rule 
----------------------------

Simplify the rule of the `%.dat` target using automatic variables:

* `$@` is the target of the current rule.
* `$<` is the first dependency only.

Exercise 3 - use a macro
------------------------

Replace occurrences of `wordcount.py` with the macro-name,
`$(COUNTER)`.

Exercise 4 - add another processing stage
-----------------------------------------

Add a rule to create `.jpg` files from `.dat` files, using wild-cards.

Modify the `analysis.tar.gz` rule to add the `.jpg` files to the `.gz` file.

Use a macro to hold the script name `plotcount.py`.

Add a `clean` rule to remove `.jpg` and `.dat` files and `analysis.tar.gz`.