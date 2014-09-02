Automation and Make Exercises
=============================

Exercise 1 - add a rule 
-----------------------

Add in a rule for `96x96.dat`, based on the rules for `32x32.dat` and
`64x64.dat`.

Be sure to use a single tab when indenting the actions, not spaces!

`touch` all the `configs/*.cfg` files.

Re-run `make` and all the `.dat` files should be rebuilt.

Exercise 2 - simplify a rule 
----------------------------

Simplify the rule of the `%.dat` target using automatic variables:

* `$@` is the target of the current rule.
* `$<` is the first dependency only.

Exercise 3 - use a macro
------------------------

Replace occurrences of `cfd.py` and `python cfd.py` with appropriate
use of the the macros `$(CFD_EXE)` and `$(CFD_SRC)`.

Exercise 4 - add another processing stage
-----------------------------------------

Add a rule to create `.jpg` files from `.dat` files, using wild-cards.

Modify the `analysis.tar.gz` rule to add the `.jpg` files to the `.gz` file.

Use macros to hold the script name `plot_flow.py`.

Add a `clean` rule to remove `.jpg` and `.dat` files and `analysis.tar.gz`.
