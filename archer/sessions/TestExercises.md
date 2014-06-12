Testing Exercises
=================

Exercise 1 - add more tests
---------------------------

Add additional tests using `bridge.txt` and `war.txt`.

Exercise 2 - recode `wordcount.py`
----------------------------------

With the simple test harness in-place recode, wordcount.py

* Replace the `DELIMITERS` list with `".,;:?$@^<>#%`!*-=()[]{}/\"\'"`
* Define: `TRANSLATE_TABLE = string.maketrans(DELIMITERS, len(DELIMITERS) * " ")`
* Replase the inefficient `for purge` loop with use of Python's string `translate` function (Google for its use).

