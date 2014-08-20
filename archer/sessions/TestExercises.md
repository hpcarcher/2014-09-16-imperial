Testing Exercises
=================

Exercise 1 - write a test for no file
-------------------------------------

Copy `file_exists`, rename the copy to `file_not_exists` and modify it to test that a file does not exist.

Add an example test that calls this function.

Exercise 2 - recode `wordcount.py`
----------------------------------

With the simple test harness in-place recode, wordcount.py

* Replace the `DELIMITERS` list with ".,;:?$@^<>#%`!*-=()[]{}/\"\'"
* Define: `TRANSLATE_TABLE = string.maketrans(DELIMITERS, len(DELIMITERS) * " ")`
* Replase the inefficient `for purge` loop with Python's string `translate` function (Google for its use).

Exercise 3 - propose tests for `wordcount.py`
---------------------------------------------

With a partner, or in threes, write down possible tests for each function in `wordcount.py`.

Remember that testing with invalid arguments or boundary conditions can be as important if testing with valid arguments one knows to be correct.

Exercise 4 - implement unit tests for `wordcount.py`
----------------------------------------------------

Implement more unit tests for `wordcount.py`.
