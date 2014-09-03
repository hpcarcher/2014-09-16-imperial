Testing Exercises
=================

Exercise 1 - write a test for no file
-------------------------------------

Copy `file_exists`, rename the copy to `file_not_exists` and modify it to test that a file does not exist.

Add an example test that calls this function.

Exercise 2 - recode `jacobi.py`
-------------------------------

With the simple test harness in-place recode, `jacoby.py`:

* The `while` loops are of form:

<p/>

    VARIABLE = INITIAL_VALUE
    while VARIABLE < FINAL_VALUE:
        DO SOMETHING
        VARIABLE += 1

* A more 'Pythonic' way (a way more in the spirit of the Python language) to write loops is to replace the above with:

<p/>

    for VARIABLE in range(INITIAL_VALUE, FINAL_VALUE)
        DO SOMETHING

* Replace the five `while` loops with `for` loops.

Exercise 3 - propose unit tests for `cfd.py` and `jacobi.py`
------------------------------------------------------------

With a partner, or in threes, note down possible tests for each function in `cfd.py` and `jacobi.py`.

Remember that testing with invalid arguments or boundary conditions can be as important if testing with valid arguments one knows to be correct.

Exercise 4 - implement unit tests for `cfd.py` and `jacobi.py`
--------------------------------------------------------------

Implement unit tests for `cfd.py` and `jacobi.py`
