Testing
=======

Question: how do you test?

* Compile the code ("if it builds, ship it")?
* Run on correct inputs, check results by visual inspection?
* Run on correct inputs, check results automatically?
* Run tests after every change or bug fix?
* Use continuous integration?
* Document manual testing?
* Run on inputs known to cause failures?
* Run to ensure fixed bugs don't reoccur?

Question: if you don't test, why not?

* "I don't write buggy code". Almost all code has bugs.
* "It's too hard". This can be a sign that the code is not well designed.
* "My code can't be tested". Why not?
* "It's not interesting"
* "It takes too much time and I've research to do"

Avoid embarrassment:

* [Geoffrey Chang](http://en.wikipedia.org/wiki/Geoffrey_Chang) had to retract 3 papers from [Science](http://www.sciencemag.org) due to a flipped sign bit.
* McKitrick and Michaels published an [erratum](http://www.int-res.com/articles/cr2004/27/c027p265.pdf) to a [Climate Research 26(2) 2004](http://www.int-res.com/abstracts/cr/v26/n2/p159-173/) paper due to a [problem](http://crookedtimber.org/2004/08/25/mckitrick-mucks-it-up/) caused by degrees and radians.
* Ariane 5 used Ariane 4 software. Ariane 5's new engines caused the code to produce a buffer overflow. Ariane 5 blew up!

Save money:

* Find a bug on your laptop for free before you submit a job to a charged-for HPC resource.
 
Save time:

* Spot bugs before you analyse data produced by your scripts.
* 1-10-100 rule.

Safety net:

* Fix bugs, optimise and parallelise without introducing (new) bugs.
* EPCC and Colon Cancer Genetics Group (CCGG) of MRC Human Genetics Unit at Western General Hospital Edinburgh optimised and parallelised FORTRAN genetics code.

Documentation:

* How to use, or not to use, scripts and code and what they do.

Verification - "Have we built it correctly?" Is it bug free, precise, accurate, and repeatable?

Validation - "Have we built the right thing?" Is it designed in such a way as to produce the answers we are interested in, data we want, etc?

Finding bugs before testing
---------------------------

Question: what is single most effective way of finding bugs?

Answer: 

* Fagan (1976) Rigorous inspection can remove 60-90% of errors before the first test is run. M.E., Fagan (1976). [Design and Code inspections to reduce errors in program development](http://www.mfagan.com/pdfs/ibmfagan.pdf). IBM Systems Journal 15 (3): pp. 182-211.
* Cohen (2006) The value of a code review comes within the first hour, after which reviewers can become exhausted and the issues they find become ever more trivial. J. Cohen (2006). [Best Kept Secrets of Peer Code Review](http://smartbear.com/SmartBear/media/pdfs/best-kept-secrets-of-peer-code-review.pdf). SmartBear, 2006. ISBN-10: 1599160676. ISBN-13: 978-1599160672.

Introducing tests from the outside-in
-------------------------------------

* Unit tests test small individual functions.
* Researchers inherit large codes, which may not have any tests. 
* Where to start with a unit test?
* Evolve tests from the outside-in.
* [Software Sustainability Institute](http://www.software.ac.uk) initial test framework for [FABBER](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FABBER) C++ image analysis software.

End-to-end tests and Python
----------------------------

Count the words in a text file and output their frequencies:

    python wordcount.py books/war.txt war.dat"

Question: what is possibly the simplest test we could do? 

Answer: check there is an output file produced for a valid input file

Use Python to write end-to-end tests. Program being tested does not have to be Python. Can use the same approach using shell scripts.

Create `test_wordcount_end_to_end.py`:

    import os
    import os.path

    def file_exists(filename):
      if (os.path.isfile(filename)):
        print "OK ", filename, "exists"
      else:
        print "FAIL ", filename, "does not exist"

    for f in os.listdir("."):
      if f.endswith(".dat"):
        os.remove(f)

    print "Test war"
    os.system("python wordcount.py books/war.txt war.dat")
    file_exists("war.dat")

Use functions as these commands will be called more than once - anticipate reuse.

    python test_wordcount_end_to_end.py

<p/>

    print "Test abyss"
    os.system("python wordcount.py books/abyss.txt abyss.dat")
    file_exists("abyss.dat")

Tests that code, or functions, work with valid inputs are complemented by tests that they fail with invalid inputs.

Exercise 1 - write a test for no file
-------------------------------------

See [exercises](TestExercises.md).

Solution:

    def file_not_exists(filename):
      if (os.path.isfile(filename)):
        print "FAIL ", filename, "exists"
      else:
        print "OK ", filename, "does not exist"

    print "Test none"
    os.system("python wordcount.py books/none none.dat")
    file_not_exists("none.dat")

Check actual outputs against expected outputs
---------------------------------------------

Create expected outputs:

    make abyss.dat
    make bridge.dat
    make war.dat
    make kim.dat

Or:

    python wordcount.py books/abyss.txt abyss.dat
    python wordcount.py books/bridge.txt bridge.dat
    python wordcount.py books/kim.txt kim.dat
    python wordcount.py books/war.txt war.dat

<p/>

    mkdir expected/
    mv *.dat expected/

    python wordcount.py books/abyss.txt abyss.dat
    diff abyss.dat expected/abyss.dat              # Compare files for equality
    diff -q abyss.dat expected/abyss.dat           # Return result only

`os.system` returns the command's exit code, `0` for OK, and non-zero for errors.

    def files_equal(file1,file2):
      cmd = "diff -q " + file1 + " " + file2
      result = os.system(cmd)
      if (result == 0):
        print "OK ", file1, "equals", file2
      else:
        print "FAIL ", file1, "does not equal", file2

    files_equal("war.dat", "expected/war.dat")
    files_equal("abyss.dat", "expected/abyss.dat")

Reduce duplicated code and loop over files:

    for f in os.listdir("books"):
      if f.endswith(".txt"):
        name = os.path.splitext(f)[0]
        txt = os.path.join("books",f)
        dat = name + ".dat"
        cmd = "python wordcount.py " + txt + " " + dat
        os.system(cmd)
        file_exists(dat)
        expected = os.path.join("expected", dat)
        files_equal(dat,expected)

Exercise 2 - recode `wordcount.py`
----------------------------------

See [exercises](TestExercises.md).

Solution:

    DELIMITERS=".,;:?$@^<>#%`!*-=()[]{}/\"\'"
    TRANSLATE_TABLE = string.maketrans(DELIMITERS, len(DELIMITERS) * " ")

    def update_word_counts(line, counts):
      line = string.translate(line, TRANSLATE_TABLE) 
      words = line.split()
      ...

Test structure
--------------

Test structure:

* Set-up expected outputs given known inputs e.g. `expected/.dat` files or `0` for the return code.
* Run component on known inputs.
* Check if actual outputs match expected outputs, or, for invalid inputs, that behaviour is as expected.

Regardless of whether it is a test of a:

* 10 line function.
* Component or library.
* Serial application running on a single processor.
* Parallel application running on a multiple processors.
* Automated or manual.

Data file meta-data
-------------------

Add meta-data to record the provenance of the output data file. 

    import datetime

    f.write("# Frequency data\n")
    f.write("# Created by: %s\n" % __file__)
    f.write("# Input data: %s\n" % file)
    f.write("# Date: %s\n" % datetime.datetime.today())
    f.write("# Format: word frequency\n")

<p/>

    python wordcount.py books/abyss.txt abyss.dat
    head abyss.dat
    python test_wordcount_end_to_end.py

Question: what is the problem?

Answer: the meta-data. `diff` is too simplistic now. 

* Want finer-grained tests of equality between data files. 
* Use information about the file content and structure.
* Difference between syntactic and semantic content.

When 0.1 + 0.2 == 3.00000000004
-------------------------------

Question: what other problems might `diff` experience with data files?

Answer: floating point values.

    python
    a = 0.1
    b = 0.2
    print a + b
    print a + b == 0.3
    a + b

Simple tests for the equality of two floating point values are problematic due to imprecision in values.

Compare for equality within a given threshold, or delta e.g. *expected* and *actual* to be equal if *expected - actual < 0.000000000001*.

Python [nose](https://pypi.python.org/pypi/nose/) library includes functions for floating point equality.

    python
    from nose.tools import assert_almost_equal
    expected = 2
    expected = 2.000001
    actual = 2.0000000001
    assert_almost_equal(expected, actual, 0)
    assert_almost_equal(expected, actual, 1)
    assert_almost_equal(expected, actual, 2)
    assert_almost_equal(expected, actual, 3)
    assert_almost_equal(expected, actual, 4)
    assert_almost_equal(expected, actual, 5)
    assert_almost_equal(expected, actual, 6)

`nose.testing` uses absolute tolerance: abs(x, y) <= delta.

[Numpy](http://www.numpy.org/) `numpy.testing` uses relative tolerance: abs(x, y) <= delta * (max(abs(x), abs(y)). 

`data/` has files produced by the same software, with the same inputs, under the same configuration. 

Question: why might they differ?

Answer: they were run on different numbers on processors e.g. 2x1, 4x2 etc.

    diff -q data/2x1.dat data/data4x2.dat

<p/>

    import numpy as np

    def test_data_files_equal():
      file21 = np.loadtxt("data/data2x1.dat")
      file42 = np.loadtxt("data/data4x2.dat")
      np.testing.assert_equal(file21, file42)

    test_data_files_equal()

<p/>

    python test_wordcount_end_to_end.py

<p/>

    np.testing.assert_allclose(file21, file42, rtol=0, atol=1e-7)

What is a suitable threshold for equality? That is application-specific - for some domains round to the nearest whole number, for others be far, far more accurate.

Testing at finer-granularities - towards unit tests
---------------------------------------------------

* Tests at varying levels of granularity should be written.
* Changes to a specific component can be tested before the component is integrated.
* Quicker to discover a problem when testing a 10 line function in isolation then testing it as part of an end-to-end application which may take 1 hour to run and may not even, depending upon the inputs, invoke that function. 
* Finest level of granularity is a unit test where a unit is the smallest testable part of an application e.g. function or module, method or class.

Exercise 3 - propose some tests for `wordcount.py`
--------------------------------------------------

See [exercises](TestExercises.md).

Solution (examples):

`load_text(file)`

* A file returns a list of length equal to the number of lines in the file.
* An empty file returns an empty list.
* A non-existent file raises an error.

`save_word_counts(file, counts)`

* An empty list results in an empty file.
* A non-empty list results in a file with a number of lines equal to the number of tuples + 5 (for the meta-data lines).

`load_word_counts(file)`

* A file with no comments (prefixed by `#`) returns a list of length equal to the number of lines in the file.
* A file only with comments returns an empty list.
* An empty file returns an empty list.
* A non-existent file raises an error.

`update_word_counts(line, counts)`

* A sentence of distinct words and an empty dictionary of frequencies results in a dictionary with an entry for each word with a count of 1.
* A zero-length sentence and an empty dictionary of frequencies results in an empty dictionary of frequencies.
* A sentence of delimiters and an empty dictionary of frequencies results in an empty dictionary of frequencies.
* A sentence of distinct words and a dictionary with entries for all of those words results in each count in the dictionary being increased by 1.

`calculate_word_counts(lines)`:

* An empty list returns an empty dictionary.
* Other tests as for `update_word_counts` but across a number of lines.

`word_count_dict_to_tuples(counts, decrease = True)`:

* An empty dictionary returns an empty list.
* A dictionary of words all with equal counts returns a list of tuples, one for each word.
* A dictionary of words with distinct counts returns list of tuples in descending order.
* A dictionary of words with distinct counts returns list of tuples in ascending order if `decrease = False`.

`filter_word_counts(counts, min_length = 1)`

* An empty list returns an empty list.
* A list of tuples of words of length 3 or less, returns an empty list if `min_length = 3`.

A unit test for `add_frequencies`
---------------------------------

Create `test_wordcount.py`:

    from wordcount import update_word_counts

    def test_update_word_counts():
      line = "software! software! software!"
      counts = {}
      update_word_counts(line, counts)

Python [nose](https://pypi.python.org/pypi/nose/) library includes tests for equality, inequality, boolean values, thrown exceptions etc.

    from nose.tools import assert_equal

    assert_equal(1, len(counts))
    assert_equal(3, counts.get("software"))

    test_update_word_counts()

<p/>

    python test_wordcount.py

`nosetests` automatically finds, runs and reports on tests.

    nosetests test_wordcount.py

`.` denotes successful test function calls.

Uses 'reflection' to find out the test functions - `test_` function, module and file prefixes.

Remove `test_update_word_counts()` call.

<p/>

    python test_wordcount.py

<p/>

    def test_update_word_counts_distinct():
      line = "software carpentry software training"
      counts = {}
      update_word_counts(line, counts)
      assert_equal(3, len(counts))
      assert_equal(2, counts.get("software"))
      assert_equal(1, counts.get("carpentry"))
      assert_equal(1, counts.get("training"))

`nose` is an [xUnit test framework](http://en.wikipedia.org/wiki/XUnit). Others are JUnit, CUnit, google-test, FRUIT, pFUnit etc.

xUnit test report, standard format, convert to HTML, present online.

    nosetests --with-xunit test_wordcount.py
    cat nosetests.xml

Defensive programming
---------------------

Suppose an invalid set of counts are passed to `calculate_percentages`:

    from wordcount import calculate_percentages

    counts = [("software", 1), ("software", -4)]
    print calculate_percentages(counts)

Assume that mistakes will happen and guard against them. Defensive programming.

    def calculate_percentages(counts):
      total = 0
      for count in counts:
        assert count[1] >= 0
        total += count[1]

Programs like Firefox  are full of assertions: 10-20% of their code is to check that the other 80-90% is working correctly.

For HPC resources, want to exit a.s.a.p rather than continue executing and burn up resource allocations.

Types:

* Precondition - must be true at the start of a function in order for it to work correctly.
* Postcondition - guaranteed to be true when a function finishes.
* Invariant - always true at a particular point inside a piece of code.

Help other developers understand program and whether their understanding matches the code. Users should never see these sorts of failure!

    from nose.tools import assert_raises

    def test_calculate_percentages_invalid():
      counts = [("software", 1), ("software", -4)]
      assert_raises(AssertionError, calculate_percentages, counts)

Exercise 4 - write more unit tests for `wordcount.py`
-----------------------------------------------------

See [exercises](TestExercises.md).

Allow 15-30 minutes.

Automated testing jobs
----------------------

Automated tests can be run:

* Manually.
* At regular intervals.
* Every time code is commited to revision control.

A simple automatic triggering of automated tests is via a Unix `cron` job.

A more advanced approach is via a continuous integration server. These trigger automated test runs and publish the results.

* [Muon Ion Cooling Experiment](http://www.mice.iit.edu/) (MICE) have a large number of tests written in Python. They use [Jenkins](http://jenkins-ci.org/) to build their code and trigger the running of the tests which are then [published online](http://test.mice.rl.ac.uk/).
* [Apache Hadoop Common Jenkins dashboard](https://builds.apache.org/job/Hadoop-Common-trunk/)

Tests are code
--------------

Review tests and avoid tests that:

* Pass when they should fail, false positives.
* Fail when they should pass, false negatives.
* Don't test anything. 

<p/>

    def test_critical_correctness():
        # TODO - will complete this tomorrow!
        pass

Conclusion
----------

Tests:

* Set-up expected outputs given known inputs.
* Run component on known inputs.
* Check if actual outputs match expected outputs.

When to test:

* Always!
* Early. Don't wait till after the code's been used to generate data for an important paper, or been given to someone else.
* Often. So any bugs can be identified a.s.a.p. Bugs are easier to fix if they're identified at the time the relevant code is being actively developed.
* Turn bugs into assertions or tests. Check that bugs do not reappear.

When to finish:

* "It is nearly impossible to test software at the level of 100 percent of its logic paths", fact 32 in R. L. Glass (2002) [Facts and Fallacies of Software Engineering](http://www.amazon.com/Facts-Fallacies-Software-Engineering-Robert/dp/0321117425) ([PDF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.2037&rep=rep1&type=pdf)).
* But, no excuse for not testing anything.
* When do you finish proof reading a paper? Learn from experience. 

"If it's not tested, it's broken" - Bruce Eckel, in [Thinking in Java, 3rd Edition](http://www.mindview.net/Books/TIJ/).

["Testing is science"](http://maori.geek.nz/post/testing_your_code_is_doing_science) - Graham Jenson.Check scripts and code:
