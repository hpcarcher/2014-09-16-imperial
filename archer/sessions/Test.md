Testing
=======

Question: how do you test?

* Just compile the code ("if it builds, ship it")?
* Run on sets of known inputs, validate results by visual inspection?
* Run on sets of known inputs, validate results with additional tools run manually?
* Run on sets of known inputs, validate results automatically?
* Run tests after every change or bug fix?
* Use continuous integration?
* Document manual validation processes?
* Run on sets of inputs known to cause failures?

Question: if you don't test, why not?

* "I don't write buggy code" - almost all code has bugs.
* "It's too hard" - if it's hard to write a test for some code, then this is a good sign that the code is not well designed.
* "My code can't be tested" 
 * Question: why not?
* "It's not interesting" - which usually means...
* "It takes too much time and I've research to do"

Why testing is important:

* Correct, trustworthy research:
 * [Geoffrey Chang](http://en.wikipedia.org/wiki/Geoffrey_Chang) had to retract 3 papers from [Science](http://www.sciencemag.org) due to a flipped sign bit.
 * McKitrick and Michaels published an [erratum](http://www.int-res.com/articles/cr2004/27/c027p265.pdf) to a [Climate Research 26(2) 2004](http://www.int-res.com/abstracts/cr/v26/n2/p159-173/) paper due to a [problem](http://crookedtimber.org/2004/08/25/mckitrick-mucks-it-up/) caused by degrees and radians.
* Avoid embarassment:
 * Ariane 5 used Ariane 4 software. Ariane 5's new engines caused the code to produce a buffer overflow. Ariane 5 blew up!
* Saves time for research.

Testing checks code:

* Behaves as expected and produces valid output data given valid input data. Any valid input data.
* Fails gracefully if given invalid input data, does not crash, behave mysteriously, above all, not continue on regardless and burn CPU cycles.
* Handles extreme boundaries of input domains, output ranges, parametric combinations or any other edge cases.
* Behaves the same after it changes (add new features, fix bugs, optimise, parallelise) - regression testing. Nothing is worse than fixing a bug only to introduce a new one.

Testing roles:

* Verification - "Have we built the software correctly?" That is, is the code bug free, precise, accurate, and repeatable?
* Validation - "Have we built the right software?" That is, is the code designed in such a way as to produce the answers we are interested in, data we want, etc.

Testing as documentation:

* Help remember what code does.
* Help remember how to use code e.g. how to use each function, its inputs, outputs, behaviour and restrictions.

Finding bugs before testing
---------------------------

Question: what is single most effective way of finding bugs?

Answer: Fagan (1976) Rigorous inspection can remove 60-90% of errors before the first test is run. Cohen (2006) The value of a code review comes within the first hour, after which reviewers can become exhausted and the issues they find become ever more trivial.
* M.E., Fagan (1976). [Design and Code inspections to reduce errors in program development](http://www.mfagan.com/pdfs/ibmfagan.pdf). IBM Systems Journal 15 (3): pp. 182-211.
* J. Cohen (2006). [Best Kept Secrets of Peer Code Review](http://smartbear.com/SmartBear/media/pdfs/best-kept-secrets-of-peer-code-review.pdf). SmartBear, 2006. ISBN-10: 1599160676. ISBN-13: 978-1599160672.

Introducing tests from the outside-in
-------------------------------------

Software testing often is introduced via unit tests, which test small individual functions.

Researchers inherit large codes, which may not have any tests. Where does one start with a unit test?

Incrementally evolve a test framework from the outside in and explore options available.

Examples:

* Software Sustainability Institute project to introduce a test framework for BASIL and FABBER shell and C++ image analysis software.
* EPCC and the Colon Cancer Genetics Group (CCGG) of the MRC Human Genetics Unit at the Western General as part of an [Oncology](http://www.edikt.org/edikt2/OncologyActivity) project to optimise and parallelise a FORTRAN genetics code.

Will use as a research code `wordcount.py` which takes in a text file and outputs a data file with words and their frequencies.

End-to-end tests and the shell
------------------------------

Question: what is possibly the simplest test we could do? `wordcount.py` reads an input file and writes an output file.

Answer: check there is an output file produced for a valid input file.

Create `test_word_count.sh`:

    #!/bin/sh

    # $1 File to check existence for.
    test_file_exists() {
      if [ -f "$1" ]
      then
        echo "OK: $1 exists"
      else
        echo "FAILURE: $1 not found"
      fi
    }

    rm -f *.dat
    echo "Test 1"
    python wordcount.py abyss.txt abyss.dat
    test_file_exists abyss.dat

Use shell functions as these commands will be called more than once. Think ahead. Plan for reuse.

Run:

    $ chmod +x test_word_count.sh
    $ ./test_word_count,sh

Extend:

    echo "Test 2"
    python wordcount.py war.txt war.dat
    test_file_exists war.dat

Run:

    $ ./test_word_count,sh

Another simple test is for failure, that there is no output file if there is an invalid, or no, input file.

Exercise 1 - write a test for no file
-------------------------------------

See [exercises](TestExercises.md).

Solution:

    # $1 File to check non-existence for.
    test_file_not_exists() {
      if [ -f "$1" ]
      then
        echo "FAILURE: $1 exists"
      else
        echo "OK: $1 not found"
      fi
    }

    echo "Test 3"
    python wordcount.py no_such_file.txt none.dat
    test_file_not_exists none.dat

Check actual outputs against expected outputs
---------------------------------------------

Create expected outputs:

    $ python wordcount.py abyss.txt abyss.dat
    $ python wordcount.py bridge.txt bridge.dat
    $ python wordcount.py kim.txt kim.dat
    $ python wordcount.py war.txt war.dat
    $ mkdir expected/
    $ mv *.dat expected/

`diff` compares files for equality:

    $ python wordcount.py abyss.txt  > abyss.dat
    $ diff abyss.dat expected/abyss.dat

`$?` holds the exit code of the command, `0` for OK, and non-zero for errors:

    $ echo $?
    $ diff abyss.txt kim.txt
    $ echo $?

Extend tests to check actual outputs against expected outputs:

    # $1 file to compare
    # $2 file to compare
    test_files_equal() {
      compare=`diff -rq $1 $2`
      if [ -z "$compare" ]; then
        echo "OK: $1 equals $2"
      else
        echo "FAILURE: $1 does not equal $2"
      fi
    }

    echo "Test 4"
    python wordcount.py abyss.txt abyss.dat
    test_files_equal abyss.dat expected/abyss.dat

    $ ./test_word_count.sh

Check no cheating:

    test_files_equal abyss.dat expected/kim.dat

    $ ./test_word_count.sh

Restore:

    test_files_equal abyss.dat expected/abyss.dat

    $ ./test_word_count.sh

Hard-coding the sample file names can be problematic. Automate:

    for file in $(ls *.txt); do
      name=`basename $file .txt`
      output_file=$name.dat
      echo "Test - $file"
      python wordcount.py $file $output_file
      test_file_exists $output_file
      test_files_equal $output_file expected/$output_file
    done

    $ ./test_word_count.sh

Exercise 2 - recode `wordcount.py`
----------------------------------

See [exercises](TestExercises.md).

Solution:

    DELIMITERS=".,;:?$@^<>#%`!*-=()[]{}/\"\'"
    TRANSLATE_TABLE = string.maketrans(DELIMITERS, len(DELIMITERS) * " ")

    def add_frequencies(line, frequencies, min_length = DEFAULT_MIN_LENGTH):
      line = string.translate(line, TRANSLATE_TABLE) 
      words = line.split()
      ...

Test structure
--------------

Tests follow a common pattern:

* Set-up expected outputs given known inputs e.g. `expected/.dat` files or `0` for the return code.
* Run component on known inputs.
* Check if actual outputs match expected outputs.

This includes:

* A unit test of a 10 line function.
* Test of a component or library.
* Test of a serial application running on a single processor.
* Test of a parallel application running on a multiple processors.

...whether manually done or automated!

Data file meta-data
-------------------

Add meta-data to the output file to record the provenance of the data file. 

    import datetime

    f.write("# Frequency data\n")
    f.write("# Created by: %s\n" % __file__)
    f.write("# Input data: %s\n" % file)
    f.write("# Date: %s\n" % datetime.datetime.today())
    f.write("# Format: word frequency\n")

Run:

    $ python wordcount.py abyss.txt abyss.dat
    $ head abyss.dat

Run:

    $ ./test_wordcount.sh

Question: what is the problem?

Answer: the meta-data. `diff` is too simplistic now. 

One workaround would be to use shell commands to trim off the problematic lines. This is just delaying the problem. The files may have too complex a structure for such manipulations.

Write our tests in a programming language.

When 0.1 + 0.2 == 3.00000000004
-------------------------------

Question: what other problems might `diff` experience with data files?

Answer: floating point values.

    $ python
    > a = 0.1
    > b = 0.2
    > print a + b
    > print a + b == 0.3
    > a + b

Computers don't do floating point arithmetic too well. This can make simple tests for the equality of two floating point values problematic due to imprecision in the values being compared. 

Will typically want finer grained tests of equality between data files.

For floating point values, compare for equality within a given threshold, or delta, for example we may consider *expected* and *actual* to be equal if *expected - actual < 0.000000000001*.

This applies not only for comparing data files en masse but also when testing components at any level of granularity.

Testing at finer-granularities - towards unit tests
---------------------------------------------------

End-to-end automated testing is better than nothing. Ideally, though, tests at varying levels of granularity should be written.

If every component has a set of tests then changes to the component can be tested before the component is integrated.

It can be quicker to discover a problem when testing a 10 line function in isolation then testing it as part of an end-to-end application which may take 1 hour to run and may not even, depending upon the inputs, invoke that function. 

The finest level of granularity is called a unit test, typically of individual functions, where a unit is the smallest testable part of an application. This could be a function or module, method or class.

Exercise 3 - propose some tests for `wordcount.py`
--------------------------------------------------

See [exercises](TestExercises.md).

Solution:

Many examples exist including:

`load_text(file)`

* A standard file returns a list of length equal to the number of lines in the file.
* An empty file returns an empty list.
* A non-existent file raises an error.

`add_frequencies(line, frequencies, min_length = DEFAULT_MIN_LENGTH)`

* A sentence of distinct words and an empty dictionary of frequencies results in a dictionary with an entry for each word with a count of 1.
* A zero-length sentence and an empty dictionary of frequencies results in an empty dictionary of frequencies.
* A sentence of distinct words and a dictionary with entries for all of those words results in each count in the dictionary being increased by 1.
* A sentence of words each less than 3 and a min_length of 4 and an empty dictionary results in an empty dictionary.

`get_frequencies(lines, min_length = DEFAULT_MIN_LENGTH)`

* An empty list returns an empty list of pairs.
* Other tests as for `add_frequencies` but across a number of lines.

`save_pairs(file, pairs)`

* An ampty list results in an empty file.
* A non-empty list results in a file with a number of lines equal to the number of pairs in the list.

A unit test for `add_frequencies`
---------------------------------

Create `test_wordcount.py`:

    from wordcount import add_frequencies

    def test_add_frequencies():



What is:

    from nose.tools import assert_equal




import unittests
class MyTests(unittest.TestCase):
if __name__ == '__main__':
    unittest.main()
A test case is defined by subclassing `unittest.TestCase`. We have defined two rather uninteresting tests. If we run this we get:
TestHarness.py



nosetests
---------

Why did I call tests test_?

[nose](https://pypi.python.org/pypi/nose/) automatically finds, runs and reports on tests.

[xUnit test framework](http://en.wikipedia.org/wiki/XUnit).

JUnit, CUnit, FUnit, ...

`test_` file and function prefix, `Test` class prefix.

    $ nosetests test_morse.py

`.` denotes successful tests.

Remove `__main__`.

    $ nosetests test_morse.py

xUnit test report, standard format, convert to HTML, present online.

    $ nosetests --with-xunit test_morse.py
    $ cat nosetests.xml

`nose` defines additional functions which can be used to check for a rich range of conditions e.g..

    $ python
    >>> from nose.tools import *
    >>> expected = 123
    >>> actual = 123
    >>> assert_equal(expected, actual)
    >>> actual = 456
    >>> assert_equal(expected, actual)
    >>> expected = "GATTACCA"
    >>> actual = ["GATC", "GATTACCA"]
    >>> assert_true(expected in actual)
    >>> assert_false(expected in actual)

Problem
-------

End to end is fine!
Same applies to others

CAN FLOATING POINT BE DONE HERE
    >>> assert_true("GTA" in actual, "Expected value was not in the output list")












Exercise 4 - write more unit tests for `wordcount.py`
-----------------------------------------------------

See [exercises](TestExercises.md).

Allow 15 minutes or so.







Defensive programming and testing for failures
----------------------------------------------

http://apawlik.github.io/2014-04-09-GARNET/novice/python/05-defensive.html
"defensive programming"

        try:
            for ch in sequence:
                weight += NUCLEOTIDES[ch]
            return weight
        except TypeError:
            print 'The input is not a sequence e.g. a string or list'

Now, the exception is *caught* by the `except` block. This is a *runtime test*. It alerts the user to exceptional behavior in the code. Often, exceptions are related to functions that depend on input that is unknown at compile time. Such tests make our code robust and allows our code to behave gracefully - they anticipate problematic values and handle them.
Often, we want to pass such errors to other points in our program rather than just print a message and continue. So, for example we could do,
    except TypeError:
        raise ValueError('The input is not a sequence e.g. a string or list')
which raises a new exception, with a more meaningful message. If writing a complex application, our user interface could then present this to the user e.g. as a dialog box.


    try:
        calculate_weight(123) 
        assert False
    except ValueError:
        assert True

This is like catching a runtime error. If an exception is raised then our test passes (`assert True`), else if no exception is raised, it fails. Alternatively, we can use `assert_raises` from `nose.tools`,

    from nose.tools import assert_raises

    def test_123():
        assert_raises(ValueError, calculate_weight, 123)

The assert fails if the named exception is *not* raised.









Floating point pains
--------------------

Back to our end to end

Add in %age total of words

Add in numpy-based end-to-end tests and fudge factors
 import numpy as np
 expected = np.loadtxt("expected_data.csv", delimiter=",")
 actual = np.loadtxt("output_data.csv", delimiter=",")
 assert expected.shape == actual.shape
 assert np.allclose(actual, expected, 0.001)















Add in an end to end-called in-code test
----------------------------------------

What if tests not function based?
Redesign
Capture inputs and outputs e.g. FABBER



Automated testing jobs
----------------------

Automated tests can be run:

* Manually.
* At regular intervals.
* Every time code is commited to revision control.

A simple automatic triggering of automated tests is via a Unix `cron` job.

A more advanced approach is via a continuous integration server. These trigger automated test runs and publish the results.

* [Muon Ion Cooling Experiment](http://www.mice.iit.edu/) (MICE) have a large number of tests written in Python. They use [Jenkins]() to build their code and trigger the running of the tests which are then [published online](https://micewww.pp.rl.ac.uk/tab/show/maus).
* [Apache Hadoop Common Jenkins dashboard](https://builds.apache.org/job/Hadoop-Common-trunk/)

Tests are code
--------------

Tests should be reviewed for the same reasons code should be reviewed.

Avoid tests that:

* Pass when they should fail, false positives.
* Fail when they should pass, false negatives.
* Don't test anything. 

An example inspired by real-life:

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

When to finish writing tests:

* "It is nearly impossible to test software at the level of 100 percent of its logic paths", fact 32 in R. L. Glass (2002) [Facts and Fallacies of Software Engineering](http://www.amazon.com/Facts-Fallacies-Software-Engineering-Robert/dp/0321117425) ([PDF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.2037&rep=rep1&type=pdf)).
* Not being able to test everything is no excuse for not testing anything.
* Consider, when do you finish proof reading a paper? Learn from experience. 

Testing:

* Saves time.
* Gives confidence that code does what it is meant to.
* Promotes trust that data, and results, and research, derived from that code is correct.

Remember [Geoffrey Chang](http://en.wikipedia.org/wiki/Geoffrey_Chang)

"If it's not tested, it's broken" - Bruce Eckel, in [Thinking in Java, 3rd Edition](http://www.mindview.net/Books/TIJ/).








RESOURCES
---------

Suite for automating testing of the out of scientific codes (with custom tolerances, etc.).
http://www.cmth.ph.ic.ac.uk/people/j.spencer/code/docs/testcode/
Already used in CASTEP, one of the most widely used DFT codes in the UK and Europe.

http://maori.geek.nz/post/testing_your_code_is_doing_science
http://www.rbcs-us.com/documents/Why-Most-Unit-Testing-is-Waste.pdf
http://swcarpentry.github.io/2014-01-18-ucb/lessons/jk-python/testing.html

Python package uncertainties
"handles calculations with numbers with uncertainties"
version 2.4.2 is compatible with NumPy 1.8
https://pypi.python.org/pypi/uncertainties/
http://pythonhosted.org/uncertainties/







## When 1 + 1 = 2.0000001

    $ python
    >>> tot = 0.0
    >>> for i in range(1,100):
    >>>...  tot = tot + 0.01 # Expect answer to be 1.0
    >>>...
    >>> print tot
    >>> 0.99

Computers don't do floating point arithmetic too well. This can make simple tests for the equality of two floating point values problematic due to imprecision in the values being compared. 

    $ python
    >>> expected = 0
    >>> actual = 0.1 + 0.1 + 0.1 - 0.3
    >>> assert expected == actual
    >>> print actual

We can get round this by comparing to within a given threshold, or delta, for example we may consider *expected* and *actual* to be equal if *expected - actual < 0.000000000001*.

Test frameworks such as `nose`, often provide functions to handle this for us. For example, to test that 2 numbers are equal when rounded to a given number of decimal places,

    $ python
    >>> from nose.tools import assert_almost_equal
    >>> assert_almost_equal(expected, actual, 0)
    >>> assert_almost_equal(expected, actual, 1)
    >>> assert_almost_equal(expected, actual, 3)
    >>> assert_almost_equal(expected, actual, 6)
    >>> assert_almost_equal(expected, actual, 7)
    ...
    AssertionError: 2 != 2.0000000999999998 within 7 places

Python [decimal](http://docs.python.org/2/library/decimal.html), floating-point arithmetic functions.

    $ python
    >>> from nose.tools import assert_almost_equal
    >>> assert_almost_equal(expected, actual, 0)
    >>> assert_almost_equal(expected, actual, 10)
    >>> assert_almost_equal(expected, actual, 15)
    >>> assert_almost_equal(expected, actual, 16)

`nose.testing` uses absolute tolerance: abs(x, y) <= delta

[Numpy](http://www.numpy.org/)'s `numpy.testing` uses relative tolerance: abs(x, y) <= delta * (max(abs(x), abs(y)). 

    `assert_allclose(actual_array, expected_array, relative_tolerance, absolute_tolerance)

What do we consider to be a suitable threshold for equality? That is application-specific - for some domains we might be happy to round to the nearest whole number, for others we may want to be far, far more accurate.


Note that we assume that we are running the script from `ipython`. Each "." represents a test that has passed. A test can:

* Pass, indicated by a `.`
* Fail, indicated by an `F`
* Terminate with an Error, indicated by an `E`, that is something went wrong like your code through a segmentation fault.

%run TestHarness.py -v
%run TestHarness.py -v MyTests.test2
Can add code that will run before and after every test:
    def setUp(self):
        print "\nRunning test: ",self.id(),"\n"
    def tearDown(self):
        print "Ending test: ",self.id(),"\n"
So that execution would now look like:
%run TestHarness.py
Running test:  __main__.MyTests.test1
Ending test:  __main__.MyTests.test1
Running test:  __main__.MyTests.test2
Ending test:  __main__.MyTests.test2
This can be useful set up the *fixtures* for your tests or you could
use it to time each test:

import time
logfile ="timings.txt"
class MyTests(unittest.TestCase):
    def setUp(self):
        fh = open(logfile,"a")
        self.startTime = time.time()
        fh.write("Test %s.\n" % (self.id()))
        fh.close()
    # Run after every test.
    def tearDown(self):
        fh = open(logfile,"a")
        t  = time.time() - self.startTime
        fh.write("Time to run test: %.3f seconds.\n" % (t))
        fh.close()
Writing to a file so as not to pollute the output. The `unittest` module comes with a number of different ways that you can check that your code is working ok, for instance:
* `assertEqual(a,b)` checks that `a == b`.
* `assertNotEqual(a,b)` checks that `a != b`.
* `assertTrue(x)` checks that `x`, a boolean, is `True`.
* `assertFalse(x)` checks that x, a boolean, is `False`.
* `assertRaises(exc,fun,*args,**kwds)` checks that `fun(*args,**kwds)` raises exception `exc`.
* `assertAlmostEqual(a,b)` checks that `round(a-b,7) == 0`
* `assertNotAlmostEqual(a,b)` checks that `round(a-b,7)!=0`
def MySum(a,b)
    if(type(a) == str or type(b) == str):
       raise TypeError("Can only have integers or floats")
    return(a+b)
We can also add a test to ensure that an exception is being raised:
    from MyFunctions import MyRepeatedSum
    def testMySumExceptionArg1(self):
        self.assertRaises(TypeError,MySum,1,"a")
    def testMySumExceptionArg2(self):
        self.assertRaises(TypeError,MySum,"a",2)
def MyRepeatedSum(num,repeat):
    Sums num a number of times specified by repeat.
    tot = 0
    for i in range(repeat):
        tot += num
    return tot
Now lets import that into our test routine and do a couple of new tests:
    def testMyRepeatedSum1(self):
        self.assertEqual(MyRepeatedSum(1,100),100)
    def testMyRepeatedSum2(self):
        self.assertEqual(MyRepeatedSum(0.1,100),10.0)
When we run this we can see that we have a test failure:
======================================================================
FAIL: testMyRepeatedSum2 (__main__.MyTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "TestHarness.py", line 26, in testMyRepeatedSum2
    self.assertEqual(MyRepeatedSum(0.1,100),10.0)
AssertionError: 9.99999999999998 != 10.0
Ran 6 tests in 0.002s
FAILED (failures=1)
We have our first test failure. You should ***never*** test for equality (`==`) or inequality (`!=`) with floating numbers due to round off errors (amongst other things). We should instead use:
    def testMyRepeatedSum2(self):
         NumDecPlaces = 3 # default is 7
        self.assertAlmostEqual(MyRepeatedSum(0.1,100),10.0,NumDecPlaces)
Now all tests should pass:
import unittest
class TestLTKV(unittest.TestCase):
    def test1(self):
        pass
if __name__ == '__main__':
   unittest.main()
Now, we want to specify:
* What the script we are running is called
* What config file is called
* What the output files is going to be called
* What the reference file we are comparing this against is going to be called
* We want a function that will run the script to generate the output
* We want a test function that will compare the two files and return `True` if the two files are the same
So the test, essentially becomes:


diff Test1Output.csv outputT1.csv
2c2
< # Produced by simulate_lv.py on Mon Dec 02 15:32:34 2013
---
> # Produced by simulate_lv.py on Mon Dec 02 17:23:51 2013
```

The temptation here is to go back into the script and comment out the line that is producing the comment however this is not good practice as having that kind of provenance in your data can prove invaluable. We will just have to work a little harder in our comparison function.

Basically we want to look at a line by line comparison but ignore the remaining parts of any lines that have a `#` in them. Let's try a new version. What we want is:

```
def compareFiles(file1, file2):
    # Read the contents of each file.
    # Check we have the same number of lines
      # If not return False
    # Iterate over the lines
      # Strip out any content that begins with a hash
      # Compare lines
      # If different return False
    # Return true
```

Now all we have to do is fill in the code. This gives us:

```
def compareFiles(file1,file2):

    # open each file and read in the lines
    f1 = open(file1,"r")
    lines1 = f1.readlines()
    f1.close()
    f2 = open(file2,"r")
    lines2 = f2.readlines()
    f2.close()

    # Check we have the same number of lines else the
    # files are not the same.
    if(len(lines1) != len(lines2)):
       print "File does not have the same number of lines.\n"
       return(False)

    # Now iterate over the lines
    for i in range(len(lines1)):

        # This splits the string on a '#' character, then keeps
        # everything before the split. The 1 argument makes the .split()
        # method stop after a one split; since we are just grabbing the
        # 0th substring (by indexing with [0]) you would get the same 
        # answer without the 1 argument, but this might be a little bit 
        # faster. From steveha at http://tinyurl.com/noyk727

        lines1[i] = lines1[i].split("#",1)[0]
        line1     = lines1[i].rsplit()

        lines2[i] = lines2[i].split("#",1)[0]
        line2     = lines2[i].rsplit()

        if(line1 != line2):
           print "Line ",i+1," not the same\n",file1,":",line1,"\n",file2,
           print ": ",line2,"\n"
           return False

    # Got through to here so it appears all lines are the same.
    return True
```

If we use this routine we now find that our test passes:

```
python TestLTKV.py
.
----------------------------------------------------------------------
Ran 1 test in 0.610s

OK
```

But now we have a paradigm that we can use for multiple different number
of configuration files and output files. 

Now time for you to try and write some tests. You could try to ensure
that if the initial conditions for the prey is set to zero, the predator
numbers will decay to zero or try testing some other feature of the trial
scripts. You will find that you will deepen your understanding of the code
by doing these tests.

In essence too you will see  that you can use Python as a test harness for non-Python codes as well - in this case we used a Python script but you could have based it on a C or Fortran executable. In that case though you may have to 
look at individual elements, element by element if you are using floating point values.
