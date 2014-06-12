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

Shell-based testing
-------------------

`diff` compares files for equality.

    $ diff abyss.txt kim.txt

    $ mkdir data/
    $ cp *.txt data
    $ python wordcount.py abyss.txt  > abyss.dat
    $ python wordcount.py bridge.txt  > bridge.dat
    $ python wordcount.py kim.txt  > kim.dat
    $ python wordcount.py war.txt  > war.dat
    $ mkdir expected/
    $ mv *.dat expected/

Create `test_word_count.sh`:

    #!/bin/sh
    rm -rf tmpdats
    mkdir tmpdats
    echo "Test 1"
    python wordcount.py data/abyss.txt > tmpdats/abyss.dat
    diff -rq expected/abyss.dat tmpdats/abyss.dat

    $ chmod +x test_word_count.sh
    $ ./test_word_count.sh

Exercise 1 - add more tests
---------------------------

See [exercises](TestExercises.md).

Solution:

    echo "Test 2"
    python wordcount.py data/bridge.txt > tmpdats/bridge.dat
    diff -rq expected/bridge.dat tmpdats/bridge.dat
    echo "Test 3"
    python wordcount.py data/war.txt > tmpdats/war.dat
    diff -rq expected/war.dat tmpdats/war.dat

Refactor
--------

Remove repeated code and modularise into reusable functions.

    # $1 input data file
    # $2 output data file
    # $3 expected data file
    test_wordcount() {
      python wordcount.py $1 > $2
      check_outputs $3 $2
    }

    # $1 file to compare
    # $2 file to compare
    check_outputs() {
      compare=`diff -rq $1 $2`
      if [ -z "$compare" ]; then
        echo "."
      else
        echo "FAILURE: $compare"
      fi
    }

    rm -rf tmptests
    mkdir tmptests

    for file in $(ls data/*.txt); do
      filename=`basename $file .txt`
      test_wordcount $file tmptests/$filename.dat expected/$filename.dat
    done

Test for failures
-----------------

Test for return code. Expect non-zero return code if script fails:

    python wordcount.py
    EXIT=$? # Capture exit code.
    if [ $EXIT -ne 0 ]; then
        echo "."
    else
        echo "FAILURE: exit code should be non-zero"
    fi

Exercise 2 - recode `wordcount.py`
----------------------------------

See [exercises](TestExercises.md).

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

Add meta-data to the output:

    import datetime

    print "# Frequency data"
    print "# Created by:",  __file__
    print "# Input data:", file
    print "# Date:", today
    print "# Format: word frequency"

Run:

    ./test-word-count.sh

Question: what is the problem?

Answer: the meta-data. 

Will typically want finer grained tests of equality between data files.


TODO
TODO
TODO



Programming tests
-----------------

Shell problems:
* diff
* return code
* granularity is coarse
* Prechelt and language.


WRITE FILE VALIDATOR IN PYTHON!!!!!


Add more information to the failure messages by providing additional string arguments e.g.

    >>> assert_true("GTA" in actual, "Expected value was not in the output list")

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



Add in an end to end-called in-code test
----------------------------------------

What if tests not function based?
Redesign
Capture inputs and outputs e.g. FABBER

finer grained tests
----------

But can drill in as code is Python and tests in Python


Write unit test for freq

    def test_encode_sos(self):
        expected = "... --- ..."
        actual = self.translator.encode("SOS")                     
        assert expected == actual

Defensive programming
---------------------

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

Testing for failures and robustness
-----------------------------------

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

Automated testing jobs
----------------------

Link to revision control

The [Muon Ion Cooling Experiment](http://www.mice.iit.edu/) (MICE) have a large number of tests written in Python. They use [Jenkins](), a *continuous integration server* to build their code and trigger the running of the tests which are then [published online](https://micewww.pp.rl.ac.uk/tab/show/maus).
Continuous integration server e.g. [Jenkins](http://jenkins-ci.org/) - detect commit to version control, build, run tests, publish.

[Muon Ion Cooling Experiment](http://www.mice.iit.edu/) (MICE) - Bazaar version control, Python tests, Jenkins, [published online](https://micewww.pp.rl.ac.uk/tab/show/maus).

[Apache Hadoop Common Jenkins dashboard](https://builds.apache.org/job/Hadoop-Common-trunk/)

Tests are code
--------------

Tests should be reviewed for the same reasons code should be reviewed.

Avoid tests that:

* Pass when they should fail, false positives.
* Fail when they should pass, false negatives.
* Don't test anything. 

For example,

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

https://code.google.com/p/shunit2/ 
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

## Structure

Fixture: what the test is run on
Action: what we do to the fixture
Expected result: result that should happen
Actual result: what actually happen
Report: summary

## Let's start writing a python test harness

Going to use the `unittest` framework. This has been in python since python 2.6. Create a `TestHarness.py` file and type in the following:

```
import unittests

class MyTests(unittest.TestCase):

    def test1(self):
        pass

    def test2(self):
        pass

if __name__ == '__main__':

   unittest.main()
```

A test case is defined by subclassing `unittest.TestCase`. We have defined two rather uninteresting tests. If we run this we get:

```
%run TestHarness.py
..
----------------------------------------------------------------------
Ran 2 tests in 0.002s

OK
```

Note that we assume that we are running the script from `ipython`. Each "." represents a test that has passed. A test can:

* Pass, indicated by a `.`
* Fail, indicated by an `F`
* Terminate with an Error, indicated by an `E`, that is something went wrong like your code through a segmentation fault.

Also if you want more detail you can run your tests using the `-v` flag then you would get:

```
%run TestHarness.py -v
test1 (__main__.MyTests) ... ok
test2 (__main__.MyTests) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.007s

OK
```
Or you can run a single test if you wish:
```
%run TestHarness.py -v MyTests.test2
test2 (__main__.MyTests) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.004s

OK
```
Can add code that will run before and after every test:
```
    def setUp(self):
        print "\nRunning test: ",self.id(),"\n"

    def tearDown(self):
        print "Ending test: ",self.id(),"\n"
```
So that execution would now look like:
```
%run TestHarness.py

Running test:  __main__.MyTests.test1

Ending test:  __main__.MyTests.test1

.
Running test:  __main__.MyTests.test2

Ending test:  __main__.MyTests.test2

.
----------------------------------------------------------------------
Ran 2 tests in 0.020s

OK
```
This can be useful set up the *fixtures* for your tests or you could
use it to time each test:

```
import time

logfile ="timings.txt"
...
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
...
```
Writing to a file so as not to pollute the output. The `unittest` module comes with a number of different ways that you can check that your code is working ok, for instance:

* `assertEqual(a,b)` checks that `a == b`.
* `assertNotEqual(a,b)` checks that `a != b`.
* `assertTrue(x)` checks that `x`, a boolean, is `True`.
* `assertFalse(x)` checks that x, a boolean, is `False`.
* `assertRaises(exc,fun,*args,**kwds)` checks that `fun(*args,**kwds)` raises exception `exc`.
* `assertAlmostEqual(a,b)` checks that `round(a-b,7) == 0`
* `assertNotAlmostEqual(a,b)` checks that `round(a-b,7)!=0`

Let's make it a little more interesting. Create a new python file: `MyFunctions.py` and add a function:

```
def MySum(a,b):
    return(a+b)
```
Now we want to import this function into our test harness. Add the line at the top of `TestHarness.py`:
```
from MyFunctions import MySum
```
We can now add a test to check this works:
```
    def testMySum(self):
        self.assertEqual(MySum(1,3),4)
```     
That should have worked. What happens if we try the following?

```
    MySum("a","b")
    "ab"
```
It may well be that that is perfectly acceptable behaviour but, on the other hand, you may only want numbers to be added and not to have string concatenation. We can change the sum function accordingly:
```
def MySum(a,b)
    if(type(a) == str or type(b) == str):
       raise TypeError("Can only have integers or floats")
    return(a+b)
```
We can also add a test to ensure that an exception is being raised:
```
    from MyFunctions import MyRepeatedSum
     ...
    def testMySumExceptionArg1(self):
        self.assertRaises(TypeError,MySum,1,"a")

    def testMySumExceptionArg2(self):
        self.assertRaises(TypeError,MySum,"a",2)
```
We can keep on playing these games and add further tests for `MySum` but let's define a new function:
```
def MyRepeatedSum(num,repeat):
    """
    Sums num a number of times specified by repeat.
    """
    tot = 0
    for i in range(repeat):
        tot += num
    return tot
```
Now lets import that into our test routine and do a couple of new tests:
```
    def testMyRepeatedSum1(self):
        self.assertEqual(MyRepeatedSum(1,100),100)

    def testMyRepeatedSum2(self):
        self.assertEqual(MyRepeatedSum(0.1,100),10.0)
```
When we run this we can see that we have a test failure:
```
..F...
======================================================================
FAIL: testMyRepeatedSum2 (__main__.MyTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "TestHarness.py", line 26, in testMyRepeatedSum2
    self.assertEqual(MyRepeatedSum(0.1,100),10.0)
AssertionError: 9.99999999999998 != 10.0

----------------------------------------------------------------------
Ran 6 tests in 0.002s

FAILED (failures=1)
```
We have our first test failure. You should ***never*** test for equality (`==`) or inequality (`!=`) with floating numbers due to round off errors (amongst other things). We should instead use:

```
    def testMyRepeatedSum2(self):
         NumDecPlaces = 3 # default is 7
        self.assertAlmostEqual(MyRepeatedSum(0.1,100),10.0,NumDecPlaces)
```
Now all tests should pass:
```
......
----------------------------------------------------------------------
Ran 6 tests in 0.002s

OK
```
This then forms the basics of how one might go on to develop a test framework using Python. We shall come back to this later but lets have a look at `nosetests`.

Start by pulling an updated version of the scripts. In the students directory you checked out from GitHub run the following commands:

Will use the `unittest` framework as that can be used on its own and with unit tests. Call this script `TestLTKV.py`

import unittest

class TestLTKV(unittest.TestCase):

    def test1(self):
        pass


if __name__ == '__main__':

   unittest.main()
```

Now, we want to specify:

* What the script we are running is called
* What config file is called
* What the output files is going to be called
* What the reference file we are comparing this against is going to be called
* We want a function that will run the script to generate the output
* We want a test function that will compare the two files and return `True` if the two files are the same

So the test, essentially becomes:

```
    def test1(self):
        script  = "simulate_lv.py"
        config  = "Test1Config.cfg"
        outfile = "outputT1.csv"
        reffile = "Test1Output.csv"
        runTest(script, config, outfile)
        self.assertTrue(compareFiles(reffile,outfile))
```

Let's start by looking at the `compareFiles()` function. There is a file comparison function in Python so we can leverage off that:

```
import filecmp

def compareFiles(file1,file2):
    return filecmp.cmp(file1,file2)
```

So, that is fairly straightforward BUT is a bit naive, for instance it will NOT work if you are expecting floating point differences. In that case, you would have to employ a much more sophisticated approach where you have to inspect the
inside of each file and compare element by element showing that they are
the same within a given tolerance - to get an idea as to how you might do 
that have a look at the file [regression_test.py](http://depts.washington.edu/clawpack/users/claw/python/pyclaw/regression_test.py).

Now, let's look at how we might run the code. For this we use the `subprocess` module in Python:

```
import subprocess

def runTest(script,config,outfile):
    subprocess.call(["python",script,config,outfile])

```

So the whole script now looks like:

```
import unittest
import filecmp
import subprocess

def runTest(script,config,outfile):
    subprocess.call(["python",script,config,outfile])

def compareFiles(file1,file2):
    return filecmp.cmp(file1,file2)

class TestLTKV(unittest.TestCase):

    def test1(self):
        script  = "simulate_lv.py"
        config  = "Test1Config.cfg"
        outfile = "outputT1.csv"
        reffile = "Test1Output.csv"
        runTest(script, config, outfile)
        self.assertTrue(compareFiles(reffile,outfile))


if __name__ == '__main__':

   unittest.main()
```

So let's run the script:

```
python TestLTKV.py
F
======================================================================
FAIL: test1 (__main__.TestLTKV)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "TestLTKV.py", line 19, in test1
    self.assertTrue(compareFiles(reffile,outfile))
AssertionError: False is not true

----------------------------------------------------------------------
Ran 1 test in 0.608s

FAILED (failures=1)
```

Oops, what happened there? If you do a `diff` between the two files you get:

```
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
