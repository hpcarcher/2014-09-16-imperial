from wordcount import update_word_counts
from wordcount import load_text
from wordcount import calculate_percentages

from nose.tools import assert_equal
from nose.tools import assert_raises

import numpy as np

def test_update_word_counts():
  line = "software! software! software!"
  counts = {}
  update_word_counts(line, counts)
  assert_equal(1, len(counts))
  assert_equal(3, counts.get("software"))

def test_update_word_counts_distinct():
  line = "software carpentry software training"
  counts = {}
  update_word_counts(line, counts)
  assert_equal(3, len(counts))
  assert_equal(2, counts.get("software"))
  assert_equal(1, counts.get("carpentry"))
  assert_equal(1, counts.get("training"))

def test_calculate_percentages_invalid():
  counts = [("software", 1), ("software", -4)]
  assert_raises(AssertionError, calculate_percentages, counts)

def test_load_empty():
  lines = load_text("empty.txt")
  assert_equal(0, len(lines))

def test_data_files_equal():
  file21 = np.loadtxt("data/data2x1.dat")
  file42 = np.loadtxt("data/data4x2.dat")
  np.testing.assert_allclose(file21, file42, rtol=0, atol=1e-7)
