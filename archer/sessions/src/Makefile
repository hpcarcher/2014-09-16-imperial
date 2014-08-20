include config.mk

TXT_FILES=$(shell find books -type f -name '*.txt')
DAT_FILES=$(patsubst books/%.txt, %.dat, $(TXT_FILES))
JPG_FILES=$(patsubst books/%.txt, %.jpg, $(TXT_FILES))

.PHONY : dats
dats : $(DAT_FILES)

.PHONY : jpgs
jpgs : $(JPG_FILES)

# Calculate word frequencies.
%.dat : books/%.txt $(COUNTER)
	python $(COUNTER) $< $@

# Calculate images
%.jpg : %.dat $(PLOTTER)
	python $(PLOTTER) $< $@

analysis.tar.gz : *.dat *.jpg $(COUNTER)
	tar -czf $@ $^

clean : 
	rm -f analysis.tar.gz
	rm -f *.dat
	rm -f *.jpg
