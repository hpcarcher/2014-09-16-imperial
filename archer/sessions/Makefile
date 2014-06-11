include config.mk

TXT_FILES=$(shell find . -type f -name '*.txt')
DAT_FILES=$(patsubst %.txt, %.dat, $(TXT_FILES))
JPG_FILES=$(patsubst %.txt, %.jpg, $(TXT_FILES))

.PHONY : dats
dats : $(DAT_FILES)

.PHONY : jpgs
jpgs : $(JPG_FILES)

# Calculate word frequencies.
%.dat : %.txt $(PROCESSOR)
	python $(PROCESSOR) $< > $@

# Calculate images
%.jpg : %.dat $(PLOTTER)
	python $(PLOTTER) -save $< $@

analysis.tar.gz : *.dat *.jpg $(PROCESSOR)
	tar -czf $@ $^
