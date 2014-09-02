include config.mk

CFG_FILES=$(shell find configs -type f -name '*.cfg')
DAT_FILES=$(patsubst configs/%.cfg, %.dat, $(CFG_FILES))
JPG_FILES=$(patsubst configs/%.cfg, %.jpg, $(CFG_FILES))

.PHONY : dats
dats : $(DAT_FILES)

.PHONY : jpgs
jpgs : $(JPG_FILES)

# Determine flow pattern.
%.dat : configs/%.cfg $(CFD_SRC)
	$(CFD_EXE) $< $@

# Plot flow.
%.jpg : %.dat $(PLOT_SRC)
	$(PLOT_EXE) $< $@

.PHONY : all
all : jpgs

analysis.tar.gz : $(DAT_FILES) $(JPG_FILES) $(CFD_SRC) $(PLOT_SRC)
	tar -czf $@ $^

clean : 
	rm -f analysis.tar.gz
	rm -f *.dat
	rm -f *.jpg
