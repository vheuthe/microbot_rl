# Simple makefile to compile all fortran modules
# (adopted from https://en.wikipedia.org/wiki/Make_(software))

# look for all fortran files in the directory
F95_FILES := $(wildcard *.f95)
# determin the platform specific python module extension
PY_MODULE_SUFFIX := $(shell python3-config --extension-suffix)
# substitude to generate python module names
PY_MODULES := $(patsubst %.f95, %$(PY_MODULE_SUFFIX), $(F95_FILES))


.PHONY: all
all: $(PY_MODULES)

%$(PY_MODULE_SUFFIX): %.f95
	f2py3 -c $*.f95 -m $*

.PHONY: clean
clean:
	-rm -f $(PY_MODULES)