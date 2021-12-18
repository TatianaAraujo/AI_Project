VENV = env
PYTHON = $(venv)/bin/python3.8
PIP = $(venv)/bin/pip

SRC = src/
PLOT = plot/
data = data/

EXE = main.py

.PHONY: all run runAll help

all: $(VENV)/bin/activate

run: $(VENV)/bin/activate
	$(PYTHON) $(SRC)$(EXE) vgg1

runAll: $(VENV)/bin/activate
	$(PYTHON) $(SRC)$(EXE) vgg1
	$(PYTHON) $(SRC)$(EXE) vgg2
	$(PYTHON) $(SRC)$(EXE) vgg3

$(VENV)/bin/activate: requirements.txt
	python3.8 -m venv .venv 
	. $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

help:
	@echo "Usage: make"

clean:
	rm -rf __pyache__

