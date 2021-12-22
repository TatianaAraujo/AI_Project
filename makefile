VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python3.8

SRC = src/
PLOT = plot/
data = data/

EXE = $(SRC)main.py

EXE_ARGS = vgg1 vgg2 vgg3 dropout

.PHONY: all run runAll install activate clean

all: runAll

run: install activate
	$(PYTHON) $(EXE)  vgg1

runAll: install activate
	for arg in $(EXE_ARGS); do \
		echo Argument: $$arg; \
		$(PYTHON) $(EXE) $$arg; \
	done
	@echo Argument: vgg3 imgAgu
	@$(PYTHON) $(EXE) vgg3 imgAgu

install: activate
	$(VENV_NAME)/bin/pip install --upgrade pip
	$(VENV_NAME)/bin/pip install -r requirements.txt

activate: $(VENV_NAME)
	. $(VENV_NAME)/bin/activate

$(VENV_NAME):
	python3.8 -m venv $(VENV_NAME)

clean:
	rm -r $(VENV_NAME)