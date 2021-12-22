VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python3.8

SRC = src/
PLOT = plot/
LOG = log/
MODEL = models/

EXE = $(SRC)main.py

EXE_ARGS = vgg1 vgg2 vgg3 dropout

.PHONY: all run runAll install activate clean

all: $(PLOT) $(LOG) $(MODEL) run

run: install activate
	$(PYTHON) $(EXE)  vgg1 > $(LOG)exec.vgg1.log 2> $(LOG)error.vgg1.log

runAll: install activate
	for arg in $(EXE_ARGS); do \
		echo Argument: $$arg; \
		$(PYTHON) $(EXE) $$arg > $(LOG)exec.$$arg.log 2> $(LOG)error.$$arg.log; \
	done
	@echo Argument: vgg3 imgAgu 
	@$(PYTHON) $(EXE) vgg3 imgAgu > $(LOG)exec.vgg3.imgAgu.log 2> $(LOG)error.vgg3.imgAgu.log

install: activate
	$(VENV_NAME)/bin/pip install --upgrade pip
	$(VENV_NAME)/bin/pip install -r requirements.txt

activate: $(VENV_NAME)
	. $(VENV_NAME)/bin/activate

$(VENV_NAME):
	python3.8 -m venv $(VENV_NAME)

$(LOG) $(PLOT) $(MODEL):
	mkdir -p $@

clean:
	rm -r $(VENV_NAME)