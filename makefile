VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python3.8

SRC = src/
PLOT = plot/
LOG = log/
MODEL = models/

EXE = $(SRC)main.py

EXE_ARGS = vgg1 vgg2 vgg3 dropout

.PHONY: all run runAll install activate clean checkDir

all: checkDir runAll

run: install activate
	$(PYTHON) $(EXE) vgg1 > $(LOG)stdout.vgg1.log 2> $(LOG)stderr.vgg1.log

runAll: install activate
	for arg in $(EXE_ARGS); do \
		echo Argument: $$arg; \
		$(PYTHON) $(EXE) $$arg > $(LOG)stdout.$$arg.log 2> $(LOG)stderr.$$arg.log; \
	done
	@echo Argument: vgg3 imgAgu 
	@$(PYTHON) $(EXE) vgg3 imgAgu > $(LOG)stdout.vgg3.imgAgu.log 2> $(LOG)stderr.vgg3.imgAgu.log

install: activate
	$(VENV_NAME)/bin/pip install --upgrade pip > $(LOG)stdout.python.log 2> $(LOG)stderr.python.log
	$(VENV_NAME)/bin/pip install -r requirements.txt >> $(LOG)stdout.python.log 2>> $(LOG)stderr.python.log

activate: $(VENV_NAME)
	. $(VENV_NAME)/bin/activate

$(VENV_NAME):
	python3.8 -m venv $(VENV_NAME)

checkDir: 
	mkdir -p $(LOG) $(PLOT) $(MODEL)

clean:
	rm -r $(VENV_NAME)