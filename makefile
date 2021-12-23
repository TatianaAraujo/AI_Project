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
	for arg in $(EXE_ARGS); do \
		echo Argument: --model $$arg --pandas; \
		$(PYTHON) $(EXE) --model $$arg --pandas > $(LOG)stdout.$$arg.pandas.log 2> $(LOG)stderr.$$arg.pandas.log; \
	done
	@echo Argument: --model vgg3 --imgAgu --pandas
	@$(PYTHON) $(EXE) --model vgg3 --imgAgu --pandas > $(LOG)stdout.vgg3.imgAgu.pandas.log 2> $(LOG)stderr.vgg3.imgAgu.pandas.log

runAll: install activate
	for arg in $(EXE_ARGS); do \
		echo Argument: --model $$arg; \
		$(PYTHON) $(EXE) --model $$arg > $(LOG)stdout.$$arg.log 2> $(LOG)stderr.$$arg.log; \
	done
	@echo Argument: --model vgg3 --imgAgu 
	@$(PYTHON) $(EXE) --model vgg3 --imgAgu > $(LOG)stdout.vgg3.imgAgu.log 2> $(LOG)stderr.vgg3.imgAgu.log

install: activate
	$(VENV_NAME)/bin/pip install --upgrade pip > $(LOG)stdout.python.log 2> $(LOG)stderr.python.log
	$(VENV_NAME)/bin/pip install -r requirements.txt >> $(LOG)stdout.python.log 2>> $(LOG)stderr.python.log

activate: $(VENV_NAME)
	. $(VENV_NAME)/bin/activate

$(VENV_NAME):
	python3.8 -m venv $(VENV_NAME)

checkDir: install activate
	mkdir -p $(LOG) $(PLOT) $(MODEL)
	$(PYTHON) $(EXE) --onlyCreateDir

clean:
	rm -r $(VENV_NAME)
