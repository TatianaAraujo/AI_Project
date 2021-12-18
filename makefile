VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python3.8

SRC = src/
PLOT = plot/
data = data/

EXE = $(SRC)main.py

EXE_ARGS = vgg1 vgg2 vgg3 dropout vgg3ImgAgu

run: install activate
	$(PYTHON) $(EXE)  vgg1

runAll:
	for arg in $(EXE_ARGS); do \
		echo Argument: $$arg; \
		$(PYTHON) $(EXE) $$arg; \
	done

install: activate
	$(VENV_NAME)/bin/pip install --upgrade pip
	$(VENV_NAME)/bin/pip install -r requirements.txt

activate: $(VENV_NAME)
	. $(VENV_NAME)/bin/activate

$(VENV_NAME):
	python3.8 -m venv $(VENV_NAME)

clean:
	rm -r $(VENV_NAME)