BINARY_NAME=main
INPUT_FILES=-I Eigen/ -O2 -o main *.*pp

all: build

build:
	g++ $(INPUT_FILES) -o $(BINARY_NAME)

run:
	./$(BINARY_NAME) $(ARGS)

memcheck: 
	valgrind --tool=memcheck ./$(BINARY_NAME) $(ARGS)