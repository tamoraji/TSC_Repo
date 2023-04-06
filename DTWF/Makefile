OBJECTS = util.o cache.o example.o experiment.o Euclidean_classifier.o SAX_classifier.o DTW_classifier.o DTW_R_classifier.o SVM_classifier.o
INCLUDES = util.h cache.h example.h experiment.h Euclidean_classifier.h SAX_classifier.h DTW_classifier.h DTW_R_classifier.h SVM_classifier.h
CC=g++ -std=c++0x -O3 
all: main
main: main.cpp $(OBJECTS) $(INCLUDES)
	$(CC) $(OBJECTS) main.cpp -o main
util.o: util.h util.cpp
	$(CC) -c util.cpp 
cache.o: cache.h cache.cpp
	$(CC) -c cache.cpp 
example.o: util.h example.h example.cpp
	$(CC) -c example.cpp 
experiment.o: util.h example.h experiment.h experiment.cpp
	$(CC) -c experiment.cpp 	
Euclidean_classifier.o: example.h Euclidean_classifier.h Euclidean_classifier.cpp
	$(CC) -c Euclidean_classifier.cpp 
SAX_classifier.o: example.h SAX_classifier.h SAX_classifier.cpp
	$(CC) -c SAX_classifier.cpp 
DTW_classifier.o: example.h DTW_classifier.h DTW_classifier.cpp
	$(CC) -c DTW_classifier.cpp 
DTW_R_classifier.o: example.h DTW_R_classifier.h DTW_R_classifier.cpp
	$(CC) -c DTW_R_classifier.cpp 
SVM_classifier.o: example.h SVM_classifier.h SVM_classifier.cpp
	$(CC) -c SVM_classifier.cpp 
clean:
	rm -f $(OBJECTS) main
