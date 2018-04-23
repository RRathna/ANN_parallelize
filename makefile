CCFLAGS = -Wall -Wshadow -O2 -g
LDLIBS = -lm


all: example4

example4: example4.o genann.o


clean:
	$(RM) *.o
	$(RM) *.exe
	$(RM) persist.txt
