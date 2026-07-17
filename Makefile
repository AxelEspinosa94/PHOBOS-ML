CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude

SRC = src/main.c
OUT = build/phobosml

all: $(OUT)

$(OUT): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

clean:
	rm -rf build/*
