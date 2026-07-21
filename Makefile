# ================================
# PHOBOS-ML Makefile (PRO)
# ================================

# Directories
SRC_DIR := src
TEST_DIR := tests
BUILD_DIR := build
INC_DIR := include

# Compiler
CC := gcc
CFLAGS := -Wall -Wextra -O2 -I$(INC_DIR)

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)
TESTS := $(wildcard $(TEST_DIR)/*.c)

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

# Remove main.o from engine objects for tests
ENGINE_OBJS := $(filter-out $(BUILD_DIR)/main.o, $(OBJS))

TEST_OBJS := $(patsubst $(TEST_DIR)/%.c, $(BUILD_DIR)/%.o, $(TESTS))

# Binaries
TARGET := $(BUILD_DIR)/phobosml

# ================================
# Default target
# ================================
all: $(TARGET)

# ================================
# Main binary
# ================================
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)

# ================================
# Test binaries (auto-detected)
# ================================
# Example: build/test_tensor
$(BUILD_DIR)/test_%: $(BUILD_DIR)/test_%.o $(ENGINE_OBJS)
	$(CC) $(BUILD_DIR)/test_$*.o $(ENGINE_OBJS) -o $(BUILD_DIR)/test_$*

# ================================
# Generic compilation rules
# ================================
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ================================
# Formatting
# ================================
format:
	clang-format-18.1.3 -i $(SRC_DIR)/*.c $(TEST_DIR)/*.c $(INC_DIR)/*.h

# ================================
# Clean
# ================================
clean:
	rm -f $(BUILD_DIR)/*.o
	rm -f $(TARGET)
	rm -f $(BUILD_DIR)/test_*
