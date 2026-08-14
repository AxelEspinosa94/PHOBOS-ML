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
CFLAGS += -Wall -Wextra -O2 -I$(INC_DIR)

# Linker flags (math library)
LDLIBS := -lm

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)
TESTS := $(wildcard $(TEST_DIR)/*.c)

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/src_%.o, $(SRCS))
TEST_OBJS := $(patsubst $(TEST_DIR)/%.c, $(BUILD_DIR)/test_%.o, $(TESTS))
ENGINE_OBJS := $(filter-out $(BUILD_DIR)/src_main.o, $(OBJS))

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
	$(CC) $(OBJS) -o $(TARGET) $(LDLIBS) $(LDFLAGS)

# ================================
# Test binaries (auto-detected)
# ================================
# Example: build/test_tensor
$(BUILD_DIR)/test_%: $(BUILD_DIR)/test_%.o $(ENGINE_OBJS)
	$(CC) $(BUILD_DIR)/test_$*.o $(ENGINE_OBJS) -o $(BUILD_DIR)/test_$* $(LDLIBS) $(LDFLAGS)

# ================================
# Generic compilation rules
# ================================
# Engine objects
$(BUILD_DIR)/src_%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Test objects
$(BUILD_DIR)/test_%.o: $(TEST_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ================================
# Formatting
# ================================
format:
	clang-format-18 -i $(SRC_DIR)/*.c $(TEST_DIR)/*.c $(INC_DIR)/*.h

# ================================
# Clean
# ================================
clean:
	rm -f $(BUILD_DIR)/*.o
	rm -f $(TARGET)
	rm -f $(BUILD_DIR)/test_*
