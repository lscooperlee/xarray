BUILDDIR ?= ./build

CHECK ?= cppcheck
CHECKFLAGS = --enable=all --language=c++ -I src --suppress=missingIncludeSystem --suppressions-list=suppressions.txt --inconclusive -q --std=c++17

MKDIR ?= mkdir -p
DIRNAME ?= dirname
RM ?= rm -rf


CPPFLAGS = -std=c++2a -O2 -Wall -Wextra -Wshadow
LDFLAGS =

LIBCPPFLAGS = 
LIBLDFLAGS = 


LIBCPPSRCS=
LIBOBJS=$(patsubst %,$(BUILDDIR)/%,$(LIBCPPSRCS:.cpp=.o))

CPPSRCS=src/main.cpp
CPPOBJS=$(patsubst %,$(BUILDDIR)/%,$(CPPSRCS:.cpp=.o))

APP=main

.PHONY:all clean

all:$(APP)

$(APP):$(CPPOBJS)
	@echo LD $@
	@$(MKDIR) `$(DIRNAME) $@`
	@$(CC) -o $(BUILDDIR)/$@ $? $(LDFLAGS)

$(CPPOBJS):$(BUILDDIR)/%.o:%.cpp
	@echo CC		$^
	@$(MKDIR) `$(DIRNAME) $@`
	@$(CC) $(CPPFLAGS) -c -o $@ $<

test:
	@make -C tests BUILD=$(abspath $(BUILDDIR))/tests

clean:
	$(RM) $(BUILDDIR)

check:
	@$(CHECK) $(CHECKFLAGS) .

