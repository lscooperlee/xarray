BUILD?=build

CPPFLAGS=-std=c++2a -O2 -Wall -Wextra -Wshadow

CPPSRCS=$(shell find . -name "*.cpp")
CPPOBJS=$(patsubst ./%, $(BUILD)/%, $(CPPSRCS:.cpp=.o))

UNITTESTEXEC=$(BUILD)/unit_test

all: $(UNITTESTEXEC)

$(UNITTESTEXEC): $(CPPOBJS)
	@echo LD 	$@
	@$(CXX) $(STATICFLAG) $(CPPFLAGS) $(CPPOBJS) -o $@ $(LDFLAGS)

$(CPPOBJS) :$(BUILD)/%.o: %.cpp
	@echo CXX 	$@
	@mkdir -p `dirname $@`
	$(CXX) $(CPPFLAGS) -c $< -o $@

clean:
	@rm -rf $(UNITTESTEXEC) $(CPPOBJS) $(BUILD)
