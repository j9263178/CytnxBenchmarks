LIBRARY_DIR=~/itensor

APP=dmrg

MYFLAGS=-fmax-errors=3 -lrt
HEADERS=dmrg.h localmpo.h localop.h

CCFILES=$(APP).cc

#################################################################
#################################################################
#################################################################
#################################################################

include $(LIBRARY_DIR)/this_dir.mk
include $(LIBRARY_DIR)/options.mk

#Mappings --------------
OBJECTS=$(patsubst %.cc,%.o, $(CCFILES))
GOBJECTS=$(patsubst %,.debug_objs/%, $(OBJECTS))

#Rules ------------------

%.o: %.cc $(HEADERS)
	$(CCCOM) -c $(CCFLAGS) $(MYFLAGS) -o $@ $<

.debug_objs/%.o: %.cc $(HEADERS)
	$(CCCOM) -c $(CCGFLAGS) $(MYFLAGS) -o $@ $<

#Targets -----------------

build: $(APP)
debug: $(APP)-g

$(APP): $(OBJECTS) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS) -o $(APP) $(LIBFLAGS) $(MYFLAGS)

$(APP)-g: mkdebugdir $(GOBJECTS) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS) -o $(APP)-g $(LIBGFLAGS) $(MYFLAGS)

clean:
	rm -fr .debug_objs *.o $(APP) $(APP)-g

mkdebugdir:
	mkdir -p .debug_objs

