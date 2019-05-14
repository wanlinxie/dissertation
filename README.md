This project is forked from: https://github.com/grimpil/dissertation


NOTES FOR INSTALLATION


-----------------------------------------------------------
CREATING VIRTUAL ENVIRONMENT, INSTALLING DEPENDENCIES

* can only use /usr/bin/python to run program
create virtual env:
/usr/bin/python -m virtualenv env
source env/bin/activate

pip install: (TODO: create Requirements.txt)
pyutilib
numpy
scipy
unidecode
nltk
psutil
stemming
matplotlib


-----------------------------------------------------------
INSTALLING SRILM, SWIG, AND SWIG_SRILM (to create language model)

install srilm (http://www.speech.sri.com/projects/srilm/)
- set SRILM variable in Makefile to point to srilm directory
- make World MAKE_PIC=yes
go to srilm/lm/bin/(either macosx or i686-m64)
to create language model, run: 
./ngram-count -text /path/to/data.txt -order 3 -lm /path/to/test.lm


install swig (http://swig.org/doc.html)
- can only run this project using the default python interpreter due to restrictions on the srilm-swig library (this is /usr/bin/python for mac)
installation instructions:
$ ./configure --prefix=/project/path
$ make
$ make install


install swig-srilm: 
git clone https://github.com/desilinguist/swig-srilm.git
rename swig-srilm to swig_srilm


-----------------------------------------------------------
INSTALLING GUROBI

- download: http://www.gurobi.com/index

- follow instructions here:
http://www.gurobi.com/documentation/8.1/quickstart_linux.pdf

- get free academic license


-----------------------------------------------------------
in text/annotations/__init__.py:
replace (~line 14): module_spec = imp.find_module('text/annotations/' + mod_name)
with: module_spec = imp.find_module(mod_name,['full path'])

change datapath in pyrfusion and compression files 
- make data/compression folder
- make data/pyrfusion/corpora folder

mkdir tools
mkdir tools/tagchunk
mkdir tools/stanford-parser
mkdir tooks/rasp3os/scripts

in text/annotations/tagchunk.py:
replace (~line 15) with your made directory for tools/tagchunk

in text/annotations/stanford.py:
replace (~line 15) with your made directory for tools/stanford-parser

in text/annotations/rasp.py:
replace (~line 15) with your made directory for tools/rasp3os/scripts

