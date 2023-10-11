FILE=main

pdf .:
	latexmk ${FILE}.tex -silent

once 1: 
	latexmk ${FILE}.tex -silent
	make view

view v:
	open ${FILE}.pdf

clean c:
	latexmk -c
	rm -rf _minted-main .auctex-auto *.log *.run.xml *.auxlock *.synctex.gz

nuke n:
	latexmk -C
	rm -rf _minted-main .auctex-auto *.log *.run.xml *.txt *.auxlock
