FILE=main

pdf:
	latexmk ${FILE}.tex -silent -pvc
	make clean 

view:
	open ${FILE}.pdf

clean:
	latexmk -c
	rm -rf _minted-main .auctex-auto *.log *.run.xml *.auxlock

nuke:
	latexmk -C
	rm -rf _minted-main .auctex-auto *.log *.run.xml *.txt *.auxlock
