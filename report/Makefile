FILE=main

pdf:
	latexmk ${FILE}.tex -silent -pvc

once:
	latexmk ${FILE}.tex -silent
	make view

view:
	open ${FILE}.pdf

clean:
	latexmk -c
	rm -rf _minted-main .auctex-auto *.log *.run.xml *.auxlock *.synctex.gz

nuke:
	latexmk -C
	rm -rf _minted-main .auctex-auto *.log *.run.xml *.txt *.auxlock
