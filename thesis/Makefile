render-xelatex:
	xelatex main.tex
	biber main
	xelatex main.tex
	xelatex main.tex

render-pdflatex:
	pdflatex main.tex
	biber main
	pdflatex main.tex
	pdflatex main.tex

render-no-biber:
	xelatex main.tex


clean:
	rm report.aux report.bbl report.bcf report.blg report.log report.out report.run.xml report.tex.bbl report.tex.blg report.toc texput.log
