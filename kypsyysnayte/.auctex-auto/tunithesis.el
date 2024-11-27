;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "tunithesis"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("graphicx" "") ("psfrag" "") ("subfigure" "") ("wrapfig" "") ("fancyhdr" "") ("supertabular" "") ("rotating" "") ("amsmath" "") ("setspace" "") ("caption" "font=small" "it" "labelsep=space") ("hyperref" "") ("url" "") ("listings" "") ("color" "")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "report"
    "rep10"
    "graphicx"
    "psfrag"
    "subfigure"
    "wrapfig"
    "fancyhdr"
    "supertabular"
    "rotating"
    "amsmath"
    "setspace"
    "caption"
    "hyperref"
    "url"
    "listings"
    "color")
   (TeX-add-symbols
    '("makelabel" 1)
    "chapterheadbefore"
    "chapterheadafter"
    "chapfigname"
    "chaptertocstyle"
    "chaptertocstr"
    "l"
    "thesistype"
    "sectionmark"
    "chaptermark")
   (LaTeX-add-environments
    "termlist")
   (LaTeX-add-pagestyles
    "headings")
   (LaTeX-add-lengths
    "chapterspace")
   (LaTeX-add-listings-lstdefinestyles
    "a1listing"
    "console")
   (LaTeX-add-color-definecolors
    "gray95"))
 :latex)

