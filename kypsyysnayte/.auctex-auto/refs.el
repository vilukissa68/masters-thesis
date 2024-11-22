;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "refs"
 (lambda ()
   (LaTeX-add-bibitems
    "newlib"
    "DeepLearningBook"
    "TVM"
    "Ballast"
    "tinyperf"
    "wolf08"
    "bennett2010porting"
    "linux_times_man_page"
    "rectifier"
    "he2015deepresiduallearningimage"
    "howard2017mobilenetsefficientconvolutionalneural"
    "tensorflow2015-whitepaper"
    "pytorch"
    "Cifar10Krizhevsky09learningmultiple"
    "mitchell1997machine"))
 '(or :bibtex :latex))

