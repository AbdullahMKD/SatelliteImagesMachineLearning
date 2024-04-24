# LaTeX Template for MMP Report

This folder contains a LaTeX template for the project report.

There is one combined template for the Engineering-oriented and Research-oriented projects. The organisation of the files has been designed to make it easier to maintain this resource.

## Engineering or Research

Within `mmp-report.tex`, there is a defintion for `\reporttype`. This can have one of two values: `Engineering` or `Research`. This value is used to determine which of the two `Chapters_x` folders are read, where x is one of `Engineering` or `Research`.

The intention is that you only use chapters from one of these two directories. They are handled in this way to make it easier to manage the production of the template. If you are doing an engineering-oriented project, then you do not need to keep the research-oriented files.

## Colours

The title, chapter and section headings are assigned a blue colour within the template. To control that colour and, if you prefer, remove the colour to simply use the default font colour, you can adjust the setting in `mmp-report.tex`.  Look for the comments on `Title and Section Colours`.

## Style File

The LaTeX style file is defined in  `StylesAndReferences/mmp-report.sty`.  Editing this file is for those who are advanced users of LaTeX or the curious who are willing to learn something new. One area you may want to modify is the default choice of font. The document is setup to use `tgheros`, a sans serif font. You may prefer a different font or even a serif font. Look for the `% Fonts` comment in the style file to find the elements to change.

## Text alignment

The default text alignment in a LaTeX document is to justify the text. That can cause a problem for some people, such as those with dyslexia. The LaTeX command `\raggedright` has been added to the main report document.

## References

The set of example references is contained in the file `StylesAndReferences/references.bib`. This has been manually created but it could easily be something that is generated from a reference management tool such as Mendeley.

Two example reference styles are provided for in the `mmp-report.tex` file. One is an author-date style (`StylesAndReferences/authordate2annot`) and one is an IEEE style (`StylesAndReferences/IEEEannotU`).

The two styles are capable of producing annotated references. For projects submitted in 2022, it has been decided that we will not require annotated references. The two style files have been updated to stop the generation of the annotations.

## Source of help for LaTeX

The internet is not short of good resources to help you get the most out of LaTeX. One useful resource is hosted by [Overleaf](https://www.overleaf.com/learn/latex/Main_Page), which is also an online LaTeX editor.

**Note:** an earlier version of MMP project templates have been available on Overleaf, but they are not maintained. Please start with these files and not anything that you find in Overleaf.

You can also ask questions about LaTeX in the Blackboard Discussion area for the module.

Neil Taylor
12th March 2024
