The KU Leuven Engineering Master's Thesis Class
===============================================
The kulemt LaTeX class helps you to format your Master's thesis according to
the rules of the KU Leuven Faculty of Engineering Science. It can be adapted
to other Master's thesis layouts by tweaking a configuration file.

**Warning**: Since the template changes almost every year, it is not a good
idea to use templates available elsewhere because they are probably outdated.
Please do not use the templates found on Overleaf!

Bug reports and remarks can be sent to
[Luc Van Eycken](https://www.kuleuven.be/wieiswie/en/person/00010701).


Table of contents
-----------------
* Documentation files
* Installation instructions


Documentation files
-------------------
  * [`changes.txt`](changes.txt) (aka "What's new"): a list of changes
  * [`guidelines_thesis.pdf`](guidelines_thesis.pdf): guidelines for writing
    a Master's thesis at the KU Leuven Faculty of Engineering Science
  * [`kulemt.pdf`](kulemt.pdf): user manual of kulemt
    (KU Leuven Engineering Master Thesis document class)
  * [`kulemt-src.pdf`](kulemt-src.pdf): documented source
    (of `kulemt.cls` and `kulemt.cfg`)
  * [`kulemtx.pdf`](kulemtx.pdf): user manual and documented source
    of the package kulemtx (kulemt extension)
  * `sjabloon`: directory with Dutch templates
  * `template`: directory with English templates

All documentation is available from the [`kulemt-doc.zip`](kulemt-doc.zip)
file or from the `doc/latex/kulemt` directory in the
[`kulemt-tds.zip`](kulemt-tds.zip) file. To locate it on your system after
installation, read the installation instructions below.


Installation instructions
-------------------------

Note: All source files can be found on the
[LaTeX template website](https://eng.kuleuven.be/docs/kulemt).

### Quick and dirty installation on any system ###
This is not the preferred way to install this class, because it only works
if all the LaTeX files of your thesis are in a single directory, which we
shall call `$XYZ`, and the documentation must be installed separately.
However, if everything else fails or you want to put `$XYZ` on Overleaf,
this is the way to go.

 1) Before installing this package, make sure that the memoir class is
    installed on your TeX system. Check out the documentation of your TeX
    installation to find out how to do this.

 2) Unzip [`kulemt-tex.zip`](kulemt-tex.zip) (from the website)
    into the `$XYZ` directory. 

Optionally, you can install the documentation wherever you want.

 3) Unzip [`kulemt-doc.zip`](kulemt-doc.zip) into your preferred
    documentation directory.


### Proper installation on a Unix like system ###
A Unix like system is any recent TeX installation on a Unix machine, MacOS X,
or Cygwin on Windows. These TeX installations are all based on TeXLive.

 1) Before installing this package, make sure that the memoir class is
    installed on your TeX system. To check this, issue the command
   
        kpsewhich memoir.cls
   
    and check if a file path is echoed.

 2) Find a suitable texmf tree to install in. We'll use `$ROOT` to refer to
    the root directory of that tree.

      * For a system-wide installation:
   
            ROOT=`kpsewhich -var-value=TEXMFLOCAL`

      * For an installation for the current user only:
      
            ROOT=`kpsewhich -var-value=TEXMFHOME`
	  
        Note: some older installations use HOMETEXMF instead of TEXMFHOME.
        If no directory name is returned, use the `texmf` subdirectory of
	    your home directory:
	  
            ROOT="$HOME/texmf"
   
    Make sure this root directory exists:
   
        mkdir -p "$ROOT"

 3) Unzip [`kulemt-tds.zip`](kulemt-tds.zip) (from the website) in `$ROOT`:
   
        unzip -d "$ROOT" kulemt-tds.zip

 4) If you also want to install the sources,
    unzip [`kulemt-src.zip`](kulemt-src.zip) in the appropriate directory:
   
        mkdir -p "$ROOT/source/latex"
        unzip -d "$ROOT/source/latex" kulemt-src.zip

 5) Update the filename database (only needed for system-wide installations):

        mktexlsr "$ROOT"

Note: All documentation is available in the `$ROOT/doc/latex/kulemt` directory.


### Proper installation on MikTeX ###
This also includes MikTeX derived installations such as proTeXt.

Note: The following procedure has only been tested on MikTeX 2.8.
      The MikTeX options can be accessed through the start menu
      MikTeX 2.8 | Maintenance | Settings. For system-wide installations,
      use "Maintenance (Admin)" instead of "Maintenance".

 1) Before installing this package, make sure that you enabled the automatic
    installation of missing packages ("Package installation" on the General
    tab of the MikTeX options)

 2) Find a suitable texmf tree to install in. We'll use `%ROOT%` to refer
    to the root directory of that tree. On the "Roots" tab of the MikTeX
    options, you can select any directory, which is not a MikTeX maintained
    root directory. If you don't find a suitable candidate, create a new
    directory and add it to that tab.

 3) Unzip (using 7-zip or winzip or ...) [`kulemt-tds.zip`](kulemt-tds.zip)
    (from the website) into the `%ROOT%` directory.

 4) If you also want to install the sources, unzip
    [`kulemt-src.zip`](kulemt-src.zip) in the directory `%ROOT%\source\latex` .

 5) Update the filename database by clicking the "Refresh FNDB" button on
    the General tab of the MikTeX options.
    Depending on your previous installations, you might need to update in Admin
    as well as in User mode.
    

Note: All documentation is available in the `%ROOT%\doc\latex\kulemt directory`.
