# Recognising prompts and how to exit

## I don't recognise my prompt...where am I?

`GNU nano` is at the top of your window? You're in `nano`.

Your window looks like the following? You're in `vi`.

    ~
    ~
    ~    
    ~
    "somefilename" ...


Something like `somefilename  (Fundamental) ----` is at the bottom of your window? You're in `emacs` or `xemacs`.

`XEmacs: somefilename  (Fundamental) ----` is at the bottom of your window? You're in `xemacs`.

Your prompt is `>>>` ? You're in `python`.

Your prompt is `In [123]:` ? You're in `ipython`.

`--More--(...%)` is at the bottom-left of your window? You're in `more`.

`Manual page...` is at the bottom-left of your window? You're in a `man` page.

`:` is at the bottom-left of your window? You're in `less` or a `man` page.

## How do I exit from...

* `nano`
 * Press `CTRL-X`
 * If you have unsaved changes, you will be asked to save these - press `y` to save, or `n` to quit without saving.
* `vi`
 * Type `:q!` to exit without saving.
 * If this text just appears on screen then press `ESC` then type `:q!`
* `emacs` or `xemacs`
 * Press `CTRL-X CTRL-C`. 
 * If you have unsaved changes, you will be asked to save these - press `y` to save, or `n` then type `yes` to quit without saving.
* `python`
 * Type `exit()` or `CTRL-D`
* `ipython`
 * Type `exit()`, or `CTRL-D` then press `y`
* `man` page
 * Press `q`
* `more` or `less`
 * Press `q`
