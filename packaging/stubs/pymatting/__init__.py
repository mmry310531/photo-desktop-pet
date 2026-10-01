"""打包用的替身：rembg 一載入就會 import pymatting（只有「alpha matting」功能才用到），
但 pymatting 依賴 numba/llvmlite（約 170MB）。本程式不使用 alpha matting，
所以 exe 版用這個空殼代替，安裝包能小一大截。"""
