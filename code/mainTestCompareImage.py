from numpy import *
from Pgm import *

path = "letra a/0.pgm"
pgm = Pgm(path,20)
pgm.read_pgm_file()
var1 = pgm.resultMatrix

path1 = "letra a/3.pgm"
pgm1 = Pgm(path1,20)
pgm1.read_pgm_file()
pgm1.get_matches_between_images(var1);
