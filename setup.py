from optparse import OptionParser
import configparser
import math

parser = OptionParser()
parser.add_option('--space', '-s', dest='space', default='sphere_3',
                  help='Enter space: sphere_3, cylinder_3, torus_3, torus_4')
parser.add_option('--points', '-p', dest='points', default=500,
                  help='Enter number of points to generate')
options, _ = parser.parse_args()
print(options.space)
print(options.points)


cfile = configparser.ConfigParser()
cfile.read("config.ini")

space = options.space.upper()
number_of_points = options.points

lower_x = eval(cfile[space]['LOWER_X'])
upper_x = eval(cfile[space]['UPPER_X'])
lower_y = eval(cfile[space]['LOWER_Y'])
upper_y = eval(cfile[space]['UPPER_y'])
is_plot = eval(cfile[space]['PLOT'])

if space == "TORUS_3":
    lower_r = eval(cfile[space]['LOWER_R'])
    upper_r = eval(cfile[space]['UPPER_R'])

