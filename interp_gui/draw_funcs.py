# Colors
LIGHT_BLUE    = "#0092CB"
SKY_BLUE      = "#2375DB"
CHARCOAL      = "#1E1E1E"
DARK_GREY     = "#282828"
LIGHT_GREY    = "#3C3C3C"

PASTEL_RED    = "#E80531"
PASTEL_BLUE   = "#3843FF"
PASTEL_GREEN  = "#46D150"
BABY_BLUE     = "#90CDFF"
PASTEL_YELLOW = "#F9FF11"


# Unit conversions

# Inches to centimeters
def in2cm(dim):
    return dim*2.54

# Centimeters to inches
def cm2in(dim):
    return dim/2.54

# Feet to meters
def ft2m(dim):
    return dim*0.3048

# Meters to Feet
def m2ft(dim):
    return dim*3.2808399

# Meters to Pixels
# 200 pixels per meter
def m2pix(m):
    return m*200.0

# Pixels to Meters
# 1/200 meters per pixel
def pix2m(p):
    return p/200.0


# Viewport Transformations

# Input in cm
# Scale by the zoom level and translate by the origin
def world2screen_x(canvas, x):
    return m2pix(x/100.)*canvas.zl + canvas.origin[0]

# Input in cm
# Invert the y-axis, scale by the zoom level, and translate by the origin
def world2screen_y(canvas, y):
    return m2pix(-y/100.)*canvas.zl + canvas.origin[1]

# Output in cm
# Undo translation, then undo scaling.
def screen2world_x(canvas, x):
    return 100*pix2m( (x - canvas.origin[0])/canvas.zl )
    
# Output in cm
# Undo translation, then undo scaling, then undo the inversion of the y-axis
def screen2world_y(canvas, y):
    return 100*pix2m( -(y - canvas.origin[1])/canvas.zl )
