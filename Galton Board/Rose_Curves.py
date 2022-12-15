from math import sin
from math import cos
from math import radians

import pygame

# The width and height of our program
(width, height) = (800, 600)

# Setup the window and init the screen surface
screen = pygame.display.set_mode((width, height))

pygame.display.set_caption('Rose Curve in Python !')

# The background color of the screen
screen.fill((250, 250, 205)) # lemonChiffon color

# Draws a rose with n petals and of radius size about `size`
def drawRhodoneaCurve(n, size):
	points =[]
	for i in range(0, 361):
		# The equation of a rhodonea curve
		r = size * sin(radians(n * i))

		# Converting to cartesian co-ordinates
		x = r * cos(radians(i))
		y = r * sin(radians(i))

		list.append(points, (width / 2 + x, height / 2 + y))

	# Draws a set of line segments connected by set of vertices points
	# Also don't close the path and draw it black and set the width to 5
	pygame.draw.aalines(screen, (0, 0, 0), False, points, 2)

def drawPattern():
	# Try changing these values to what you want
	drawRhodoneaCurve(4, 200)


# Our function for drawing the rose pattern
drawPattern()

# Flip the drawn canvas with the newly created canvas (double-buffering)
# Basically we're refreshing the screen surface after drawing
pygame.display.flip()

# Our Main Loop
while True :
	
	# Poll the events in the event queue
	for event in pygame.event.get() :

		# if the user closed the window
		if event.type == pygame.QUIT :

			# deactivate pygame and quit the program
			pygame.quit()
			quit()

		# Draws the surface object to the screen.
		pygame.display.update()

