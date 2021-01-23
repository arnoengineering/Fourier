class Circle:
    def __init__(self, rad, pos):
        self.rad = rad
        self.pos = pos

    def update(self, pos):
        self.pos = pos


class Vector:
    def __init__(self, rad, ang, c_pos):
        self.len = rad
        self.ang = ang
        self.c_pos = c_pos

    def update(self, pos, ang):
        self.c_pos = pos
        self.ang = ang


class EpiCircle:  # maybe superclass both inherit
    def __init__(self, pos, rad, angle):
        self.vec = Vector(rad, angle, pos)
        self.cir = Circle(rad, pos)
        self.pos = pos
        self.angle = angle

    def update(self):
        self.pos = 1  # placehold
        self.angle = (self.angle + 1) % 360  # rad?
        self.cir.update(self.pos)
        self.vec.update(self.pos, self.angle)
