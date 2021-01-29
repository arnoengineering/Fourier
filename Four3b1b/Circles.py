from manim import *
from Four3b1b.FourierImage import *
import numpy as np


# change color based on cmap
class Four(Scene):
    def construct(self):
        self.origin = [0, 0, 0]   # __init or x
        pass

    def list_vec_cir(self):
        origin = self.origin  # maybe redundant
        for fr, amp in four_dict.items():
            self.create_circ_vect(fr, amp, origin)

    def create_circ_vect(self, freq, am, loc_orig):  # origial rort
        dtt = 0  # ph
        amplitude = np.abs(am)  # rad of coeff
        ang = np.tan(am.real, am.imag)
        # set origin
        circle = Circle(radius=am)
        circle.move_to(loc_orig)

        time_offset = 0  # freq dt; time_off delta t
        # init dot
        loc_dot = Dot()
        # percent of way arrount
        loc_dot.move_to(circle.point_from_proportion(ang / (2 * np.pi)))

        def move_vect():  # add vect
            vect = Line(circle.get_center(), loc_dot.get_center())
            return vect

        def move_tip(mob, dt):
            global loc_orig
            time_offset += (dt * freq)
            mob.move_to((circle.point_from_proportion(0)))
            loc_orig = mob.get_center()  # for next

        loc_dot.add_updater(move_tip)

        loc_vect = always_redraw(move_vect)  # move along path, always rotate  # final line
        self.add(loc_dot, loc_vect)
