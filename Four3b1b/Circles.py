from manim import *
import os
# from FourierImage import *
import numpy as np
import itertools as it
from scipy import fft

"""
Idea from Smarter Every Day Fourier, and 3b1b Fourier, 
Code is condensed form of 3b1b -> 
Chanel:
Code:
Made using Manim library() ---> also designed by 3b1b
"""


# change color based on cmap
class Fourier_e_pi(Scene):
    #     "big_radius": 2,
    #
    #     "circle_style": {

    #     "drawn_path_stroke_width": 2,
    # }

    def setup(self):
        # config
        self.vect_conf = {"buff": 0, "max_tip_length_to_length_ratio": 0.35, "tip_length": 0.15,
                          "max_stroke_width_to_length_ratio": 10, "stroke_width": 2}

        self.circle_conf = {"stroke_width": 1}
        self.path_color = YELLOW
        self.param_step = 0.001

        self.vect_num = 200
        self.slow_factor = 0.2

        self.vect_clock = ValueTracker(0)
        self.vect_clock.add_updater(lambda m, dt: m.increment_value(dt * self.slow_factor))
        self.add(self.vect_clock)

        self.sine_x = [4, 0]
        self.sine_y = [0, 4]

        self.colors = [BLUE_D, BLUE_C, BLUE_E, GREY_BROWN]
        self.x_min = 0
        self.y_min = 0

        self.x_max = 250
        self.y_max = 250

    def get_elapsed_time(self):
        return self.vect_clock.get_value()

    def get_color(self):
        return it.cycle(self.colors)

    def get_total_path(self, vectors):
        # path on x, y parametric makes image
        vect_data = {v.coef: v.freq for v in vectors}
        # x = complex_to_R3(vect_data.keys()[0] * np.exp(TAU * 1j * freq * t)
        path = ParametricFunction(lambda t: ORIGIN + np.sum(np.stack([complex_to_R3(coef * np.exp(TAU * 1j * freq * t))
                                                                      for coef, freq in vect_data.items()]), axis=0),
                                  t_min=0, t_max=1, step_size=self.param_step, color=self.path_color)
        return path
    
    # def path_til_now(self, tot_path):
    #     # todo add to glob
    #     c_p = np.array([tot_path.point_from_proportion(t) for t in np.linspace()]) # elaped time
    #
    # def get_time_path(self, vectors):
    #     # hight vs time
    #     # ie y,t
    #     # map circ --> sin, just many
    #     # create line then update high, move x over for t-1 then ad line from pos(t-1) to pos t
    #     # uses path above, just shifts x by t evey t
    #     # todo add dom---range
    #     move_rate = 5
    #     path = self.get_total_path(vectors)
    #     path_func = ParametricFunction(lambda t: (move_rate * LEFT * t, path.function(t)[1] * UP))  # get y
    #     path_func_xt = ParametricFunction(lambda t: (path.function(t)[0] * RIGHT, move_rate * DOWN * t))  # get y
    #     path_func.add_updater(lambda m: m.shift(m.get_left()[0] - self.sine_x))  # moves x
    #     path_func_xt.add_updater(lambda m: m.shift(m.get_top()[1] - self.sine_y))  # moves y
    #     return path_func, path_func_xt
    #
    # def intersect_line(self, curve):  # todo glob?
    #
    #     hor_line = Line(tip.get_center(), curve.get_end())
    #     vert_line = Line([curve.get_end()[0], 0], curve.get_end())  # from 0 at x to y at x, maybe
    #     # call from previops to ignore redefining values
    #     return vert_line, hor_line

    def create_vectors(self, freq, coef):
        four_d = dict(sorted(zip(freq[:self.vect_num + 1], coef[:self.vect_num + 1]),
                             key=lambda item: np.abs(item[1])))

        vectors = VGroup()
        center_tracker = VectorizedPoint(ORIGIN)

        prev_vect = None  # placehold
        for freq, coeff in four_d.items():  # transposed
            if prev_vect:  # already started get end, else or
                vect_start = prev_vect.get_end
            else:
                vect_start = center_tracker.get_location

            vector = self.create_vect(freq.real, coeff, vect_start)  # creates individ vector
            vectors.add(vector)
            prev_vect = vector  # for end
        return vectors

    def create_vect(self, freq, coef, vect_origin):
        vect = Vector(RIGHT, color=GREEN, **self.vect_conf)  # , **self.vector_config)  # start ang
        vect.scale(np.abs(coef))
        # avoid er
        if np.abs(coef) == 0:
            ang = 0
        else:
            ang = np.log(coef).imag  # coef = e^f*i----> f*i = ln(coef) thus f = ln(coef).imag
        vect.rotate_about_origin(ang)

        # setup const
        vect.freq = freq
        vect.coef = coef
        vect.start = vect_origin
        vect.add_updater(self.update_vect)
        return vect

    def update_vect(self, vect):
        time = self.get_elapsed_time()
        coef = vect.coef
        freq = vect.freq
        ang = np.log(coef).imag

        vect.set_length(np.abs(coef))
        vect.set_angle(ang + time * freq * TAU)

        vect.shift(vect.start() - vect.get_start())  # start vect t, start vect t-1: thus the differnce is trans vect
        return vect

    def create_circles(self, vectors):
        # wrapper
        circles = VGroup(*[self.create_circ(vector, color) for vector, color in
                         zip(vectors, self.get_color())])
        # vect is lkist of vect cacl above corespondincg circ
        return circles

    def create_circ(self, vect, color):  # origial rort
        # set vect_origin
        circle = Circle(color=color)
        circle.start = vect.start
        circle.rad = vect.get_length

        circle.add_updater(self.update_circ)
        return circle

    def update_circ(self, circle):  # call circle to update,,,,should alto update sub...moves to pos nect frame
        # hopfully loc pos gets returned, then we can creat v group with all circs and update all,
        # on last follow pat, show orig
        circle.set_width(2 * circle.rad())  # back prop
        circle.move_to(circle.start())


class Image_show(Fourier_e_pi):
    def construct(self):
        # conf
        self.f_name = os.path.join(os.path.dirname(__file__), 'Do_Mayor_armadura.svg')
        self.height = 10
        # self.n_samp = 1000
        run_time = 5  # time of anim
        self.tex = "\\pi"
        path = self.svg_obj()
        # path = self.get_path()
        fre, coef = self.four(path)
        vectors = self.create_vectors(fre, coef)
        circles = self.create_circles(vectors)

        traced_path = self.get_total_path(vectors)
        # path_curr = self.path_til_now(traced_path)
        # wavey, wavex = self.get_time_path(vectors)
        # h_line = always_redraw(
        #     lambda: self.intersect_line(circles, wave)
        # )

        # Why?
        # vectors.update(-1 / self.camera.frame_rate * self.slow_factor)
        #
        self.add(vectors, circles, traced_path, path)  # wavey, wavex)  # , h_line)

        self.wait(run_time)
        
    def svg_obj(self):
        svg = SVGMobject(self.f_name, height=self.height, stroke_width=2, fill=False, fill_opacity=0)
        path = svg.family_members_with_points()[0]
        return path

    def get_path(self):
        tex_mob = TexMobject(self.tex)
        tex_mob.set_height(6)
        path = tex_mob.family_members_with_points()[0]
        path.set_fill(opacity=0)
        path.set_stroke(WHITE, 1)
        return path

    def four(self, path):
        samples = path.points  # np.array([path.point_from_proportion(t) for t in np.linspace(0, 1, self.n_samp)])
        four_samples = samples[:, 0] + 1j * samples[:, 1]
        for_ls = fft.fft(four_samples) / four_samples.size
        four_size = for_ls.size

        four_freq = fft.fftfreq(four_size, 1 / four_size)
        # shift = fft.fftshift(four_freq)
        # four_coefs_pos = for_ls[:four_size//2]
        # four_coefs_neg = for_ls[-(four_size // 2):]
        return four_freq, for_ls
