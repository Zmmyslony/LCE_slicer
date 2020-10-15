# TODO write header to gcode
# TODO add rotation of lines
# TODO add translation of lines
# TODO write lines to gcode
# TODO write finisher to gcode
from printer import Printer
import numpy as np
import os


def calculate_distance(current_position, x, y, z):
    return np.sqrt(np.sum(np.array(current_position - [x, y, z]) ** 2))


class GcodeBody:
    content = ""
    current_position = np.array([0, 0, 0])
    moving_mode = 0  # 0 for absolute, 1 for relative

    def __init__(self, printer: Printer, filename):
        self.speed = printer.nozzle.speed
        self.fast_speed = printer.nozzle.fast_speed
        self.extrusion = printer.nozzle.extrusion
        self.set_nozzle_temperature(printer.nozzle.temperature)
        self.wait_until_nozzle_reaches(printer.nozzle.temperature)
        self.file = open("temp.gcode", "w")
        self.filename = filename

    def __del__(self):
        self.save_file(self.filename)

    def write_command(self, command):
        self.file.write(command + "\n")

    def set_nozzle_temperature(self, temp):
        self.write_command("M104 \tS{}".format(temp))

    def wait_until_nozzle_reaches(self, temp):
        self.write_command("M109 \tS{}".format(temp))

    def set_bed_temperature(self, temp):
        self.write_command("M140 \tS{}".format(temp))

    def wait_until_bed_reaches(self, temp):
        self.write_command("M190 \tS{}".format(temp))

    def set_relative_positioning(self):
        self.write_command("G91")
        self.moving_mode = 1

    def set_relative_extruding(self):
        self.write_command("M83")

    def set_absolute_positioning(self):
        self.write_command("G90")
        self.moving_mode = 0
        self.set_relative_extruding()

    def set_current_position_as_coordinates(self, x, y, z):
        self.write_command("G92 \tX{} \tY{} \tZ{]".format(x, y, z))

    def set_current_position_as_start(self):
        self.set_current_position_as_coordinates(0, 0, 0)

    def wait_ms(self, time):
        self.write_command("G04 \tP{}".format(time))

    def move_extruding_absolute(self, x, y, z):
        if self.moving_mode == 0:
            distance = calculate_distance(self.current_position, x, y, z)
            self.write_command("G01 \tX{} \tY{} \tZ{} \tF{} \tE{}".format(x, y, z, self.speed,
                                                                          self.extrusion * distance))
            self.current_position = np.array([x, y, z])
        else:
            print("Error 1: Wrong moving mode selected")

    def move_absolute(self, x, y, z):
        if self.moving_mode == 0:
            self.write_command("G01 \tX{} \tY{} \tZ{} \tF{}".format(x, y, z, self.fast_speed))
            self.current_position = np.array([x, y, z])
        else:
            print("Error 1: Wrong moving mode selected")

    def move_extruding_relative(self, x, y, z):
        if self.moving_mode == 1:
            distance = calculate_distance(np.array([0, 0, 0]), x, y, z)
            self.write_command(
                "G01 \tX{} \tY{} \tZ{} \tF{} \tE{}".format(x, y, z, self.speed, self.extrusion * distance))
            self.current_position += np.array([x, y, z])
        else:
            print("Error 1: Wrong moving mode selected")

    def move_relative(self, x, y, z):
        if self.moving_mode == 1:
            self.write_command("G01 \tX{} \tY{} \tZ{} \tF{}".format(x, y, z, self.fast_speed))
            self.current_position += np.array([x, y, z])
        else:
            print("Error 1: Wrong moving mode selected")

    def save_file(self, filename):
        self.file.close()
        if not os.path.isdir("output"):
            os.mkdir("output")
        os.rename("temp.gcode", "/output/{}.gcode".format(filename))
