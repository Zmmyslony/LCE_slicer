class Nozzle:
    def __init__(self, speed, fast_speed, line_width, layer_height, extrusion, temperature):
        self.speed = speed
        self.fast_speed = fast_speed
        self.line_width = line_width
        self.layer_height = layer_height
        self.extrusion = extrusion
        self.temperature = temperature


class Printer:
    def __init__(self, nozzle, accuracy=0.04):
        self.accuracy = accuracy
        self.nozzle = nozzle


d23 = Nozzle(40, 1000, 0.32, 0.2, 0.002, 110)
# TODO optimize more nozzles
standard_print = Printer(d23)
