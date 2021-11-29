import math

METERS_PER_DEGREE = 111300.


class Cartesian:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class LatLng:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

    def get_latlng(self, x, y):
        return LatLng(self.lat + y / METERS_PER_DEGREE, self.lng +
                      x / METERS_PER_DEGREE/math.cos(self.lat / 180.0 * math.pi))

    def get_xy(self, latlng):
        return Cartesian((latlng.lng - self.lng) * METERS_PER_DEGREE *
                         math.cos(self.lat / 180.0 * math.pi), (latlng.lat - self.lat) * METERS_PER_DEGREE)

    def get_distance(self, latlng):
        p = self.get_xy(latlng)
        return math.sqrt(p.x*p.x + p.y*p.y)
