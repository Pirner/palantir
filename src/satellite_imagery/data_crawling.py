import urllib.request
from PIL import Image
import os
import math


class SatelliteDataCrawler:
    def __init__(self, lat, lng, zoom=12):
        """
        class for crawling satellite image data
        :param lat: latitude to crawl
        :param lng: longitude to crawl
        :param zoom: zoom level (0 - 23)
        """
        self.lat = lat
        self.lng = lng
        self.zoom = zoom

        self.tile_size = 256

    def get_x_y_tile(self):
        """
        generates an x, y tile coordinate
        :return:
        """
        n_tiles = 1 << self.zoom
        # Find the x_point given the longitude
        point_x = (self.tile_size / 2 + self.lng * self.tile_size / 360.0) * n_tiles // self.tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self.lat * (math.pi / 180.0))

        # calculate the y coordinate
        point_y = ((self.tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
                    self.tile_size / (2 * math.pi))) * n_tiles // self.tile_size

        return int(point_x), int(point_y)

    def generate_image(self, **kwargs):
        """
        generate image
        :return:
        """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 5)
        tile_height = kwargs.get('tile_height', 5)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None:
            start_x, start_y = self.get_x_y_tile()

        # Determine the size of the image
        width, height = 256 * tile_width, 256 * tile_height

        # Create a new image of the size require
        map_img = Image.new('RGB', (width, height))

        for x in range(0, tile_width):
            for y in range(0, tile_height):
                url = 'https://mt0.google.com/vt?x=' + str(start_x + x) + '&y=' + str(start_y + y) + '&z=' + str(
                    self.zoom)

                current_tile = str(x) + '-' + str(y)
                data = urllib.request.urlretrieve(url, current_tile)
                # urllib.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x * 256, y * 256))

                os.remove(current_tile)

        return map_img
