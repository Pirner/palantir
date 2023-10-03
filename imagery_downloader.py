from src.satellite_imagery.data_crawling import SatelliteDataCrawler


def main():
    # Create a new instance of GoogleMap Downloader
    sdc = SatelliteDataCrawler(51.5171, 0.1062, 13)

    print("The tile coorindates are {}".format(sdc.get_x_y_tile()))

    try:
        # Get the high resolution image
        img = sdc.generate_image()
    except IOError:
        print("Could not generate the image - try adjusting the zoom level and checking your coordinates")
    else:
        # Save the image to disk
        img.save("high_resolution_image.png")
        print("The map has successfully been created")


if __name__ == '__main__':
    main()
