import pygame
from PIL import Image
from PIL.PngImagePlugin import PngInfo

def screenshot(screen_surface, appstate):
    filename = "screenshot.png"
    # TODO - add timestamp
    pygame.image.save(screen_surface, filename)
    # add metadata
    metadata_info = appstate.get_info_table()
    targetImage = Image.open(filename)
    metadata = PngInfo()
    for key, value in metadata_info.items():
        print(f"Adding metadata {key}:{value}")
        metadata.add_text(key, f"{value}")
    targetImage.save(filename, pnginfo=metadata)
    print(f"Saved screenshot to {filename}")

def load_metada(filename, appstate):
    print(f"Loading metadata from {filename}")
    srcImage = Image.open(filename)
    srcImage.load()
    info_table = srcImage.info
    print(f"Metadata info_table: {info_table}")
    appstate.set_from_info_table(info_table)
    return info_table