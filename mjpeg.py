"""Pure JPEG-frame-to-data-URL helper, factored out of doom/server.py so
flappy/server.py doesn't need to import the ViZDoom-coupled server module
just for this. No game state, no vizdoom import.
"""
import base64
import io

from PIL import Image


def encoded_frame(rgb):
    f = io.BytesIO()
    Image.fromarray(rgb).save(f, format='JPEG', quality=75)
    return 'data:image/jpeg;base64,' + base64.b64encode(f.getvalue()).decode()
