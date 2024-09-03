import OpenEXR
import Imath
import numpy as np

def read_exr(file_path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)
    
    # Get the header
    header = exr_file.header()
    
    # Get the data window (this specifies the window of the image)
    data_window = header['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    
    # Define the pixel type for the EXR file
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    # Read the single channel (assuming the channel is named 'Y' for luminance)
    # You may need to change this if your channel has a different name
    channel_data = exr_file.channel('Z', pt)
    
    # Convert the raw string data to a numpy array
    img = np.fromstring(channel_data, dtype=np.float32)
    img.shape = (height, width)  # Reshape the array to the correct dimensions
    
    return img