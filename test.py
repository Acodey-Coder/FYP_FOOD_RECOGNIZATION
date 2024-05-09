# import requests
#
# # Replace 'your_image.jpg' with the actual path to the image file z
# image_path = './cup.jfif'
# files = {'image': open(image_path, 'rb')}
#
# # Replace 'http://127.0.0.1:5000/predict' with the correct URL of your Flask API
# # url = 'http://127.0.0.1:5000/predict'

import requests
from tkinter import Tk, filedialog

# Create a Tkinter root window (it will be hidden)
root = Tk()
root.withdraw()

# Ask the user to select a file using a file dialog
file_path = filedialog.askopenfilename(title="Select an image file")

# Check if the user selected a file
if file_path:
    # Prepare the files dictionary for the request
    files = {'image': open(file_path, 'rb')}

    # Replace 'http://127.0.0.1:5000/predi`ct' with the correct URL of your FastAPI API
    url = 'http://3.111.15.46:5000/predict'  # Note: Correct the port if it's different
    # url='http://18.138.121.3:5000/predict'

    # Send a POST request
    response = requests.post(url, files=files)

    print(response.json())
else:
    print("No file selected.")


