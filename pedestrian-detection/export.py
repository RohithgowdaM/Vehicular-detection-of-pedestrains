# import pyttsx3

# engine = pyttsx3.init()

# def voice_alert(message):
#     engine.say(message)
#     engine.runAndWait()

# voice_alert("Pedestrian detected on the right")

import geocoder

# Get current location based on IP address
g = geocoder.ip('me')
location = g.latlng
print(f"Latitude: {location[0]}, Longitude: {location[1]}")
