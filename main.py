import os.path

#from facer.facer import create_average_face_from_directory
from facer.facer_v2 import create_average_face_from_directory_v2

if __name__ == "__main__":
  dir_in = "Images"
  dir_out = "Results"

  if not os.path.exists(dir_out):
      os.makedirs(dir_out)
  #create_average_face_from_directory(dir_in, dir_out, "face_average")
  create_average_face_from_directory_v2(dir_in, dir_out, "face_average")