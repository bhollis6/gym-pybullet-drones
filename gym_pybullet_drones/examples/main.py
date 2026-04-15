import sys
from drone import Drone

def main():
    drone = Drone()

    
    hiker_located = drone.run_simulation()
    drone.env.close()

    if hiker_located:
        print("Hiker located")
    else:
        print("Could not located hiker")

if __name__ == "__main__":
    sys.exit(main())
