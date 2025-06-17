import time
import json
from PIL import Image, ImageDraw, ImageSequence
import argparse
from gpiozero import PWMLED
import RPi.GPIO as GPIO

# Constants
BAR_WIDTH = 10
BAR_SPACING = 15
MAX_BAR_HEIGHT = 30
CLASS_NAMES = ['Whistle', 'Click', 'BP']
DOLPHIN_SIZE = (42, 42) 

dolphin = Image.open("rotating_dolphin_64x64.gif").convert('1').resize(DOLPHIN_SIZE)

class ClassificationVisualizer:
    def __init__(self, whistle_pin=17, click_pin=27, bp_pin=22):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        self.whistle_pin = whistle_pin
        self.click_pin = click_pin
        self.bp_pin = bp_pin
        self.whistle_led = PWMLED(whistle_pin)
        self.click_led = PWMLED(click_pin)
        self.bp_led = PWMLED(bp_pin)
        
        print("LEDs are turned On!")
        print(f"Using GPIO pins: Whistle={whistle_pin}, Click={click_pin}, BP={bp_pin}")
        
        self.test_leds()
    
    def test_leds(self):
        print("\nTesting LEDs")
        test_leds = {
            'Whistle': self.whistle_led,
            'Click': self.click_led,
            'BP': self.bp_led
        }
        
        for name, led in test_leds.items():
            print(f"\nTesting {name} LED:")
            try:
                print("Fade in")
                for brightness in range(0, 101, 5):
                    led.value = brightness / 100
                    print(f"Brightness: {brightness}%")
                    time.sleep(0.1)
                print("Fade out")
                for brightness in range(100, -1, -5):
                    led.value = brightness / 100
                    print(f"Brightness: {brightness}%")
                    time.sleep(0.1)
                
                led.off()
                print(f"{name} LED Off")
                
            except Exception as e:
                print(f"Error testing {name} LED: {str(e)}")
        
        print("\nLED Test Done!")
    
    def update_lights(self, probabilities):
        try:
            def map_probability(prob):
                if prob < 0.3:  # Below threshold
                    return 0.0
                else:  # Map 0.3-1.0 to 0.0-1.0
                    return (prob - 0.3) / 0.7
            
            self.whistle_led.value = map_probability(probabilities['Whistle'])
            self.click_led.value = map_probability(probabilities['Click'])
            self.bp_led.value = map_probability(probabilities['BP'])
            
            print(f"\nCurrent LED brightness levels:")
            print(f"Whistle LED: {int(map_probability(probabilities['Whistle']) * 100)}% (from {int(probabilities['Whistle'] * 100)}%)")
            print(f"Click LED: {int(map_probability(probabilities['Click']) * 100)}% (from {int(probabilities['Click'] * 100)}%)")
            print(f"BP LED: {int(map_probability(probabilities['BP']) * 100)}% (from {int(probabilities['BP'] * 100)}%)")
            
        except Exception as e:
            print(f"Error updating lights: {str(e)}")
    
    def clear_lights(self):
        self.whistle_led.off()
        self.click_led.off()
        self.bp_led.off()
        print("All LEDs turned off")
    
    def cleanup(self):
        self.clear_lights()
        self.whistle_led.close()
        self.click_led.close()
        self.bp_led.close()
        print("GPIO cleaned")

def create_frame(probabilities, window_num):
    if not DISPLAY_AVAILABLE:
        return None
        
    image = Image.new('1', (device.width, device.height))
    draw = ImageDraw.Draw(image)
    draw.text((device.width - 30, 2), f'W{window_num}', fill=255)
    
    if dolphin:
        image.paste(dolphin, (device.width - DOLPHIN_SIZE[0], device.height - DOLPHIN_SIZE[1]))

    for i, class_name in enumerate(CLASS_NAMES):
        prob = probabilities.get(class_name, 0)
        x = i * (BAR_WIDTH + BAR_SPACING) + 10
        bar_height = int(prob * MAX_BAR_HEIGHT)
        y = device.height - bar_height - 12 
        draw.rectangle(
            [x, y, x + BAR_WIDTH, device.height - 12],
            outline=255,
            fill=255
        )
        percentage = f"{int(prob * 100)}%"
        draw.text((x - 2, y - 10), percentage, fill=255)
        short_label = {
            'Whistle': 'Whs',
            'Click': 'Clk',
            'BP': 'BP'
        }.get(class_name, class_name[:3])
        draw.text((x, device.height - 10), short_label, fill=255)

    return image

def visualize_from_json(json_path):
    print(f"Loading classification data from: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            segment_probs = json.load(f)
    except FileNotFoundError:
        print(f"File {json_path} not found")
        return
    
    visualizer = ClassificationVisualizer()
    
    print("Starting live. Press Ctrl+C to stop.")
    try:
        while True:
            for i, probs in enumerate(segment_probs):
                if DISPLAY_AVAILABLE:
                    frame = create_frame(probs, i + 1)
                    if frame:
                        device.display(frame)
                visualizer.update_lights(probs)
                
                time.sleep(0.125)
    except KeyboardInterrupt:
        print("\nDisplay stopped")
    finally:
        visualizer.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live Results')
    parser.add_argument('json_path', help='Path to the JSON file')
    args = parser.parse_args()
    visualize_from_json(args.json_path)
