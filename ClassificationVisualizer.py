import time
import json
from PIL import Image, ImageDraw, ImageSequence
import argparse
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

# Initialize display
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial)

# Constants
BAR_WIDTH = 10
BAR_SPACING = 15
MAX_BAR_HEIGHT = 30
CLASS_NAMES = ['Whistle', 'Click', 'BP']
DOLPHIN_SIZE = (42, 42) 

dolphin = Image.open("rotating_dolphin_64x64.gif").convert('1').resize(DOLPHIN_SIZE)

def create_frame(probabilities, window_num):
    image = Image.new('1', (device.width, device.height))
    draw = ImageDraw.Draw(image)
    draw.text((device.width - 30, 2), f'W{window_num}', fill=255)
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
        print(f"Error: File {json_path} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: {json_path} is not a valid JSON file")
        return
    
    print("Starting live visualization. Press Ctrl+C to stop.")
    try:
        while True:
            for i, probs in enumerate(segment_probs):
                frame = create_frame(probs, i + 1)
                device.display(frame)
                time.sleep(0.125)
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live OLED display of classification results')
    parser.add_argument('json_path', help='Path to the JSON file containing classification results')
    args = parser.parse_args()
    visualize_from_json(args.json_path)
