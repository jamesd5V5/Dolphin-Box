from gpiozero import PWMLED
import time

def main():
    print("Starting LED test...")
    
    # Create PWM LED instances with the same pins as before
    whistle_led = PWMLED(17)  # GPIO17
    click_led = PWMLED(27)    # GPIO27
    bp_led = PWMLED(22)       # GPIO22
    
    print("LEDs initialized")
    
    try:
        # Fade in and out twice
        for cycle in range(2):
            print(f"\nCycle {cycle + 1}/2")
            
            # Fade in
            print("Fading in...")
            for brightness in range(0, 101, 5):
                value = brightness / 100
                whistle_led.value = value
                click_led.value = value
                bp_led.value = value
                print(f"Brightness: {brightness}%")
                time.sleep(0.1)
            
            # Fade out
            print("Fading out...")
            for brightness in range(100, -1, -5):
                value = brightness / 100
                whistle_led.value = value
                click_led.value = value
                bp_led.value = value
                print(f"Brightness: {brightness}%")
                time.sleep(0.1)
        
        # Turn all LEDs off
        print("\nTurning all LEDs OFF...")
        whistle_led.off()
        click_led.off()
        bp_led.off()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    print("Test completed")

if __name__ == "__main__":
    main() 