from ClassificationVisualizer import ClassificationVisualizer, create_frame, device, DISPLAY_AVAILABLE
import time

def main():
    print("Starting visualizer test...")
    
    # Create visualizer instance
    visualizer = ClassificationVisualizer()
    
    # Test data - simulate different probability combinations
    test_data = [
        {'Whistle': 1.0, 'Click': 0.0, 'BP': 0.0},  # Only Whistle
        {'Whistle': 0.0, 'Click': 1.0, 'BP': 0.0},  # Only Click
        {'Whistle': 0.0, 'Click': 0.0, 'BP': 1.0},  # Only BP
        {'Whistle': 0.5, 'Click': 0.5, 'BP': 0.5},  # All equal
        {'Whistle': 0.8, 'Click': 0.2, 'BP': 0.0},  # Mix
    ]
    
    try:
        print("\nTesting with different probability combinations...")
        for i, probs in enumerate(test_data):
            print(f"\nTest {i+1}/5")
            print(f"Probabilities: {probs}")
            
            # Update lights
            visualizer.update_lights(probs)
            
            # Update display if available
            if DISPLAY_AVAILABLE:
                frame = create_frame(probs, i + 1)
                if frame:
                    device.display(frame)
                    print("Display updated")
            else:
                print("Display not available")
            
            # Wait to see the effect
            time.sleep(2)
        
        # Clean up
        visualizer.cleanup()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        visualizer.cleanup()
    
    print("\nTest completed")

if __name__ == "__main__":
    main() 