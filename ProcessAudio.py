import argparse
from ModelTesting import classify_wav

def process_audio_to_json(track_path, output_json, temperature=1.0):
    print(f"Processing track: {track_path}")
    predictions, avg_probs = classify_wav(track_path, temperature)
    print(f"Results saved to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio file and save classification results')
    parser.add_argument('track_path', help='Path to the audio track file')
    parser.add_argument('--output', '-o', default='classification_results.json',
                      help='Output JSON file path (default: classification_results.json)')
    parser.add_argument('--temperature', '-t', type=float, default=1.0,
                      help='Temperature for classification (default: 1.0)')
    
    args = parser.parse_args()
    process_audio_to_json(args.track_path, args.output, args.temperature) 