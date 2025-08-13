import argparse

def main():
    parser = argparse.ArgumentParser(description='Download YouTube video and extract frames')
    parser.add_argument('--config', '-c', 
                       default='inference/data-configs/data_config_alcaraz.txt',
                       help='Path to configuration file (default: inference/data-configs/data_config_alcaraz.txt)')
    
    args = parser.parse_args()

    print("Inference TODO")
if __name__ == "__main__":
    main()