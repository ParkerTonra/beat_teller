import mirdata
from tqdm import tqdm

def setup_dataset():
    print("Initializing GiantSteps dataset...")
    dataset = mirdata.initialize('giantsteps_tempo')
    dataset.download()
    
    print("Validating downloaded data...")
    if dataset.validate():
        print("Dataset successfully downloaded and validated!")
    else:
        print("Warning: Dataset validation failed. Some files might be missing.")

if __name__ == "__main__":
    setup_dataset()