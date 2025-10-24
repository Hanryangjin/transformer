from datasets import load_dataset

main_path = '/workspace/LargeData/'

def load_huggingface(data_name):
    dataset = load_dataset(data_name)
    DATA_PATH = main_path + data_name
    dataset.save_to_disk(DATA_PATH)
    
load_huggingface("lcw99/cc100-ko-only")

data_name = "allenai/madlad-400"
madlad_multilang = load_dataset(data_name, languages=["ko"])
DATA_PATH = main_path + data_name
madlad_multilang.save_to_disk(DATA_PATH)